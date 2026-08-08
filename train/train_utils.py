import json
import pathlib

import numpy as np
import soundfile as sf
import torch
import wandb
from tqdm import tqdm

from ddsp.core import mean_std_loudness
from ddsp.utils import spectral_loss


def load_split_data(out_dir):
    '''
    Load the split data from a JSON file in the specified output directory.
    Args:
        out_dir (str or pathlib.Path): The directory where the split data JSON file is located.
    Returns:
        dict: A dictionary containing the split data for each instrument.
    '''
    with open(pathlib.Path(out_dir) / "split_files.json", "r", encoding="utf-8") as f:
        return json.load(f)


def log_step(model, loss, grad_norm, step, epoch, opt):
    '''Log training metrics to wandb.'''
    wandb.log({
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "lr": opt.param_groups[0]["lr"],
        "reverb_decay": torch.nn.functional.softplus(-model.reverb.decay).item() * 500,
        "reverb_wet": torch.sigmoid(model.reverb.wet).item(),
        "epoch": epoch,
    }, step=step)


def log_checkpoint(model, signal, reconstructed_signal, mean_loss, val_loss, best_loss, step, save_path, config):
    '''Log evaluation metrics to wandb, save a checkpoint if it's the best so far, 
    and save an audio sample of the reconstruction.
    Args:
        model (torch.nn.Module): The model being trained.
        signal (torch.Tensor): The original audio signal.
        reconstructed_signal (torch.Tensor): The reconstructed audio signal from the model.
        mean_loss (float): The mean training loss over the last evaluation period.
        val_loss (float or None): The validation loss, if available.
        best_loss (float): The best loss observed so far.
        step (int): The current training step.
        save_path (pathlib.Path): The directory where checkpoints and audio samples will be saved.
        config (dict): A dictionary containing configuration information.
    Returns:
        float: The updated best loss after evaluating the current step.
    '''

    # Concatenate the original and reconstructed signals for logging and save them as a WAV file
    audio = torch.cat([signal, reconstructed_signal], -1).reshape(-1).detach().cpu().numpy()

    # Log the mean loss, validation loss (if available), and the audio sample to wandb
    log_dict = {
        "mean_loss": mean_loss,
        "audio": wandb.Audio(audio, sample_rate=config["preprocess"]["sampling_rate"]),
    }
    if val_loss is not None:
        log_dict["val_loss"] = val_loss

    wandb.log(log_dict, step=step)

    # Save the audio sample to a WAV file in the specified save path                
    sf.write(
        save_path / f"eval_{step:06d}.wav",
        audio,
        config["preprocess"]["sampling_rate"],
    )

    # Determine which loss to track for saving the best model checkpoint
    loss_to_track = val_loss if val_loss is not None else mean_loss

    # If the current loss is better than the best loss so far, update the best loss and save the model state
    if loss_to_track < best_loss:
        best_loss = loss_to_track
        torch.save(model.state_dict(), save_path / "state.pth")

    # Save the model state every 5000 steps regardless of whether it's the best loss or not
    if not step % 5000:
        torch.save(model.state_dict(), save_path / f"state_{step}.pth")

    # Return the best loss for tracking purposes
    return best_loss


def train_step(model, batch, opt, scheduler, mean_loudness, std_loudness, config, device):
    '''
    Perform a training step: compute the loss, backpropagate, and update the model parameters. 
    Return the loss, gradient norm, and reconstructed signal for logging.
    Args:
        model (torch.nn.Module): The model to be trained.
        batch (tuple): A batch of data containing signal, pitch, and loudness.
        opt (torch.optim.Optimizer): The optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        mean_loudness (float): The mean loudness for normalization.
        std_loudness (float): The standard deviation of loudness for normalization.
        config (dict): A dictionary containing configuration information.
        device (torch.device): The device to run the computations on (CPU or GPU).
    Returns:
        tuple: A tuple containing the loss, gradient norm, and reconstructed signal.
    '''
    # Unpack the batch
    signal, pitch, loudness = batch
    # move the data to the specified device
    signal = signal.to(device)
    # Add a trailing feature dimension expected by the model, then move to device
    pitch = pitch.unsqueeze(-1).to(device)
    loudness = loudness.unsqueeze(-1).to(device)

    # Standardize loudness (zero mean, unit variance) using precomputed stats
    loudness = (loudness - mean_loudness) / std_loudness

    # Run the model and drop the trailing feature dim to match signal shape
    reconstructed_signal = model(pitch, loudness).squeeze(-1)

    # Multi-scale spectral loss between original and reconstructed signal
    loss = spectral_loss(signal, reconstructed_signal, config["train"]["scales"], config["train"]["overlap"])
    # Backward pass
    opt.zero_grad()
    loss.backward()
    # computes the gradient norm for logging purposes
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

    # update model parameters
    opt.step()
    # update the learning rate according to the scheduler
    scheduler.step()

    return loss, grad_norm, reconstructed_signal


def evaluate(model, dataloader, mean_loudness, std_loudness, config, device):
    '''
    Return the average loss on the dataloader, or None if dataloader is None (e.g. no validation set).
    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The dataloader for the evaluation dataset
        mean_loudness (float): The mean loudness for normalization.
        std_loudness (float): The standard deviation of loudness for normalization.
        config (dict): A dictionary containing configuration information.
        device (torch.device): The device to run the computations on (CPU or GPU).
    Returns:
        float or None: The average loss over the evaluation dataset, or None if dataloader is None.
    '''
    if dataloader is None:
        return None
    # Set the model to evaluation mode and disable gradient computation for efficiency
    model.eval()

    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            # unpack the batch and move to device
            signal, pitch, loudness = batch
            signal = signal.to(device)
            # Add a trailing feature dimension expected by the model, then move to device
            pitch = pitch.unsqueeze(-1).to(device)
            loudness = loudness.unsqueeze(-1).to(device)
            # standardize loudness (zero mean, unit variance) using precomputed stats
            loudness = (loudness - mean_loudness) / std_loudness
            # run the model and drop the trailing feature dim to match signal shape
            reconstructed_signal = model(pitch, loudness).squeeze(-1)
            # Accumulate loss and batch count
            total_loss += spectral_loss(signal, reconstructed_signal, config["train"]["scales"], config["train"]["overlap"]).item()
            n += 1
    # Set the model back to training mode after evaluation
    model.train()
    # Return the average loss over the evaluation dataset, or infinity if no batches were evaluated
    return total_loss / n if n > 0 else float("inf")


def train(model, dataloaders, opt, scheduler, config, save_path, device, total_steps):
    '''
    Train the model using the provided dataloaders, optimizer, and scheduler.
    Args:
        model (torch.nn.Module): The model to be trained.
        dataloaders (tuple): A tuple containing the training, validation, and test dataloaders.
        opt (torch.optim.Optimizer): The optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        config (dict): A dictionary containing configuration information.
        save_path (pathlib.Path): The directory where checkpoints and audio samples will be saved.
        device (torch.device): The device to run the computations on (CPU or GPU).
        total_steps (int): The total number of training steps to perform.
    Returns:
        None: The function saves the best model checkpoint and logs training metrics to wandb.
    '''
    # Unpack the dataloaders for training, validation, and test sets
    train_dataloader, val_dataloader, _ = dataloaders

    # Get the mean and standard deviation of loudness for the training dataset
    mean_loudness, std_loudness = mean_std_loudness(train_dataloader)
    # Store the mean and standard deviation of loudness in the config dictionary for later use
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    # Convert mean and std loudness to torch tensors and move them to the specified device
    mean_loudness = torch.tensor(mean_loudness, device=device)
    std_loudness = torch.tensor(std_loudness, device=device)

    best_loss = float("inf")
    best_step = 0
    mean_loss = torch.zeros(1, device=device)
    n_element = 0
    step = 0
    # Calculate the number of epochs needed to reach the total_steps based on the length of the training dataloader
    epochs = int(np.ceil(total_steps / len(train_dataloader)))

    for e in tqdm(range(epochs)):
        for batch in train_dataloader:
            # Perform a training step and get the loss, gradient norm, and reconstructed signal
            loss, grad_norm, reconstructed_signal = train_step(
                model, batch, opt, scheduler, mean_loudness,
                std_loudness, config, device
            )
            # Accumulate the mean loss and count the number of elements for averaging
            mean_loss += loss.detach()
            n_element += 1
            step += 1

            # Log training metrics to wandb every 100 steps
            if not step % 100:
                log_step(model, loss, grad_norm, step, e, opt)

            # Evaluate the model on the validation set and log checkpoints every 1000 steps
            if not step % 1000:
                mean_loss_val = (mean_loss / n_element).item()
                mean_loss = torch.zeros(1, device=device)
                n_element = 0
                val_loss = evaluate(model, val_dataloader, mean_loudness, std_loudness, config, device)
                
                signal = batch[0].to(device)
                # Determine which loss to track for saving the best model checkpoint
                loss_to_track = val_loss if val_loss is not None else mean_loss_val
                if loss_to_track < best_loss:
                    best_step = step
                # Log the evaluation metrics and save a checkpoint if it's the best so far
                best_loss = log_checkpoint(
                    model, signal, reconstructed_signal, mean_loss_val, val_loss, best_loss, step, save_path, config
                )

    # Store the best step and best loss in the config dictionary for later use
    config["train"]["best_step"] = best_step
    config["train"]["best_loss"] = best_loss