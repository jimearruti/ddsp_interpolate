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
    ''''Log evaluation metrics to wandb, save a checkpoint if it's the best so far, and save an audio sample of the reconstruction.'''
    audio = torch.cat([signal, reconstructed_signal], -1).reshape(-1).detach().cpu().numpy()

    log_dict = {
        "mean_loss": mean_loss,
        "audio": wandb.Audio(audio, sample_rate=config["preprocess"]["sampling_rate"]),
    }
    if val_loss is not None:
        log_dict["val_loss"] = val_loss

    wandb.log(log_dict, step=step)
                    
    sf.write(
        save_path / f"eval_{step:06d}.wav",
        audio,
        config["preprocess"]["sampling_rate"],
    )

    loss_to_track = val_loss if val_loss is not None else mean_loss
    if loss_to_track < best_loss:
        best_loss = loss_to_track
        torch.save(model.state_dict(), save_path / "state.pth")

    if not step % 5000:
        torch.save(model.state_dict(), save_path / f"state_{step}.pth")
    
    return best_loss


def train_step(model, batch, opt, scheduler, mean_loudness, std_loudness, config, device):
    '''Perform a training step: compute the loss, backpropagate, and update the model parameters. Return the loss, gradient norm, and reconstructed signal for logging.'''
    signal, pitch, loudness = batch
    signal = signal.to(device)
    pitch = pitch.unsqueeze(-1).to(device)
    loudness = loudness.unsqueeze(-1).to(device)

    loudness = (loudness - mean_loudness) / std_loudness

    reconstructed_signal = model(pitch, loudness).squeeze(-1)

    loss = spectral_loss(signal, reconstructed_signal, config["train"]["scales"], config["train"]["overlap"])
    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    opt.step()
    scheduler.step()

    return loss, grad_norm, reconstructed_signal


def evaluate(model, dataloader, mean_loudness, std_loudness, config, device):
    '''Return the average loss on the dataloader, or None if dataloader is None (e.g. no validation set).'''
    if dataloader is None:
        return None

    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            signal, pitch, loudness = batch
            signal = signal.to(device)
            pitch = pitch.unsqueeze(-1).to(device)
            loudness = loudness.unsqueeze(-1).to(device)
            loudness = (loudness - mean_loudness) / std_loudness

            reconstructed_signal = model(pitch, loudness).squeeze(-1)

            total_loss += spectral_loss(signal, reconstructed_signal, config["train"]["scales"], config["train"]["overlap"]).item()
            n += 1

    model.train()
    return total_loss / n if n > 0 else float("inf")


def train(model, dataloaders, opt, scheduler, config, save_path, device, total_steps):
    train_dataloader, val_dataloader, _ = dataloaders

    mean_loudness, std_loudness = mean_std_loudness(train_dataloader)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    mean_loudness = torch.tensor(mean_loudness, device=device)
    std_loudness = torch.tensor(std_loudness, device=device)

    best_loss = float("inf")
    best_step = 0
    mean_loss = torch.zeros(1, device=device)
    n_element = 0
    step = 0
    epochs = int(np.ceil(total_steps / len(train_dataloader)))

    for e in tqdm(range(epochs)):
        for batch in train_dataloader:
            loss, grad_norm, reconstructed_signal = train_step(
                model, batch, opt, scheduler, mean_loudness,
                std_loudness, config, device
            )

            mean_loss += loss.detach()
            n_element += 1
            step += 1

            if not step % 100:
                log_step(model, loss, grad_norm, step, e, opt)

            if not step % 1000:
                mean_loss_val = (mean_loss / n_element).item()
                mean_loss = torch.zeros(1, device=device)
                n_element = 0

                val_loss = evaluate(model, val_dataloader, mean_loudness, std_loudness, config, device)
 
                signal = batch[0].to(device)

                loss_to_track = val_loss if val_loss is not None else mean_loss_val
                if loss_to_track < best_loss:
                    best_step = step
                    
                best_loss = log_checkpoint(
                    model, signal, reconstructed_signal, mean_loss_val, val_loss, best_loss, step, save_path, config
                )

    config["train"]["best_step"] = best_step
    config["train"]["best_loss"] = best_loss