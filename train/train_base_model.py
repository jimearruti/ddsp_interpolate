import os
import pathlib
from datetime import datetime

import torch
import wandb
import yaml
from dotenv import load_dotenv
from effortless_config import Config
from torch.utils.data import ConcatDataset

from ddsp.model import DDSP
from .preprocess import DatasetMultiInstrument
from .train_utils import train


# training config
class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    STEPS = 30000
    BATCH = 16
    START_LR = 1e-3
    GAMMA = 0.98
    STEP_SIZE =10000
    INSTRUMENT = None
    BASE_MODEL_PATH = None


def build_subset_dataset(base_dir, subset, instruments):
    '''
    Build a dataset for a specific subset (train, val, test) by concatenating datasets for each instrument.
    Args:
        base_dir (str): The base directory where the preprocessed data is stored.
        subset (str): The subset of the dataset to build ('train', 'val', or 'test').
        instruments (list): A list of instrument names to include in the dataset.
    Returns:
        ConcatDataset: A concatenated dataset containing data for the specified subset and instruments.
    '''
    datasets = []
    for instrument in instruments:
        instrument_dir = pathlib.Path(base_dir) / instrument / subset
        if instrument_dir.exists():
            datasets.append(DatasetMultiInstrument(base_dir, instrument, subset=subset))
    if not datasets:
        return None
    return ConcatDataset(datasets)


def make_dataloaders(out_dir, instruments, batch_size):
    '''
    Create dataloaders for the training, validation, and test subsets by concatenating datasets for each instrument.
    Args:
        out_dir (str): The base directory where the preprocessed data is stored.
        instruments (list): A list of instrument names to include in the datasets.
        batch_size (int): The batch size for the dataloaders.
    Returns:
        tuple: A tuple containing the train, validation, and test dataloaders.
    '''
    train_dataset = build_subset_dataset(out_dir, "train", instruments)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True
    )

    val_dataloader = None
    # create the validation dataloader if the instrument has a validation split
    val_dataset  = build_subset_dataset(out_dir, "val", instruments)
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
    
    test_dataloader = None
    # create the test dataloader if the instrument has a test split
    test_dataset = build_subset_dataset(out_dir, "test", instruments)
    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

    return train_dataloader, val_dataloader, test_dataloader
      

def main():
    '''Main training loop'''
    args.parse_args()

    # model config
    with open(args.CONFIG, "r") as config_file:
        config = yaml.safe_load(config_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get the list of instruments to train on, either from the command line argument or from the config file
    instruments = config["data"]["instruments"]
    save_path = pathlib.Path(args.ROOT) / args.NAME / timestamp / "base_model"
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize the DDSP model and move it to device
    model = DDSP(**config["model"]).to(device)
    # Save initial state
    torch.save(model.state_dict(), save_path / "state_000000.pth")

    # Load the base model weights if a path is provided
    if args.BASE_MODEL_PATH is not None:
        model.load_state_dict(torch.load(args.BASE_MODEL_PATH, map_location=device))

    dataloaders = make_dataloaders(
        config["preprocess"]["out_dir"], instruments, args.BATCH
    )

    # Initialize a Weights & Biases run for experiment tracking, logging the configuration and instrument name
    run = wandb.init(
        project=args.NAME,
        name="base_model",
        config={**config, "instrument": "base_model"},
    )

    # Set up the optimizer and learning rate scheduler for training the model
    opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.STEP_SIZE, gamma=args.GAMMA)

    # Train the model
    train(model, dataloaders, opt, scheduler, config, save_path, device, args.STEPS)

    # Save the configuration used for this training run to the save_path directory for future reference
    with open(save_path / "config.yaml", "w") as out_config:
        yaml.safe_dump(config, out_config)
    # Finish the Weights & Biases run after training is complete
    run.finish()

if __name__ == "__main__":
    # Load environment variables from a .env file
    load_dotenv()
    # log in to Weights & Biases using the API key from the environment variables
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    main()