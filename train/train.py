import os
import pathlib
from datetime import datetime

import torch
import wandb
import yaml
from dotenv import load_dotenv
from effortless_config import Config

from ddsp.model import DDSP
from .preprocess import DatasetMultiInstrument
from .train_utils import load_split_data, train

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


def make_dataloaders(out_dir, instrument, batch_size, split_data):
    """Return (train, val, test) dataloaders, or (train, None, None) if split=False.
    Args:
        out_dir (str): Path to the output directory containing the preprocessed data.
        instrument (str): The instrument for which to create the dataloaders.
        batch_size (int): The batch size for the dataloaders.
        split_data (dict): A dictionary containing the split data for each instrument.
    Returns:
        tuple: A tuple containing the train, validation, and test dataloaders.
    """
    # create the train dataset, type multi instrument dataset
    train_dataset = DatasetMultiInstrument(out_dir, instrument, subset="train")
    # ceate the train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True
    )

    instrument_split = split_data.get(instrument, {})
    
    val_dataloader = None
    # create the validation dataloader if the instrument has a validation split
    if instrument_split.get("val"):
        val_dataset = DatasetMultiInstrument(out_dir, instrument, subset="val")
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, drop_last=False)

    test_dataloader = None
    # create the test dataloader if the instrument has a test split
    if instrument_split.get("test"):
        test_dataset = DatasetMultiInstrument(out_dir, instrument, subset="test")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)

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
    instruments = (
        [args.INSTRUMENT] 
        if args.INSTRUMENT 
        else config["data"]["instruments"]
    )

    split_data = load_split_data(config["preprocess"]["out_dir"])

    for instrument in instruments:
        # Create a directory for saving the model and logs for this instrument
        save_path = pathlib.Path(args.ROOT) / args.NAME / timestamp / instrument
        save_path.mkdir(parents=True, exist_ok=True)

        # Initialize the DDSP model with the configuration and move it to the appropriate device (GPU or CPU)
        model = DDSP(**config["model"]).to(device)

        # Load the base model weights if a path is provided
        if args.BASE_MODEL_PATH is not None:
            model.load_state_dict(torch.load(args.BASE_MODEL_PATH, map_location=device))

        dataloaders = make_dataloaders(
            config["preprocess"]["out_dir"], instrument, args.BATCH, split_data
        )

        # Initialize a Weights & Biases run for experiment tracking, logging the configuration and instrument name
        run = wandb.init(
            project=args.NAME,
            name=instrument,
            config={**config, "instrument": instrument},
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