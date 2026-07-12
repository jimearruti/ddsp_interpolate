import csv
import datetime
import pathlib
import numpy as np
import yaml
from effortless_config import Config
from ddsp.core import mean_std_loudness
from train_base_model import make_dataloaders


def analyze_dataset(config):
    out_dir = pathlib.Path(config["preprocess"]["out_dir"])
    instruments = config["data"]["instruments"]
    sampling_rate = config["preprocess"]["sampling_rate"]

    duration_summary = {}

    print("Analyzing preprocessed data...")
    
    for instrument in instruments:
        duration_summary[instrument] = {}

        for split in ["train", "val", "test"]:

            signal_file = out_dir / instrument / split / "signals.npy"
            
            if not signal_file.exists():
                print(f"[WARNING] No signals.npy found for {instrument} at {signal_file}")
                continue

            signals_mmap = np.load(signal_file, mmap_mode="r")
            total_samples = signals_mmap.size

            total_seconds = total_samples / sampling_rate
            duration_formatted = str(datetime.timedelta(seconds=int(total_seconds)))
            
            duration_summary[instrument][split] = duration_formatted

    print("\n" + "=" * 60)
    print(f"{'Instrument':<20} | {'Split':<10} | {'Duration':<15}")
    print("=" * 60)

    for inst, splits in duration_summary.items():
        for split, dur in splits.items():
            print(f"{inst:<20} | {split:<10} | {dur:<15}")

    print("=" * 60)


def get_stats_for_dataset(config, batch, split_dataset):
    out_dir = pathlib.Path(config["preprocess"]["out_dir"])
    instruments = config["data"]["instruments"]
    
    dataloaders = make_dataloaders(
        config["preprocess"]["out_dir"], instruments,
        batch, split=split_dataset
    )

    train_dataloader, _, _ = dataloaders
    mean_loudness, std_loudness = mean_std_loudness(train_dataloader)

    print(f"Mean loudness: {mean_loudness}, Std loudness: {std_loudness}")

    zscore_file = out_dir / "mean_std_loudness.yml"
    
    with open(zscore_file, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "mean_loudness": float(mean_loudness),
                "std_loudness": float(std_loudness),
                "instruments_considered": instruments,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    
    print(f"Saved z-score stats to: {zscore_file}")


def main():
    class args(Config):
        CONFIG = "config.yaml"
        BATCH = 16
        SPLIT_DATASET = True

    args.parse_args()

    with open(args.CONFIG, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    analyze_dataset(config)
    get_stats_for_dataset(config, args.BATCH, args.SPLIT_DATASET)



if __name__ == "__main__":
    main()