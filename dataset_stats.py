import csv
import datetime
import pathlib
import numpy as np
import yaml
from effortless_config import Config
from ddsp.core import mean_std_loudness
from train_base_model import make_dataloaders


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()

    with open(args.CONFIG, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

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

    dataloaders = make_dataloaders(
        config["preprocess"]["out_dir"], instruments,
        args.BATCH, split=args.SPLIT_DATASET
    )

    train_dataloader, _, _ = dataloaders
    mean_loudness, std_loudness = mean_std_loudness(train_dataloader)

    print(f"Mean loudness: {mean_loudness}, Std loudness: {std_loudness}")

    # Instruments actually considered (present in preprocessed data)
    considered_instruments = [inst for inst, splits in duration_summary.items() if len(splits) > 0]

    zscore_file = out_dir / "mean_std_loudness.csv"
    with open(zscore_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_loudness", "std_loudness", "instruments_considered"])
        writer.writerow([
            float(mean_loudness),
            float(std_loudness),
            ";".join(considered_instruments),
        ])

    print(f"Saved z-score stats to: {zscore_file}")


if __name__ == "__main__":
    main()