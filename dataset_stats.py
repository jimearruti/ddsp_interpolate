import datetime
import json
import pathlib
import os

import numpy as np
import soundfile as sf
import yaml
from effortless_config import Config

from ddsp.core import mean_std_loudness
from train_base_model import make_dataloaders


def get_duration(f):
    info = sf.info(str(f))
    return info.frames / info.samplerate

def analyze_dataset(split_data):
    instruments = split_data.keys()
    duration_summary = {}

    for instrument in instruments:
        duration_summary[instrument] = {}

        for split, files in split_data[instrument].items():
            total_seconds = sum(get_duration(f) for f in files if pathlib.Path(f).exists())
            duration_summary[instrument][split] = str(datetime.timedelta(seconds=int(total_seconds)))

    return duration_summary


def print_duration_summary(duration_summary):
    print("\n" + "=" * 60)
    print(f"{'Instrument':<20} | {'Split':<10} | {'Duration':<15}")
    print("=" * 60)

    for inst, splits in duration_summary.items():
        for split, dur in splits.items():
            print(f"{inst:<20} | {split:<10} | {dur:<15}")

    print("=" * 60)


def get_train_stats_for_dataset(config, batch, split_dataset):
    out_dir = pathlib.Path(config["preprocess"]["out_dir"])
    instruments = split_dataset.keys()
    
    dataloaders = make_dataloaders(
        config["preprocess"]["out_dir"], instruments, batch
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
                "instruments_considered": list(instruments),
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

    args.parse_args()

    with open(args.CONFIG, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    out_dir = pathlib.Path(config["preprocess"]["out_dir"])

    split_file = os.path.join(out_dir, "split_files.json")
    if not os.path.exists(split_file):
        print(f"[WARNING] No split_files.json")
        return

    with open(split_file) as f:
        split_data = json.load(f)

    duration_summary = analyze_dataset(split_data)
    print_duration_summary(duration_summary)
    get_train_stats_for_dataset(config, args.BATCH, split_data)


if __name__ == "__main__":
    main()