import datetime
import json
import pathlib
import yaml

import numpy as np
import soundfile as sf
from effortless_config import Config


def get_files(data_location: str, extension: str, **_) -> list[pathlib.Path]:
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def get_piece(f: pathlib.Path) -> str:
    """Piece name is the second underscore-delimited token of the
    containing folder, e.g. '01_Jupiter_vn_vc' -> 'Jupiter'."""
    return f.parent.name.split("_")[1]


def count_split_files(split_data: dict) -> dict:
    """Count number of files per instrument per split, e.g.
    {'vn': {'train': 12, 'val': 2, 'test': 2}, ...}"""
    counts = {}
    for instrument, subsets in split_data.items():
        counts[instrument] = {
            subset: len(files) for subset, files in subsets.items()
        }
    return counts


def print_split_counts(counts: dict) -> None:
    header = f"{'instrument':<12}{'train':>8}{'val':>8}{'test':>8}{'total':>8}"
    print(header)
    print("-" * len(header))
    for instrument, subset_counts in counts.items():
        total = sum(subset_counts.values())
        print(
            f"{instrument:<12}"
            f"{subset_counts.get('train', 0):>8}"
            f"{subset_counts.get('val', 0):>8}"
            f"{subset_counts.get('test', 0):>8}"
            f"{total:>8}"
        )


def build_piece_split_map(files, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Assign each unique piece to exactly one split, so the same piece
    can never land in different splits for different instruments."""
    pieces = sorted({get_piece(f) for f in files})

    rng = np.random.default_rng(seed)
    pieces = list(rng.permutation(pieces))

    n = len(pieces)
    n_test = max(1, int(test_ratio * n))
    n_val = max(1, int(val_ratio * n))

    piece_split = {}
    for p in pieces[:n_test]:
        piece_split[p] = "test"
    for p in pieces[n_test:n_test + n_val]:
        piece_split[p] = "val"
    for p in pieces[n_test + n_val:]:
        piece_split[p] = "train"

    return piece_split


def split_files(files, instrument, piece_split, split_instruments=("vn", "fl", "tpt")):
    """Assign files to train/val/test using a pre-computed, shared
    piece -> split mapping. Instruments outside split_instruments
    go entirely to train.
    """
    if instrument not in split_instruments:
        return files, [], []

    train = [f for f in files if piece_split.get(get_piece(f)) == "train"]
    val = [f for f in files if piece_split.get(get_piece(f)) == "val"]
    test = [f for f in files if piece_split.get(get_piece(f)) == "test"]

    if not train:
        raise ValueError(
            f"No files left for train split for instrument '{instrument}' "
            f"— reduce val/test ratios or check the piece_split mapping."
        )

    return train, val, test
    

def get_duration(f):
    info = sf.info(str(f))
    return info.frames / info.samplerate


def calculate_durations(split_data):
    instruments = split_data.keys()
    duration_summary = {}

    for instrument in instruments:
        duration_summary[instrument] = {}
        split_data_instrument = split_data[instrument]

        for split, files in split_data_instrument.items():
            total_seconds = sum(get_duration(f) for f in files if pathlib.Path(f).exists())
            duration_summary[instrument][split] = str(datetime.timedelta(seconds=int(total_seconds)))
    
    return duration_summary


def _parse_hhmmss(s: str) -> int:
    """Parse an 'H:MM:SS' string (as produced by str(datetime.timedelta))
    back into total seconds."""
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def print_durations(duration_summary: dict) -> None:
    header = f"{'instrument':<12}{'train':>12}{'val':>12}{'test':>12}{'total':>12}"
    print(header)
    print("-" * len(header))
    for instrument, splits in duration_summary.items():
        train = splits.get("train", "0:00:00")
        val = splits.get("val", "0:00:00")
        test = splits.get("test", "0:00:00")

        total_seconds = sum(_parse_hhmmss(s) for s in (train, val, test))
        total = str(datetime.timedelta(seconds=total_seconds))

        print(
            f"{instrument:<12}"
            f"{train:>12}"
            f"{val:>12}"
            f"{test:>12}"
            f"{total:>12}"
        )


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()

    with open(args.CONFIG, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    files = get_files(**config["data"])

    instruments = config["data"]["instruments"]
    root_out_path = pathlib.Path(config["preprocess"]["out_dir"])
    split_data = {}

    split_instruments = ("vn", "fl", "tpt")

    split_instrument_files = [
        f for f in files if f.stem.split("_")[2] in split_instruments
    ]
    piece_split = build_piece_split_map(split_instrument_files, val_ratio=0.1, test_ratio=0.1, seed=42)

    for instrument in instruments:
        files_instrument = [f for f in files if f.stem.split("_")[2] == instrument]

        train_files, val_files, test_files = split_files(
            files_instrument, instrument, piece_split, split_instruments=split_instruments
        )

        split_data[instrument] = {
            "train": [str(file) for file in train_files],
            "val": [str(file) for file in val_files],
            "test": [str(file) for file in test_files],
        }

    root_out_path.mkdir(parents=True, exist_ok=True)
    with open(root_out_path / "split_files.json", "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    counts = count_split_files(split_data)
    print_split_counts(counts)
    with open(root_out_path / "split_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    duration_summary = calculate_durations(split_data)
    print_durations(duration_summary)
    with open(root_out_path / "split_durations.json", "w", encoding="utf-8") as f:
        json.dump(duration_summary, f, indent=2)


if __name__ == "__main__":
    main()