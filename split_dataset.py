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


def build_piece_split_map(files, val_ratio=0.1, test_ratio=0.1, seed=42, n_trials=200, check_instruments=()):
    """Assign each piece to train/val/test, searching over random trials
    to balance file-level ratios for the instruments in `check_instruments`
    (other instruments still get split, just don't influence the search)."""
    piece_instrument_counts = {}
    for f in files:
        piece = get_piece(f)
        instrument = f.stem.split("_")[2]
        piece_instrument_counts.setdefault(piece, {})
        piece_instrument_counts[piece][instrument] = piece_instrument_counts[piece].get(instrument, 0) + 1

    pieces = sorted(piece_instrument_counts.keys())
    total_counts = {
        inst: sum(piece_instrument_counts[p].get(inst, 0) for p in pieces)
        for inst in check_instruments
    }

    n = len(pieces)
    n_test = 0 if test_ratio == 0 else max(1, int(test_ratio * n))
    n_val = 0 if val_ratio == 0 else max(1, int(val_ratio * n))

    rng = np.random.default_rng(seed)
    best_split, best_score = None, None

    for _ in range(n_trials):
        perm = list(rng.permutation(pieces))
        test_pieces = perm[:n_test]
        val_pieces = perm[n_test:n_test + n_val]
        train_pieces = perm[n_test + n_val:]

        score = 0.0
        for inst in check_instruments:
            total = total_counts[inst]
            if total == 0:
                continue
            test_count = sum(piece_instrument_counts[p].get(inst, 0) for p in test_pieces)
            val_count = sum(piece_instrument_counts[p].get(inst, 0) for p in val_pieces)
            score += (test_count / total - test_ratio) ** 2
            score += (val_count / total - val_ratio) ** 2

        if best_score is None or score < best_score:
            best_score = score
            best_split = (train_pieces, val_pieces, test_pieces)

    train_pieces, val_pieces, test_pieces = best_split
    piece_split = {}
    for p in test_pieces:
        piece_split[p] = "test"
    for p in val_pieces:
        piece_split[p] = "val"
    for p in train_pieces:
        piece_split[p] = "train"
    return piece_split


def split_files(files, piece_split):
    """Assign files to train/val/test using a pre-computed piece -> split
    mapping. All instruments are split the same way."""
    train = [f for f in files if piece_split.get(get_piece(f)) == "train"]
    val = [f for f in files if piece_split.get(get_piece(f)) == "val"]
    test = [f for f in files if piece_split.get(get_piece(f)) == "test"]

    if not train:
        raise ValueError("No files left for train split — reduce val/test ratios or check the piece_split mapping.")

    return train, val, test


def check_split_ratios(split_data, val_ratio, test_ratio, check_instruments, tolerance=0.1):
    for instrument in check_instruments:
        subsets = split_data.get(instrument, {})
        n_total = sum(len(subsets.get(s, [])) for s in ("train", "val", "test"))
        if n_total == 0:
            continue
        actual_val = len(subsets.get("val", [])) / n_total
        actual_test = len(subsets.get("test", [])) / n_total
        if abs(actual_val - val_ratio) > tolerance:
            print(f"Warning: '{instrument}' val ratio {actual_val:.2f} vs target {val_ratio:.2f}")
        if abs(actual_test - test_ratio) > tolerance:
            print(f"Warning: '{instrument}' test ratio {actual_test:.2f} vs target {test_ratio:.2f}")


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

    check_instruments = ("vn", "fl", "tpt")
    val_ratio = config["preprocess"].get("val_ratio", 0.1)
    test_ratio = config["preprocess"].get("test_ratio", 0.1)

    piece_split = build_piece_split_map(
        files, val_ratio=val_ratio, test_ratio=test_ratio, seed=42,
        n_trials=200, check_instruments=check_instruments
    )

    for instrument in instruments:
        files_instrument = [f for f in files if f.stem.split("_")[2] == instrument]
        train_files, val_files, test_files = split_files(files_instrument, piece_split)

        split_data[instrument] = {
            "train": [str(file) for file in train_files],
            "val": [str(file) for file in val_files],
            "test": [str(file) for file in test_files],
        }

    root_out_path.mkdir(parents=True, exist_ok=True)
    with open(root_out_path / "split_files.json", "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    check_split_ratios(split_data, val_ratio, test_ratio, check_instruments)

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