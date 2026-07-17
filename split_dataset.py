import pathlib
import json

import numpy as np
import yaml
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


def split_files(files, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split at the piece level so the same piece never appears in
    more than one of train/val/test."""
    pieces = sorted({get_piece(f) for f in files})

    rng = np.random.default_rng(seed)
    pieces = list(rng.permutation(pieces))

    n = len(pieces)
    n_test = max(1, int(test_ratio * n))
    n_val = max(1, int(val_ratio * n))

    test_pieces = set(pieces[:n_test])
    val_pieces = set(pieces[n_test:n_test + n_val])
    train_pieces = set(pieces[n_test + n_val:])

    if not train_pieces:
        raise ValueError(
            f"No pieces left for train split (only {n} unique pieces "
            f"for this instrument) — reduce val/test ratios or check filtering."
        )

    train = [f for f in files if get_piece(f) in train_pieces]
    val = [f for f in files if get_piece(f) in val_pieces]
    test = [f for f in files if get_piece(f) in test_pieces]

    return train, val, test


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

    for instrument in instruments:

        files_instrument = [f for f in files if f.stem.split("_")[2] == instrument]

        train_files, val_files, test_files = split_files(files_instrument)

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


if __name__ == "__main__":
    main()