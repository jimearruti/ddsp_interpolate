"""Copy dataset files into train/ and test/ folders according to a split JSON.

The split JSON maps instrument -> {"train": [...], "val": [...], "test": [...]}
with each entry a path like "../data/URMP/Dataset/<piece>/<recording>.wav"
relative to wherever the JSON was generated. Only the last two path
components (piece folder + filename) are used, joined onto source_root, so
you can point source_root at any directory that mirrors the
piece/recording.wav layout. This script flattens all train files into
<dest>/train/ and all test files into <dest>/test/, prefixing each filename
with its instrument code to avoid collisions. The 'val' split, if present,
is skipped. Train and test files are additionally copied into per-instrument
folders <dest>/train_<instrument>/ and <dest>/test_<instrument>/, alongside
(not inside) train/ and test/. For every pair of instruments, one folder is
also created per split at <dest>/train_<instA>_<instB>/ and
<dest>/test_<instA>_<instB>/ (instruments sorted alphabetically, so a given
pair always maps to a single folder regardless of order), containing the
files from both instruments.

Usage:
    python reorder_dataset_per_split.py <split_json> <source_root> <dest_root>
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_json", type=Path, help="Path to split_files.json")
    parser.add_argument("source_root", type=Path, help="Source root folder")
    parser.add_argument("dest_root", type=Path, help="Destination root folder")
    args = parser.parse_args()

    with open(args.split_json) as f:
        splits = json.load(f)

    instruments = sorted(splits.keys())

    counts = {"train": 0, "test": 0}
    for instrument, sets in splits.items():
        for split_name, paths in sets.items():
            if split_name not in ("train", "test"):
                continue
            out_dir = args.dest_root / split_name
            out_dir.mkdir(parents=True, exist_ok=True)

            instrument_dir = args.dest_root / f"{split_name}_{instrument}"
            instrument_dir.mkdir(parents=True, exist_ok=True)

            pair_dirs = []
            for other in instruments:
                if other == instrument:
                    continue
                pair_name = "_".join(sorted((instrument, other)))
                pair_dir = args.dest_root / f"{split_name}_{pair_name}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                pair_dirs.append(pair_dir)

            for rel_path in paths:
                piece_and_file = Path(rel_path).parts[-2:]
                src = args.source_root.joinpath(*piece_and_file)
                if not src.exists():
                    print(f"Warning: missing file {src}")
                    continue
                dst = out_dir / f"{instrument}_{src.name}"
                shutil.copy2(src, dst)
                shutil.copy2(src, instrument_dir / src.name)
                for pair_dir in pair_dirs:
                    shutil.copy2(src, pair_dir / f"{instrument}_{src.name}")
                counts[split_name] += 1

    print(f"Done. Copied: {counts}")


if __name__ == "__main__":
    main()
