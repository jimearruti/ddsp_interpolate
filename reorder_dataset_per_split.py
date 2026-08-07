"""Copy dataset files into train/ and test/ folders according to a split JSON.

The split JSON maps instrument -> {"train": [...], "val": [...], "test": [...]}
with each entry a relative path to a .wav file. This script flattens all
train files into <dest>/train/ and all test files into <dest>/test/,
prefixing each filename with its instrument code to avoid collisions.

Usage:
    python split_copy.py <split_json> <dest_root> [--source-root DIR]
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_json", type=Path, help="Path to split_files.json")
    parser.add_argument("dest_root", type=Path, help="Destination root folder")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Base directory the paths in the JSON are relative to "
        "(default: the split_json's own directory)",
    )
    args = parser.parse_args()

    source_root = args.source_root or args.split_json.parent
    with open(args.split_json) as f:
        splits = json.load(f)

    counts = {"train": 0, "val": 0, "test": 0}
    for instrument, sets in splits.items():
        for split_name, paths in sets.items():
            if split_name not in ("train", "val", "test"):
                continue
            out_dir = args.dest_root / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for rel_path in paths:
                src = (source_root / rel_path).resolve()
                if not src.exists():
                    print(f"Warning: missing file {src}")
                    continue
                dst = out_dir / f"{instrument}_{src.name}"
                shutil.copy2(src, dst)
                counts[split_name] += 1

    print(f"Done. Copied: {counts}")


if __name__ == "__main__":
    main()
