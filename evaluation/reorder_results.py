"""Reorganize a results folder into results_reordered/{model}/{instrument(s)}/[alpha_X]/file.

Original layout: results/<piece_folder>/<various files>.wav|.json
New layout:       results_reordered/<model>/<instrument_or_pair>/[alpha_<value>/]<original filename>

Also creates a "flattened" subfolder inside each model's folder
(results_reordered/<model>/flattened) containing all of that model's files
directly, with the source piece folder name prefixed to avoid filename
collisions.

Usage:
    python reorder_results.py <src_results_dir> <dst_results_dir>
"""

import argparse
import re
import shutil
from pathlib import Path

MODEL = r"(?:from_scratch|finetuned_from_\d+k_for_\d+k)"

PATTERNS = [
    # interpolated_output_/interpolated_weights_ + pair + model + alpha + reverb
    re.compile(
        rf"interpolated_(?:output|weights)_(?P<i1>[a-z]+)_(?P<i2>[a-z]+)_"
        rf"(?P<model>{MODEL})_alpha_(?P<alpha>\d+)_(?P<reverb>with|without)_reverb\.wav$"
    ),
    # pair models.json
    re.compile(rf"_(?P<i1>[a-z]+)_(?P<i2>[a-z]+)_(?P<model>{MODEL})_models\.json$"),
    # single-instrument synthesis wav
    re.compile(rf"_(?P<inst>[a-z]+)_(?P<model>{MODEL})\.wav$"),
]


def classify(filename):
    for pattern in PATTERNS:
        m = pattern.search(filename)
        if not m:
            continue
        g = m.groupdict()
        model = g["model"]
        if "i1" in g:
            # keep original order: alpha's meaning depends on which instrument is first
            pair = f"{g['i1']}_{g['i2']}"
        else:
            pair = g["inst"]
        alpha_dir = None
        if g.get("alpha") is not None:
            alpha_dir = f"alpha_{g['alpha']}"
        return model, pair, alpha_dir
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()

    unmatched = []
    count = 0

    for piece_dir in sorted(args.src.iterdir()):
        if not piece_dir.is_dir():
            continue
        for f in sorted(piece_dir.iterdir()):
            if not f.is_file():
                continue
            result = classify(f.name)
            if result is None:
                unmatched.append(f)
                continue
            model, pair, alpha_dir = result
            parts = [args.dst, model, pair]
            if alpha_dir:
                parts.append(alpha_dir)
            out_dir = Path(*parts)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f.name

            flat_dst = args.dst / model / "flattened"
            flat_dst.mkdir(parents=True, exist_ok=True)
            flat_path = flat_dst / f"{piece_dir.name}__{f.name}"

            if args.move:
                shutil.copy2(f, flat_path)
                shutil.move(str(f), str(out_path))
            else:
                shutil.copy2(f, out_path)
                shutil.copy2(f, flat_path)
            count += 1

    print(f"Processed {count} files.")
    print(f"Flattened copies written to {args.dst}/<model>/flattened")
    if unmatched:
        print(f"WARNING: {len(unmatched)} files did not match any pattern:")
        for f in unmatched:
            print(f"  {f}")


if __name__ == "__main__":
    main()
