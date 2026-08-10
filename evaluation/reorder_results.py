"""Reorganize a results folder into
results_reordered/{model}/[output|weights]/{instrument(s)}/[alpha_X]/[with|without_reverb]/file.

Original layout: results/<piece_folder>/<various files>.wav|.json
New layout (single-instrument synthesis):
    results_reordered/<model>/<instrument>/<original filename>
New layout (interpolation, output-space or weight-space):
    results_reordered/<model>/<output|weights>/<instrument_pair>/alpha_<value>/<with|without>_reverb/<original filename>

Files with "_sweep_" in the name are skipped entirely.

Also creates a "flattened" subfolder inside each model's folder
(results_reordered/<model>/flattened) containing all of that model's files
directly, with the source piece folder name prefixed to avoid filename
collisions.

For interpolation files, also creates an "all_alphas" subfolder per pair
(results_reordered/<model>/<output|weights>/<instrument_pair>/all_alphas/<with|without>_reverb)
gathering every alpha for that pair/method/reverb combination together,
so all audios for a given interpolation can be browsed regardless of alpha.

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
        rf"interpolated_(?P<method>output|weights)_(?P<i1>[a-z]+)_(?P<i2>[a-z]+)_"
        rf"(?P<model>{MODEL})_alpha_(?P<alpha>\d+)_(?P<reverb>with|without)_reverb\.wav$"
    ),
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
            method = g["method"]
            reverb = f"{g['reverb']}_reverb"
        else:
            pair = g["inst"]
            method = None
            reverb = None
        alpha_dir = None
        if g.get("alpha") is not None:
            alpha_dir = f"alpha_{g['alpha']}"
        return model, method, pair, alpha_dir, reverb
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
            if not f.is_file() or f.suffix != ".wav":
                continue
            if "_sweep_" in f.name:
                continue
            result = classify(f.name)
            if result is None:
                unmatched.append(f)
                continue
            model, method, pair, alpha_dir, reverb = result
            parts = [args.dst, model]
            if method:
                parts.append(method)
            parts.append(pair)
            if alpha_dir:
                parts.append(alpha_dir)
            if reverb:
                parts.append(reverb)
            out_dir = Path(*parts)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f.name

            flat_dst = args.dst / model / "flattened"
            flat_dst.mkdir(parents=True, exist_ok=True)
            flat_path = flat_dst / f"{piece_dir.name}__{f.name}"

            all_alphas_path = None
            if method and reverb and alpha_dir:
                pair_key = "_".join(sorted(pair.split("_")))
                all_alphas_dir = args.dst / model / method / pair_key / "all_alphas" / reverb
                all_alphas_dir.mkdir(parents=True, exist_ok=True)
                all_alphas_path = all_alphas_dir / f.name

            if args.move:
                shutil.copy2(f, flat_path)
                if all_alphas_path:
                    shutil.copy2(f, all_alphas_path)
                shutil.move(str(f), str(out_path))
            else:
                shutil.copy2(f, out_path)
                shutil.copy2(f, flat_path)
                if all_alphas_path:
                    shutil.copy2(f, all_alphas_path)
            count += 1

    print(f"Processed {count} files.")
    print(f"Flattened copies written to {args.dst}/<model>/flattened")
    if unmatched:
        print(f"WARNING: {len(unmatched)} files did not match any pattern:")
        for f in unmatched:
            print(f"  {f}")


if __name__ == "__main__":
    main()
