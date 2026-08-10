"""Reorganize a results folder into:

results_reordered/<model>/resynthesis/<instrument>/<file>
results_reordered/<model>/resynthesis/all/<file>
results_reordered/<model>/<output|weights>/<instrument_pair>/alpha_<value>/<file>   (one folder per alpha value)
results_reordered/<model>/<output|weights>/<instrument_pair>/intermediate_alphas/<file>   (all alphas except 0/100, grouped)
results_reordered/<model>/<output|weights>/unordered_pairs/<sorted_pair>/intermediate_alphas/<pair>__<file>
results_reordered/<model>/<output|weights>/extremes/<instrument>/<pair>__<file>
results_reordered/<model>/<output|weights>/all/<pair>__<file>

Rules:
- Files with "_sweep_" in the name are skipped entirely.
- Only "with_reverb" interpolation files are kept; "without_reverb" files are skipped.
- Every alpha value gets its own folder; alpha values other than 0/100 are additionally
  grouped together under "intermediate_alphas".
- "unordered_pairs" merges both orderings of a pair (e.g. fl_tpt and tpt_fl) into one
  folder keyed by the sorted instrument names, for intermediate alphas only.
- "extremes" groups alpha_0/alpha_100 files by the pure instrument they correspond to
  (e.g. fl_tpt's alpha_0 and tpt_fl's alpha_100 are both pure fl, so both land under
  extremes/fl), regardless of which pair/order produced them.
- "all" collects every file for that category (resynthesis, output, or weights) into a
  single flat folder, alongside the other groupings.

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
            pair = f"{g['i1']}_{g['i2']}"
            method = g["method"]
            reverb = g["reverb"]
            alpha = g["alpha"]
            i1, i2 = g["i1"], g["i2"]
        else:
            pair = g["inst"]
            method = None
            reverb = None
            alpha = None
            i1, i2 = None, None
        return model, method, pair, alpha, reverb, i1, i2
    return None


def extreme_instrument(alpha, i1, i2):
    if alpha == "0":
        return i1
    if alpha == "100":
        return i2
    return None


def is_intermediate(alpha):
    return alpha not in ("0", "100")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()

    unmatched = []
    skipped_without_reverb = 0
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
            model, method, pair, alpha, reverb, i1, i2 = result

            if method is not None:
                # interpolation file
                if reverb == "without":
                    skipped_without_reverb += 1
                    continue

                alpha_folder = f"alpha_{alpha}"

                out_dir = args.dst / model / method / pair / alpha_folder
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f.name

                all_dir = args.dst / model / method / "all"
                all_dir.mkdir(parents=True, exist_ok=True)
                extra_paths = [all_dir / f"{pair}__{f.name}"]
                if is_intermediate(alpha):
                    inter_dir = args.dst / model / method / pair / "intermediate_alphas"
                    inter_dir.mkdir(parents=True, exist_ok=True)
                    extra_paths.append(inter_dir / f.name)

                    sorted_pair = "_".join(sorted(pair.split("_")))
                    unordered_dir = (
                        args.dst / model / method / "unordered_pairs" / sorted_pair / "intermediate_alphas"
                    )
                    unordered_dir.mkdir(parents=True, exist_ok=True)
                    extra_paths.append(unordered_dir / f"{pair}__{f.name}")
                else:
                    instrument = extreme_instrument(alpha, i1, i2)
                    extremes_dir = args.dst / model / method / "extremes" / instrument
                    extremes_dir.mkdir(parents=True, exist_ok=True)
                    extra_paths.append(extremes_dir / f"{pair}__{f.name}")

                if args.move:
                    for p in extra_paths:
                        shutil.copy2(f, p)
                    shutil.move(str(f), str(out_path))
                else:
                    shutil.copy2(f, out_path)
                    for p in extra_paths:
                        shutil.copy2(f, p)
            else:
                # single-instrument resynthesis file
                out_dir = args.dst / model / "resynthesis" / pair
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f.name

                all_dir = args.dst / model / "resynthesis" / "all"
                all_dir.mkdir(parents=True, exist_ok=True)
                all_path = all_dir / f.name

                if args.move:
                    shutil.copy2(f, all_path)
                    shutil.move(str(f), str(out_path))
                else:
                    shutil.copy2(f, out_path)
                    shutil.copy2(f, all_path)

            count += 1

    print(f"Processed {count} files.")
    if skipped_without_reverb:
        print(f"Skipped {skipped_without_reverb} without_reverb files.")
    if unmatched:
        print(f"WARNING: {len(unmatched)} files did not match any pattern:")
        for f in unmatched:
            print(f"  {f}")


if __name__ == "__main__":
    main()
