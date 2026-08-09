# Adapted from https://github.com/fcaspe/BRAVE/blob/main/evaluation/scripts/fad.py

import csv
import json
import logging
import os
import re
import sys

from frechet_audio_distance import FrechetAudioDistance


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Matches: .../output/vn_tpt/alpha_50/with_reverb  or  .../weights/tpt_fl/alpha_0/with_reverb
ALPHA_RE = re.compile(
    r"/(output|weights)/([a-z]+_[a-z]+)/alpha_(\d+)/with_reverb/?$"
)


def parse_row(reference, path, distance):
    """Turn a resynth path + its FAD score into a (reference, pair, method, alpha, distance) row."""
    basename = os.path.basename(path.rstrip("/"))

    # Ground-truth test set, e.g. .../rearranged/test_vn
    if basename.startswith("test_"):
        return (reference, "baseline_test", "-", "-", distance)

    # Alpha-sweep interpolation output, e.g. .../output/vn_tpt/alpha_50/with_reverb
    match = ALPHA_RE.search(path)
    if match:
        method, pair, alpha = match.groups()
        return (reference, pair, method, alpha, distance)

    # Plain resynthesis, e.g. .../finetuned_from_30k_for_30k/vn
    return (reference, "resynthesis", "-", "-", distance)


def write_csv(rows, csv_path):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["reference", "pair", "method", "alpha", "distance"])
        writer.writerows(rows)


def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python script.py <config.json>")
        exit(1)

    config_path = sys.argv[1]; del sys.argv[1]
    # Load config file
    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        exit(1)

    background_path = config.get("background_path")
    resynth_paths = config.get("resynth_paths", [])
    if not background_path or not resynth_paths:
        logging.error("Configuration file must specify a 'background_path' and a list of 'resynth_paths'.")
        exit(1)

    # Reference instrument for this run, inferred from e.g. ".../test_vn" -> "vn"
    reference = os.path.basename(background_path.rstrip("/")).replace("test_", "")

    csv_path = config.get("csv_path", "distances.csv")

    modname = "vggish"
    # Initialize FrechetAudioDistance
    frechet = FrechetAudioDistance(
        model_name=modname,
        sample_rate=16000,  # VGGish resamples files to 16kHz
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    # Compute distances for each resynth path
    rows = []
    for resynth_path in resynth_paths:
        if not os.path.exists(resynth_path):
            logging.warning(f"Resynth path does not exist: {resynth_path}")
            continue

        distance = frechet.score(
            background_path,
            resynth_path,
            background_embds_path=background_path + f"/bkg_embeddings_{modname}.npy",
            eval_embds_path=resynth_path + f"/resynth_embeddings_{modname}.npy",
            dtype="float32",
        )
        logging.info(f"Distance for {resynth_path}: {distance:.2f}")
        rows.append(parse_row(reference, resynth_path, round(distance, 2)))

    write_csv(rows, csv_path)
    logging.info(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()