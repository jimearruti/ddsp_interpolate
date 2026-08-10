# Adapted from https://github.com/fcaspe/BRAVE/blob/main/evaluation/scripts/fad.py

import csv
import json
import logging
import os
import sys
from pathlib import Path

from frechet_audio_distance import FrechetAudioDistance


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

CSV_FIELDS = [
    "resynth_path", "model", "category", "method", "grouping",
    "pair", "instrument", "alpha", "distance",
]


def parse_resynth_path(path):
    """Pull structured fields out of a results_reordered path.

    Paths outside results_reordered (e.g. raw test data) get all fields
    empty except resynth_path.
    """
    fields = {k: "" for k in CSV_FIELDS}
    fields["resynth_path"] = path

    parts = Path(path).parts
    if "results_reordered" not in parts:
        return fields

    rel = parts[parts.index("results_reordered") + 1:]
    if len(rel) < 2:
        return fields

    model, category, rest = rel[0], rel[1], rel[2:]
    fields["model"] = model
    fields["category"] = category

    if category == "resynthesis":
        if rest:
            fields["instrument"] = rest[0]
        return fields

    # category is "output" or "weights"
    fields["method"] = category
    if not rest:
        return fields

    head = rest[0]
    if head == "all":
        fields["grouping"] = "all"
    elif head == "extremes":
        fields["grouping"] = "extremes"
        if len(rest) > 1:
            fields["instrument"] = rest[1]
    elif head == "unordered_pairs":
        fields["grouping"] = "unordered_pairs"
        if len(rest) > 1:
            fields["pair"] = rest[1]
        if len(rest) > 2:
            fields["alpha"] = rest[2]
    else:
        fields["pair"] = head
        if len(rest) > 1:
            fields["alpha"] = rest[1]

    return fields


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

    csv_path = config.get("csv_path", "fad_results.csv")

    modname = "vggish"
    # Initialize FrechetAudioDistance
    frechet = FrechetAudioDistance(
        model_name=modname,
        sample_rate=16000,  # VGGish resamples files to 16kHz
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    # Compute distances for each resynth path, writing results as we go
    # so a crash partway through doesn't lose earlier scores.
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

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

            row = parse_resynth_path(resynth_path)
            row["distance"] = distance
            writer.writerow(row)
            csv_file.flush()

    logging.info(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
