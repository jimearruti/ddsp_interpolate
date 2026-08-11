import csv
import json
import logging
import os
import sys

from nas_eval.common import load_audio_files
from nas_eval.timbre.mmd import compute_mfcc_features, MMD

from .resynth_path_parsing import CSV_FIELDS, parse_resynth_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def compute_mmd(background_path, resynth_path, sample_rate):
    background_audio = list(load_audio_files(background_path, sample_rate).values())
    resynth_audio = list(load_audio_files(resynth_path, sample_rate).values())

    background_features = compute_mfcc_features(background_audio, sample_rate)
    resynth_features = compute_mfcc_features(resynth_audio, sample_rate)

    mmd = MMD()
    return mmd(background_features, resynth_features).item()


def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python -m evaluation.mmd <config.json>")
        exit(1)

    config_path = sys.argv[1]; del sys.argv[1]
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

    csv_path = config.get("csv_path", "mmd_results.csv")
    sample_rate = config.get("sample_rate", 16000)

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for resynth_path in resynth_paths:
            if not os.path.exists(resynth_path):
                logging.warning(f"Resynth path does not exist: {resynth_path}")
                continue

            distance = compute_mmd(background_path, resynth_path, sample_rate)
            logging.info(f"MMD for {resynth_path}: {distance:.4f}")

            row = parse_resynth_path(resynth_path)
            row["distance"] = distance
            writer.writerow(row)
            csv_file.flush()

    logging.info(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
