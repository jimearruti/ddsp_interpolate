# Adapted from https://github.com/fcaspe/BRAVE/blob/main/evaluation/scripts/fad.py

import csv
import json
import logging
import os
import sys

from frechet_audio_distance import FrechetAudioDistance

from .resynth_path_parsing import CSV_FIELDS, parse_resynth_path


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


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

    frechet = FrechetAudioDistance(
        model_name=modname,
        sample_rate=16000,  # VGGish resamples files to 16kHz
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

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
