import pathlib
import json

import numpy as np
import yaml
from effortless_config import Config


def get_files(data_location: str, extension: str, **_) -> list[pathlib.Path]:
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def split_files(files, val_ratio=0.1, test_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    files = list(rng.permutation(files))
    n = len(files)
    n_test = max(1, int(test_ratio * n))
    n_val  = max(1, int(val_ratio * n))
    test  = files[:n_test]
    val   = files[n_test:n_test + n_val]
    train = files[n_test + n_val:]
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


if __name__ == "__main__":
    main()