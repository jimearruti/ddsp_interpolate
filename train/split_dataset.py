import datetime
import json
import pathlib
import yaml

import numpy as np
from effortless_config import Config

from .dataset_stats import calculate_durations_per_instrument_per_split


def get_files(data_location, extension, **_):
    '''
    Get all files with the given extension in the data_location directory and its subdirectories.
    '''
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def get_piece(f):
    '''
    Get the piece name from the file path. 
    Assumes the piece name is the second part of the parent directory name, split by underscores.
    '''
    return f.parent.name.split("_")[1]


def count_split_files(split_data):
    '''
    Count the number of files in each split (train, val, test) for each instrument.
    Args:
        split_data: Dictionary containing the split data for each instrument.
    Returns:
        counts: Dictionary containing the counts of files in each split for each instrument.
    '''
    counts = {}
    for instrument, subsets in split_data.items():
        counts[instrument] = {
            subset: len(files) for subset, files in subsets.items()
        }
    return counts


def print_split_counts(counts):
    '''
    Print the counts of files in each split (train, val, test) for each instrument.
    Args:
        counts: Dictionary containing the counts of files in each split for each instrument.
    ''' 
    header = f"{'instrument':<12}{'train':>8}{'val':>8}{'test':>8}{'total':>8}"
    print(header)
    print("-" * len(header))
    for instrument, subset_counts in counts.items():
        total = sum(subset_counts.values())
        print(
            f"{instrument:<12}"
            f"{subset_counts.get('train', 0):>8}"
            f"{subset_counts.get('val', 0):>8}"
            f"{subset_counts.get('test', 0):>8}"
            f"{total:>8}"
        )


def build_piece_split_map(files, val_ratio=0.1, test_ratio=0.1, seed=42, n_trials=200, check_instruments=()):
    '''
    Divide the dataset into train, validation, and test splits based on pieces, 
    ensuring that the specified instruments are balanced across the splits.
    Args:
        files: List of file paths to be split.
        val_ratio: Ratio of validation data.
        test_ratio: Ratio of test data.
        seed: Random seed for reproducibility.
        n_trials: Number of random splits to try for balancing.
        check_instruments: List of instruments to check for balance across splits.
    '''
    piece_instrument_counts = {}

    # Count the number of files for each instrument in each piece
    for f in files:
        piece = get_piece(f)
        instrument = f.stem.split("_")[2]
        piece_instrument_counts.setdefault(piece, {})
        piece_instrument_counts[piece][instrument] = piece_instrument_counts[piece].get(instrument, 0) + 1

    # Sort pieces to ensure consistent ordering for reproducibility
    pieces = sorted(piece_instrument_counts.keys())
    # Calculate total counts for each instrument across all pieces
    total_counts = {
        inst: sum(piece_instrument_counts[p].get(inst, 0) for p in pieces)
        for inst in check_instruments
    }

    # Determine the number of pieces to allocate to test and validation sets
    n = len(pieces)
    # Ensure at least one piece is allocated to test and validation if the ratios are non-zero
    n_test = 0 if test_ratio == 0 else max(1, int(test_ratio * n))
    n_val = 0 if val_ratio == 0 else max(1, int(val_ratio * n))

    # Use a random number generator with the specified seed for reproducibility
    rng = np.random.default_rng(seed)
    # Initialize variables to keep track of the best split and its score
    best_split, best_score = None, None

    # Try multiple random splits to find the one that best balances the specified instruments across the splits
    for _ in range(n_trials):
        # Randomly permute the pieces and allocate them to test, validation, and training sets
        perm = list(rng.permutation(pieces))
        # Allocate pieces to test, validation, and training sets based on the calculated numbers
        test_pieces = perm[:n_test]
        val_pieces = perm[n_test:n_test + n_val]
        train_pieces = perm[n_test + n_val:]

        # Calculate a score for the current split based on how well it balances the 
        # specified instruments across the splits
        score = 0.0
        # For each instrument to check, calculate the proportion of files in the test and validation sets
        # and compare it to the desired ratios, accumulating the squared differences into the score
        for inst in check_instruments:
            total = total_counts[inst]
            if total == 0:
                continue
            test_count = sum(piece_instrument_counts[p].get(inst, 0) for p in test_pieces)
            val_count = sum(piece_instrument_counts[p].get(inst, 0) for p in val_pieces)
            score += (test_count / total - test_ratio) ** 2
            score += (val_count / total - val_ratio) ** 2

        # If the current split has a lower score than the best found so far, update the best split and score
        if best_score is None or score < best_score:
            best_score = score
            best_split = (train_pieces, val_pieces, test_pieces)

    # Create a mapping from pieces to their assigned split (train, val, test) based on the best split found
    train_pieces, val_pieces, test_pieces = best_split
    piece_split = {}
    for p in test_pieces:
        piece_split[p] = "test"
    for p in val_pieces:
        piece_split[p] = "val"
    for p in train_pieces:
        piece_split[p] = "train"
    return piece_split


def split_files(files, piece_split):
    '''
    Split the list of files into train, validation, and test sets based on the piece_split mapping.
    Args:
        files: List of file paths to be split.
        piece_split: Dictionary mapping pieces to their assigned split (train, val, test).
    Returns:
        train: List of file paths assigned to the training set.
        val: List of file paths assigned to the validation set.
        test: List of file paths assigned to the test set.
    '''
    train = [f for f in files if piece_split.get(get_piece(f)) == "train"]
    val = [f for f in files if piece_split.get(get_piece(f)) == "val"]
    test = [f for f in files if piece_split.get(get_piece(f)) == "test"]

    if not train:
        raise ValueError("No files left for train split — reduce val/test ratios or check the piece_split mapping.")

    return train, val, test


def check_split_ratios(split_data, val_ratio, test_ratio, check_instruments, tolerance=0.1):
    '''
    Check if the actual validation and test ratios for the specified instruments are within 
    the given tolerance of the desired ratios.
    Args:
        split_data: Dictionary containing the split data for each instrument.
        val_ratio: Desired ratio of validation data.
        test_ratio: Desired ratio of test data.
        check_instruments: List of instruments to check for balance across splits.
        tolerance: Allowed deviation from the desired ratios.
    '''
    for instrument in check_instruments:
        subsets = split_data.get(instrument, {})
        n_total = sum(len(subsets.get(s, [])) for s in ("train", "val", "test"))
        if n_total == 0:
            continue
        actual_val = len(subsets.get("val", [])) / n_total
        actual_test = len(subsets.get("test", [])) / n_total
        if abs(actual_val - val_ratio) > tolerance:
            print(f"Warning: '{instrument}' val ratio {actual_val:.2f} vs target {val_ratio:.2f}")
        if abs(actual_test - test_ratio) > tolerance:
            print(f"Warning: '{instrument}' test ratio {actual_test:.2f} vs target {test_ratio:.2f}")


def parse_hhmmss(s):
    '''Parse an 'H:MM:SS' string (as produced by str(datetime.timedelta))
    back into total seconds.
    Args:
        s: A string in the format 'H:MM:SS'.
    Returns:
        Total seconds as an integer.
    '''
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def print_durations(duration_summary):
    '''
    Print the duration summary in a table format.
    Parameters:
        duration_summary (dict): A dictionary containing the duration summary for each instrument and split.
    '''
    header = f"{'instrument':<12}{'train':>12}{'val':>12}{'test':>12}{'total':>12}"
    print(header)
    print("-" * len(header))
    for instrument, splits in duration_summary.items():
        train = splits.get("train", "0:00:00")
        val = splits.get("val", "0:00:00")
        test = splits.get("test", "0:00:00")

        total_seconds = sum(parse_hhmmss(s) for s in (train, val, test))
        total = str(datetime.timedelta(seconds=total_seconds))

        print(
            f"{instrument:<12}"
            f"{train:>12}"
            f"{val:>12}"
            f"{test:>12}"
            f"{total:>12}"
        )


def main():
    # Parse command-line arguments and load configuration
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()

    with open(args.CONFIG, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    # Get all audio files based on the configuration
    files = get_files(**config["data"])

    # Get the list of instruments to include in dataset
    instruments = config["data"]["instruments"]

    root_out_path = pathlib.Path(config["preprocess"]["out_dir"])
    split_data = {}

    # Define the instruments to check for balance across splits and the desired validation and test ratios
    check_instruments = ("vn", "fl", "tpt")
    # Get the validation and test ratios from the configuration, defaulting to 0.1 if not specified
    val_ratio = config["preprocess"].get("val_ratio", 0.1)
    test_ratio = config["preprocess"].get("test_ratio", 0.1)

    # Build a mapping from pieces to their assigned split (train, val, test) 
    # based on the specified ratios and instruments to check
    piece_split = build_piece_split_map(
        files, val_ratio=val_ratio, test_ratio=test_ratio, seed=42,
        n_trials=200, check_instruments=check_instruments
    )

    # Split the files for each instrument based on the piece_split mapping and store them in split_data
    for instrument in instruments:
        # Filter the files for the current instrument based on the naming convention in the file stem
        files_instrument = [f for f in files if f.stem.split("_")[2] == instrument]
        # Split the files for the current instrument into train, validation, and test sets
        train_files, val_files, test_files = split_files(files_instrument, piece_split)

        # Store the split files for the current instrument in the split_data dictionary
        split_data[instrument] = {
            "train": [str(file) for file in train_files],
            "val": [str(file) for file in val_files],
            "test": [str(file) for file in test_files],
        }

    # Check if the actual validation and test ratios for the specified instruments are within the given tolerance
    check_split_ratios(split_data, val_ratio, test_ratio, check_instruments)

    # Create the output directory if it doesn't exist and save the split data to a JSON file
    root_out_path.mkdir(parents=True, exist_ok=True)
    with open(root_out_path / "split_files.json", "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    # Count the number of files in each split for each instrument, print the counts, and save them to a JSON file
    counts = count_split_files(split_data)
    print_split_counts(counts)
    with open(root_out_path / "split_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    # Calculate the total duration of audio files for each instrument and split, print the durations, 
    # and save them to a JSON file
    duration_summary = calculate_durations_per_instrument_per_split(split_data)
    print_durations(duration_summary)
    with open(root_out_path / "split_durations.json", "w", encoding="utf-8") as f:
        json.dump(duration_summary, f, indent=2)


if __name__ == "__main__":
    main()