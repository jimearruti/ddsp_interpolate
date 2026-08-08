import json
import os

import torch
import torchaudio
import yaml

from train.preprocess import preprocess


def load_config(config_path):
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def load_loudness_stats(instrument_paths, processed_folder, instruments):
    '''
    Load the mean and std loudness values for each instrument from the models' respective config files.
    Also load the global mean and std loudness values from the processed folder.
    Args:
        instrument_paths (dict): Dictionary containing paths to each instrument's model.
        processed_folder (str): Path to the processed folder containing global stats.
        instruments (list): List of instrument names.
    Returns:
        mean (dict): Dictionary containing mean loudness values for each instrument and global.
        std (dict): Dictionary containing std loudness values for each instrument and global.
    '''
    mean = {}
    std = {}
    for instrument in instruments:
        instrument_config_path = os.path.join(
            os.path.dirname(instrument_paths[instrument]["from_scratch"]), "config.yaml"
        )
        with open(instrument_config_path, "r") as config_file_training:
            instrument_config = yaml.safe_load(config_file_training)
        mean[instrument] = instrument_config["data"]["mean_loudness"]
        std[instrument] = instrument_config["data"]["std_loudness"]

    global_stats_path = os.path.join(processed_folder, "mean_std_loudness.yml")
    with open(global_stats_path, "r") as f:
        global_stats = yaml.safe_load(f)
        mean["global"] = global_stats["mean_loudness"]
        std["global"] = global_stats["std_loudness"]

    return mean, std


def load_split_data(processed_folder):
    '''
    Load the split data from the processed folder.
    Args:
        processed_folder (str): Path to the processed folder containing split data.
    Returns:
        dict: Dictionary containing the split data.
    '''
    split_files_path = os.path.join(processed_folder, "split_files.json")
    with open(split_files_path, "r") as f:
        return json.load(f)


def get_or_process_tensors(test_file, cache_folder, preprocess_config):
    '''
    Get or process the pitch and loudness tensors for a given test file.
    Args:
        test_file (str): Path to the test audio file.
        cache_folder (str): Path to the cache folder where tensors are stored.
        preprocess_config (dict): Configuration for preprocessing.
    Returns:
        filename (str): Base name of the test file.
        pitch_tensor (torch.Tensor): Pitch tensor.
        loudness_tensor (torch.Tensor): Loudness tensor.
    '''
    filename = os.path.splitext(os.path.basename(test_file))[0]
    loudness_tensor_path = os.path.join(cache_folder, f"{filename}_loudness.pt")
    pitch_tensor_path = os.path.join(cache_folder, f"{filename}_pitch.pt")

    # Check if the tensors already exist in the cache folder
    if os.path.exists(loudness_tensor_path) and os.path.exists(pitch_tensor_path):
        print(f"loading {filename} tensors")
        loudness_tensor = torch.load(loudness_tensor_path)
        pitch_tensor = torch.load(pitch_tensor_path)
    else:
        # If not, process the audio file to generate the tensors
        print(f"processing {filename}")
        _, p, l = preprocess(test_file, **preprocess_config)
        # Save the tensors to the cache folder, with the correct shape for the model input
        pitch_tensor = torch.from_numpy(p).float().view(1, -1, 1)
        loudness_tensor = (l if isinstance(l, torch.Tensor) else torch.from_numpy(l)).float().view(1, -1, 1)
        torch.save(loudness_tensor, loudness_tensor_path)
        torch.save(pitch_tensor, pitch_tensor_path)

    return filename, pitch_tensor, loudness_tensor


def save_audio_if_missing(path_out, signal, sr):
    '''
    Save the audio signal to the specified path if it does not already exist.
    Args:
        path_out (str): Path where the audio signal should be saved.
        signal (torch.Tensor): Audio signal to be saved.
        sr (int): Sample rate of the audio signal.
    '''
    if os.path.exists(path_out):
        return
    torchaudio.save(path_out, signal.reshape(1, -1).detach().cpu(), sample_rate=sr)


def save_models_used(path1, path2, instrument1, instrument2, model_type, filename, results_folder):
    '''
    Save the paths of the models used for a given pair of instruments and model type to a JSON file.
    Args:
        path1 (str): Path to the first instrument's model.
        path2 (str): Path to the second instrument's model.
        instrument1 (str): Name of the first instrument.
        instrument2 (str): Name of the second instrument.
        model_type (str): Type of the model used.
        filename (str): Base name of the test file.
        results_folder (str): Path to the results folder where the JSON file should be saved.
    '''
    path_out = os.path.join(results_folder, f"{filename}_{instrument1}_{instrument2}_{model_type}_models.json")
    with open(path_out, "w") as f:
        json.dump({instrument1: path1, instrument2: path2}, f, indent=2)
