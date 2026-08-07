import os
import json

import librosa as li
import pyloudnorm as pyln
import soundfile as sf
import yaml
from effortless_config import Config

from ddsp.utils import high_pass_filter


class args(Config):
    CONFIG = "config.yaml"
    GENERATE_CONFIG = "generate_config.yaml"

args.parse_args()

with open(args.CONFIG, "r") as config_file:
    config = yaml.safe_load(config_file)

with open(args.GENERATE_CONFIG, "r") as config_file:
    generate_config = yaml.safe_load(config_file)

sampling_rate = config["preprocess"]["sampling_rate"]
target_lufs = generate_config["postprocess"]["target_lufs"]

results_folder = config["data"]["data_location"]
normalised_results_folder = os.path.join(results_folder, "normalised")
os.makedirs(normalised_results_folder, exist_ok=True)

result_files = []
for root, _, files in os.walk(results_folder):
    for f in files:
        if f.endswith(".wav"):
            result_files.append(os.path.join(root, f))


original_loudness = {}
for file_path in result_files:
    x, _ = li.load(file_path, sr=sampling_rate)

    meter = pyln.Meter(sampling_rate)
    x = pyln.normalize.peak(x, -1.0)
    x = high_pass_filter(x, 80, fs=sampling_rate)
    loudness = meter.integrated_loudness(x)
    loudness_normalized_audio = pyln.normalize.loudness(x, loudness, target_lufs)

    # Drop the root folder from the path
    relative_path = os.path.relpath(file_path, results_folder)
    output_path = os.path.join(normalised_results_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the normalised audio
    sf.write(output_path, loudness_normalized_audio, sampling_rate)

    original_loudness[relative_path] = loudness

original_loudness_path = os.path.join(normalised_results_folder, "original_loudness.json")
with open(original_loudness_path, "w") as f:
    json.dump(original_loudness, f, indent=4)
