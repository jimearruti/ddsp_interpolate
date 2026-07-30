import os
import json

import pyloudnorm as pyln
import soundfile as sf
import yaml
from effortless_config import Config


class args(Config):
    GENERATE_CONFIG = "generate_config.yaml"

args.parse_args()

with open(args.GENERATE_CONFIG, "r") as config_file:
    generate_config = yaml.safe_load(config_file)

target_lufs = generate_config["postprocess"]["target_lufs"]
results_folder = generate_config["postprocess"]["normalize_results"]["results_dir"]
normalised_results_folder = f"{results_folder}_normalised"
os.makedirs(normalised_results_folder, exist_ok=True)

result_files = []
for root, _, files in os.walk(results_folder):
    for f in files:
        if f.endswith(".wav"):
            result_files.append(os.path.join(root, f))


original_loudness = {}
for file_path in result_files:

    data, rate = sf.read(file_path)
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, target_lufs)

    # Drop the root folder from the path
    relative_path = os.path.relpath(file_path, results_folder)
    output_path = os.path.join(normalised_results_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the normalised audio
    sf.write(output_path, loudness_normalized_audio, rate)
    original_loudness[relative_path] = loudness

original_loudness_path = os.path.join(normalised_results_folder, "original_loudness.json")
with open(original_loudness_path, "w") as f:
    json.dump(original_loudness, f, indent=4)
