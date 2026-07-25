import os
import json

import pyloudnorm as pyln


results_folder = "results"
normalised_results_folder = os.path.join(results_folder, "normalised")
os.makedirs(normalised_results_folder, exist_ok=True)

result_files = [f for f in os.listdir(results_folder) if f.endswith(".wav")]
original_loudness = {}

for file in result_files:
    file_path = os.path.join(results_folder, file)
    data, rate = pyln.load(file_path)

    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -12.0)

    # Save the normalised audio
    output_path = os.path.join(normalised_results_folder, file)
    pyln.save(output_path, loudness_normalized_audio, rate)

    original_loudness[file] = loudness

original_loudness_path = os.path.join(normalised_results_folder, "original_loudness.json")
with open(original_loudness_path, "w") as f:
    json.dump(original_loudness, f, indent=4)
