import os
import yaml

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from effortless_config import Config


class args(Config):
    CONFIG = "config.yaml"
    GENERATE_CONFIG = "inference/generate_config.yaml"

args.parse_args()

with open(args.CONFIG, "r") as config_file:
    config = yaml.safe_load(config_file)

with open(args.GENERATE_CONFIG, "r") as config_file:
    generate_config = yaml.safe_load(config_file)

sampling_rate = config["preprocess"]["sampling_rate"]

crop_config = generate_config["postprocess"]["crop"]
results_folder = crop_config["results_dir"]
output_folder = crop_config["output_dir"]

# Fade duration in seconds
fade_duration = crop_config["fade_duration"]

# Target loudness for the final normalized output
target_lufs = generate_config["postprocess"]["target_lufs"]
 
 
def apply_fade(y, sr, fade_time):
    """Apply a linear fade-in and fade-out to a mono audio array."""
    fade_samples = int(fade_time * sr)
    fade_samples = min(fade_samples, len(y) // 2)  # don't overlap fades on short clips
 
    if fade_samples <= 0:
        return y
 
    y = y.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
 
    y[:fade_samples] *= fade_in
    y[-fade_samples:] *= fade_out
 
    return y


def to_stereo_normalized(y, sr, target_loudness):
    """Duplicate a mono signal to stereo and normalize to a target LUFS."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)

    # duplicate mono -> stereo (shape becomes [n_samples, 2])
    y_stereo = np.stack([y, y], axis=-1)
 
    # pyloudnorm returns -inf for near-silent/empty signals — skip normalization then
    if np.isfinite(loudness):
        y_stereo = pyln.normalize.loudness(y_stereo, loudness, target_loudness)
 
    return y_stereo


pieces = crop_config["pieces"]

for piece, info in pieces.items():
    start_time = info["start_time"]
    end_time = info["end_time"]
    original_path = info.get("original_path")

    piece_folder = os.path.join(results_folder, piece)
    os.makedirs(output_folder, exist_ok=True)
    files_in_folder = [f for f in os.listdir(piece_folder) if f.endswith(".wav")]

    for file_path in files_in_folder:
        full_path = os.path.join(piece_folder, file_path)
        y, _ = librosa.load(full_path, sr=sampling_rate)

        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)

        y_cropped = y[start_sample:end_sample]
        y_cropped = apply_fade(y_cropped, sampling_rate, fade_duration)
        y_cropped = to_stereo_normalized(y_cropped, sampling_rate, target_lufs)
        filename = os.path.splitext(os.path.basename(full_path))[0]
        out_path = os.path.join(output_folder, f"{filename}_cropped_{int(start_time * 1000)}_{int(end_time * 1000)}.wav")
        sf.write(out_path, y_cropped, sampling_rate)

    if original_path:
        y, _ = librosa.load(original_path, sr=sampling_rate)
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)
        y_cropped = y[start_sample:end_sample]
        y_cropped = apply_fade(y_cropped, sampling_rate, fade_duration)
        y_cropped = to_stereo_normalized(y_cropped, sampling_rate, target_lufs)
        filename = os.path.splitext(os.path.basename(original_path))[0]
        out_path = os.path.join(output_folder, f"{filename}_cropped_{int(start_time * 1000)}_{int(end_time * 1000)}.wav")
        sf.write(out_path, y_cropped, sampling_rate)

