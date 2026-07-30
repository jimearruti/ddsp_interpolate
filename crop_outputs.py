import os
import yaml

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from effortless_config import Config


class args(Config):
    CONFIG = "config.yaml"

args.parse_args()

with open(args.CONFIG, "r") as config_file:
    config = yaml.safe_load(config_file)

sampling_rate = config["preprocess"]["sampling_rate"]

results_folder = "results"
output_folder = "results_cropped"

# Fade duration in seconds
fade_duration = 0.02

# Target loudness for the final normalized output
target_lufs = -24.0
 
 
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


pieces = {
    "AuSep_1_vn_02_Sonata": {
        "original_path": "../data/URMP/normalised/02_Sonata_vn_vn/AuSep_1_vn_02_Sonata.wav",
        "start_time": 4.330, 
        "end_time": 12.569
    },
    "AuSep_2_vn_02_Sonata": {
        "original_path": "../data/URMP/normalised/02_Sonata_vn_vn/AuSep_2_vn_02_Sonata.wav",
        "start_time": 4.128, 
        "end_time": 12.767
    },
    "AuSep_1_vn_35_Rondeau": {
        "original_path": "../data/URMP/normalised/35_Rondeau_vn_vn_va_db/AuSep_1_vn_35_Rondeau.wav",
        "start_time": 11.801, 
        "end_time": 18.635
    },
    "AuSep_2_vn_35_Rondeau": {
        "original_path": "../data/URMP/normalised/35_Rondeau_vn_vn_va_db/AuSep_2_vn_35_Rondeau.wav",
        "start_time": 3.809, 
        "end_time": 14.334
    },
    "AuSep_2_vn_09_Jesus": {
        "original_path": "../data/URMP/normalised/09_Jesus_tpt_vn/AuSep_2_vn_09_Jesus.wav",
        "start_time": 5.000, 
        "end_time": 12.819
    },
    "AuSep_1_fl_40_Miserere": {
        "original_path": "../data/URMP/normalised/40_Miserere_fl_fl_ob_cl_bn/AuSep_1_fl_40_Miserere.wav",
        "start_time": 8.109,
        "end_time": 15.854,
    },
    "AuSep_2_fl_40_Miserere": {
        "original_path": "../data/URMP/normalised/40_Miserere_fl_fl_ob_cl_bn/AuSep_2_fl_40_Miserere.wav",
        "start_time": 2.065,
        "end_time": 9.446,
    },
    "AuSep_1_fl_37_Rondeau": {
        "original_path": "../data/URMP/normalised/37_Rondeau_fl_vn_va_cl/AuSep_1_fl_37_Rondeau.wav",
        "start_time": 50.854,
        "end_time": 55.911,
    },
    "AuSep_1_tpt_31_Slavonic": {
        "original_path": "../data/URMP/normalised/31_Slavonic_tpt_tpt_hn_tbn/AuSep_1_tpt_31_Slavonic.wav",
        "start_time": 2.065,
        "end_time": 9.446,
    },
    "AuSep_2_tpt_31_Slavonic": {
        "original_path": "../data/URMP/normalised/31_Slavonic_tpt_tpt_hn_tbn/AuSep_2_tpt_31_Slavonic.wav",
        "start_time": 7.845,
        "end_time": 13.394,
    },
    "AuSep_1_tpt_33_Elise": {
        "original_path": "../data/URMP/normalised/33_Elise_tpt_tpt_hn_tbn/AuSep_1_tpt_33_Elise.wav",
        "start_time": 13.667,
        "end_time": 19.335,
    },
    "AuSep_2_tpt_33_Elise": {
        "original_path": "../data/URMP/normalised/33_Elise_tpt_tpt_hn_tbn/AuSep_2_tpt_33_Elise.wav",
        "start_time": 4.115,
        "end_time": 10.752,
    },
    "AuSep_1_tpt_43_Chorale": {
        "original_path": "../data/URMP/normalised/43_Chorale_tpt_tpt_hn_tbn_tba/AuSep_1_tpt_43_Chorale.wav",
        "start_time": 24.798,
        "end_time": 30.317,
    },
}

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

