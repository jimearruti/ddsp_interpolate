import json
import os
from itertools import permutations

import torch
import torchaudio
import yaml
from effortless_config import Config

from interpolation import (
    get_interpolated_output,
    get_interpolated_outputs_sweep,
    get_model_with_interpolated_weights,
    load_model_from_weights,
)

from preprocess import preprocess

class args(Config):
    CONFIG = "config.yaml"

with open(args.CONFIG, "r") as config_file:
    config = yaml.safe_load(config_file)

sr = config["preprocess"]["sampling_rate"]
processed_folder = config["preprocess"]["out_dir"]
n_fft=config["preprocess"]["n_fft"]
results_folder = "results"
files_processed_folder = f"{processed_folder}/per_track"


instrument_paths = {
    "vn": {
        "from_scratch": "runs/from_scratch_after_split_change/20260717_114838/vn/state_30000.pth",
        "finetuned": "runs/finetune_after_split_change/20260718_133157/vn/state_30000.pth",
    },
    "fl": {
        "from_scratch": "runs/from_scratch_after_split_change/20260717_114838/fl/state_30000.pth",
        "finetuned": "runs/finetune_after_split_change/20260718_133157/fl/state_30000.pth",
    },
    "tpt": {
        "from_scratch": "runs/from_scratch_after_split_change/20260717_114838/tpt/state_30000.pth",
        "finetuned": "runs/finetune_after_split_change/20260718_133157/tpt/state_30000.pth",
    },
}


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if not os.path.exists(files_processed_folder):
    os.makedirs(files_processed_folder)


mean = {}
std = {}
for instrument in ["vn", "fl", "tpt"]:
    instrument_config_path = os.path.join(os.path.dirname(instrument_paths[instrument]["from_scratch"]), "config.yaml")
    with open(instrument_config_path, "r") as config_file_training:
        instrument_config = yaml.safe_load(config_file_training)
    mean[instrument] = instrument_config["data"]["mean_loudness"]
    std[instrument] = instrument_config["data"]["std_loudness"]

global_stats_path = os.path.join(processed_folder, "mean_std_loudness.yml")
with open(global_stats_path, "r") as f:
    global_stats = yaml.safe_load(f)
    mean["global"] = global_stats["mean_loudness"]
    std["global"] = global_stats["std_loudness"]

split_files_path = os.path.join(processed_folder, "split_files.json")
with open(split_files_path, "r") as f:
    validation_files = json.load(f)["val"]

for instrument1, instrument2 in permutations(["vn", "fl", "tpt"], 2):
    print(f"working on {instrument1}->{instrument2}")

    instrument_recording_preprocesed_folder = os.path.join(files_processed_folder, instrument1)
    if not os.path.exists(instrument_recording_preprocesed_folder):
        os.makedirs(instrument_recording_preprocesed_folder)

    path1 = instrument_paths[instrument1]
    path2 = instrument_paths[instrument2]

    for validation_file in validation_files:
        filename = validation_file.split("/")[-1]
        loudness_tensor_path = os.path.join(instrument_recording_preprocesed_folder, f"{filename}_loudness.pt")
        pitch_tensor_path = os.path.join(instrument_recording_preprocesed_folder, f"{filename}_pitch.pt")
       
        if os.path.exists(loudness_tensor_path) and os.path.exists(pitch_tensor_path):
            print(f"loading {filename} tensors")
            loudness_tensor = torch.load(loudness_tensor_path)
            pitch_tensor = torch.load(pitch_tensor_path)
        else:
            print(f"processing {filename}")
            x, p, l = preprocess(validation_file, **config["preprocess"])
            pitch_tensor = torch.from_numpy(p).float().view(1, -1, 1)
            loudness_tensor = (l if isinstance(l, torch.Tensor) else torch.from_numpy(l)).float().view(1, -1, 1)
            torch.save(loudness_tensor, loudness_tensor_path)
            torch.save(pitch_tensor, pitch_tensor_path)

        # normalize loudness for each instrument
        loudness_norm_instrument1 = (loudness_tensor - mean[instrument1]) / std[instrument1]
        loudness_norm_instrument2 = (loudness_tensor - mean[instrument2]) / std[instrument2]
        loudness_norm_global = (loudness_tensor - mean["global"]) / std["global"]

        for model_type in ["from_scratch", "finetuned"]:
            print(f"the model is {model_type}")
            with torch.no_grad():
                model1 = load_model_from_weights(path1[model_type], config)
                model2 = load_model_from_weights(path2[model_type], config)
                # extremes
                instrument_1_audio = f"{results_folder}/{filename}_{instrument1}_{model_type}.wav"
                if not os.path.exists(instrument_1_audio):
                    instrument1_output = model1.forward(pitch_tensor, loudness_norm_instrument1).reshape(1, -1).detach().cpu()
                    torchaudio.save(f"{results_folder}/{filename}_{instrument1}_{model_type}.wav",  instrument1_output, sample_rate=sr)

                instrument_2_audio = f"{results_folder}/{filename}_{instrument2}_{model_type}.wav"
                if not os.path.exists(instrument_2_audio):
                    instrument2_output = model2.forward(pitch_tensor, loudness_norm_instrument2).reshape(1, -1).detach().cpu()
                    torchaudio.save(f"{results_folder}/{filename}_{instrument2}_{model_type}.wav",  instrument2_output, sample_rate=sr)

                # interpolated output middle
                alpha = 0.5
                interpolated_output_with_reverb_audio = f"{results_folder}/{filename}_interpolated_output_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}_with_reverb.wav"
                if not os.path.exists(interpolated_output_with_reverb_audio):
                    interpolated_output_middle_with_reverb = get_interpolated_output(
                        model1,
                        model2,
                        pitch_tensor,
                        loudness_norm_global,
                        alpha=alpha
                    )
                    torchaudio.save(interpolated_output_with_reverb_audio,
                                    interpolated_output_middle_with_reverb.reshape(1, -1).detach().cpu(), sample_rate=sr)

                interpolated_output_without_reverb_audio = f"{results_folder}/{filename}_interpolated_output_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}_without_reverb.wav"
                if not os.path.exists(interpolated_output_without_reverb_audio):
                    interpolated_output_middle_without_reverb = get_interpolated_output(
                        model1,
                        model2,
                        pitch_tensor,
                        loudness_norm_global,
                        alpha=alpha,
                        reverb=False
                    )
                    torchaudio.save(interpolated_output_without_reverb_audio,
                                    interpolated_output_middle_without_reverb.reshape(1, -1).detach().cpu(), sample_rate=sr)

                # interpolated weights middle
                interpolated_weights_audio_path = f"{results_folder}/{filename}_interpolated_weights_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}.wav"
                if not os.path.exists(interpolated_weights_audio_path):
                    interpolated_weights_model = get_model_with_interpolated_weights(path1[model_type], path2[model_type], alpha, config)
                    interpolated_weighs_audio = interpolated_weights_model(pitch_tensor, loudness_norm_global)
                    torchaudio.save(interpolated_weights_audio_path,
                                    interpolated_weighs_audio.reshape(1, -1).detach().cpu(), sample_rate=sr)
                    del interpolated_weights_model

                # cleanup
                del model1, model2
                torch.cuda.empty_cache()