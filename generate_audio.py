import json
import os
from itertools import permutations

import torch
import torchaudio
import yaml
from effortless_config import Config

from interpolation import (
    apply_interpolated_reverb,
    get_interpolated_output,
    get_model_with_interpolated_weights,
    load_model_from_weights,
)

from preprocess import preprocess

INSTRUMENTS = ["vn", "fl", "tpt"]
MODEL_TYPES = ["from_scratch", "finetuned"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


class args(Config):
    CONFIG = "config.yaml"


def load_config(config_path):
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def load_instrument_paths():
    return {
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


def load_loudness_stats(instrument_paths, processed_folder, instruments):
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
    split_files_path = os.path.join(processed_folder, "split_files.json")
    with open(split_files_path, "r") as f:
        return json.load(f)


def get_or_process_tensors(test_file, cache_folder, preprocess_config):
    filename = test_file.split("/")[-1]
    loudness_tensor_path = os.path.join(cache_folder, f"{filename}_loudness.pt")
    pitch_tensor_path = os.path.join(cache_folder, f"{filename}_pitch.pt")

    if os.path.exists(loudness_tensor_path) and os.path.exists(pitch_tensor_path):
        print(f"loading {filename} tensors")
        loudness_tensor = torch.load(loudness_tensor_path)
        pitch_tensor = torch.load(pitch_tensor_path)
    else:
        print(f"processing {filename}")
        _, p, l = preprocess(test_file, **preprocess_config)
        pitch_tensor = torch.from_numpy(p).float().view(1, -1, 1)
        loudness_tensor = (l if isinstance(l, torch.Tensor) else torch.from_numpy(l)).float().view(1, -1, 1)
        torch.save(loudness_tensor, loudness_tensor_path)
        torch.save(pitch_tensor, pitch_tensor_path)

    return filename, pitch_tensor, loudness_tensor


def normalize_loudness(loudness_tensor, mean, std, keys):
    return {key: (loudness_tensor - mean[key]) / std[key] for key in keys}


def save_audio_if_missing(path_out, signal, sr):
    if os.path.exists(path_out):
        return
    torchaudio.save(path_out, signal.reshape(1, -1).detach().cpu(), sample_rate=sr)


def generate_extremes(model1, model2, instrument1, instrument2, model_type,
                       pitch_tensor, loudness_norm, filename, results_folder, sr):
    instrument_1_audio = f"{results_folder}/{filename}_{instrument1}_{model_type}.wav"
    if not os.path.exists(instrument_1_audio):
        instrument1_output = model1.forward(pitch_tensor, loudness_norm[instrument1])
        save_audio_if_missing(instrument_1_audio, instrument1_output, sr)

    instrument_2_audio = f"{results_folder}/{filename}_{instrument2}_{model_type}.wav"
    if not os.path.exists(instrument_2_audio):
        instrument2_output = model2.forward(pitch_tensor, loudness_norm[instrument2])
        save_audio_if_missing(instrument_2_audio, instrument2_output, sr)


def generate_interpolated_outputs(model1, model2, instrument1, instrument2, model_type,
                                   pitch_tensor, loudness_norm_global, filename,
                                   results_folder, sr, alphas):
    for alpha in alphas:
        base_name = f"{filename}_interpolated_output_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}"

        with_reverb_path = f"{results_folder}/{base_name}_with_reverb.wav"
        without_reverb_path = f"{results_folder}/{base_name}_without_reverb.wav"

        need_with_reverb = not os.path.exists(with_reverb_path)
        need_without_reverb = not os.path.exists(without_reverb_path)
        if not need_with_reverb and not need_without_reverb:
            continue

        output_without_reverb = get_interpolated_output(
            model1, model2, pitch_tensor, loudness_norm_global, alpha=alpha, reverb=False
        )
        if need_without_reverb:
            save_audio_if_missing(without_reverb_path, output_without_reverb, sr)

        if need_with_reverb:
            output_with_reverb = apply_interpolated_reverb(
                output_without_reverb, model1, model2, alpha
            )
            save_audio_if_missing(with_reverb_path, output_with_reverb, sr)


def generate_interpolated_weights_outputs(path1, path2, instrument1, instrument2, model_type,
                                           pitch_tensor, loudness_norm_global, filename,
                                           results_folder, sr, config, alphas):
    for alpha in alphas:
        interpolated_weights_audio_path = (
            f"{results_folder}/{filename}_interpolated_weights_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}.wav"
        )
        if os.path.exists(interpolated_weights_audio_path):
            continue

        interpolated_weights_model = get_model_with_interpolated_weights(path1, path2, alpha, config)
        interpolated_weights_audio = interpolated_weights_model(pitch_tensor, loudness_norm_global)
        save_audio_if_missing(interpolated_weights_audio_path, interpolated_weights_audio, sr)
        del interpolated_weights_model


def process_pair(instrument1, instrument2, instrument_paths, split_data, mean, std,
                  preprocess_config, files_processed_folder, results_folder, sr, config,
                  model_types, alphas):
    print(f"working on {instrument1}->{instrument2}")

    instrument_cache_folder = os.path.join(files_processed_folder, instrument1)
    os.makedirs(instrument_cache_folder, exist_ok=True)

    path1 = instrument_paths[instrument1]
    path2 = instrument_paths[instrument2]

    for test_file in split_data[instrument1]["test"]:
        filename, pitch_tensor, loudness_tensor = get_or_process_tensors(
            test_file, instrument_cache_folder, preprocess_config
        )

        loudness_norm = normalize_loudness(
            loudness_tensor, mean, std, [instrument1, instrument2, "global"]
        )
        loudness_norm_global = loudness_norm["global"]

        for model_type in model_types:
            print(f"the model is {model_type}")
            with torch.no_grad():
                model1 = load_model_from_weights(path1[model_type], config)
                model2 = load_model_from_weights(path2[model_type], config)

                generate_extremes(
                    model1, model2, instrument1, instrument2, model_type,
                    pitch_tensor, loudness_norm, filename, results_folder, sr
                )

                generate_interpolated_outputs(
                    model1, model2, instrument1, instrument2, model_type,
                    pitch_tensor, loudness_norm_global, filename, results_folder, sr, alphas
                )

                generate_interpolated_weights_outputs(
                    path1[model_type], path2[model_type], instrument1, instrument2, model_type,
                    pitch_tensor, loudness_norm_global, filename, results_folder, sr, config, alphas
                )

                del model1, model2
                torch.cuda.empty_cache()


def main():
    args.parse_args()
    config = load_config(args.CONFIG)

    sr = config["preprocess"]["sampling_rate"]
    processed_folder = config["preprocess"]["out_dir"]
    results_folder = "results"
    files_processed_folder = f"{processed_folder}/per_track"

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(files_processed_folder, exist_ok=True)

    instrument_paths = load_instrument_paths()
    mean, std = load_loudness_stats(instrument_paths, processed_folder, INSTRUMENTS)
    split_data = load_split_data(processed_folder)

    for instrument1, instrument2 in permutations(INSTRUMENTS, 2):
        process_pair(
            instrument1, instrument2, instrument_paths, split_data, mean, std,
            config["preprocess"], files_processed_folder, results_folder, sr, config,
            MODEL_TYPES, ALPHAS
        )


if __name__ == "__main__":
    main()
