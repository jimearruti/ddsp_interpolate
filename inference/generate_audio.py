import os
from itertools import permutations

import torch
from effortless_config import Config

from .interpolation import (
    apply_interpolated_reverb,
    build_model_from_state_dict,
    get_interpolated_output,
    get_interpolated_outputs_sweep,
    get_interpolated_weights_sweep,
    get_model_with_interpolated_weights,
)
from .io_utils import (
    get_or_process_tensors,
    load_config,
    load_loudness_stats,
    load_split_data,
    save_audio_if_missing,
    save_models_used,
)
from .loudness import standardize_loudness, standardize_loudness_interpolated


class args(Config):
    CONFIG = "config.yaml"
    GENERATE_CONFIG = "inference/generate_config.yaml"


def generate_extremes(model1, model2, instrument1, instrument2, model_type,
                       pitch_tensor, loudness_norm, filename, results_folder, sr):
    '''
    Generate and save the model outputs for two instruments using their respective models.
    Args:
        model1 (torch.nn.Module): The first instrument's model.
        model2 (torch.nn.Module): The second instrument's model.
        instrument1 (str): Name of the first instrument.
        instrument2 (str): Name of the second instrument.
        model_type (str): Type of the model used.
        pitch_tensor (torch.Tensor): Tensor containing pitch information.
        loudness_norm (dict): Normalized loudness values for each instrument.
        filename (str): Base name of the test file.
        results_folder (str): Path to the results folder where the audio files should be saved.
        sr (int): Sample rate for the audio files.
    '''
    instrument_1_audio = f"{results_folder}/{filename}_{instrument1}_{model_type}.wav"
    if not os.path.exists(instrument_1_audio):
        instrument1_output = model1.forward(pitch_tensor, loudness_norm[instrument1])
        save_audio_if_missing(instrument_1_audio, instrument1_output, sr)

    instrument_2_audio = f"{results_folder}/{filename}_{instrument2}_{model_type}.wav"
    if not os.path.exists(instrument_2_audio):
        instrument2_output = model2.forward(pitch_tensor, loudness_norm[instrument2])
        save_audio_if_missing(instrument_2_audio, instrument2_output, sr)


def generate_interpolated_outputs(model1, model2, instrument1, instrument2, model_type,
                                   pitch_tensor, loudness_tensor, mean, std, filename,
                                   results_folder, sr, alphas):
    '''
    Generate and save the output obtained when interpolating the synth parameters of two models.
    Generate outputs with and without reverb for each interpolation parameter in the alphas list.
    Args:
        model1 (torch.nn.Module): The first instrument's model.
        model2 (torch.nn.Module): The second instrument's model.
        instrument1 (str): Name of the first instrument.
        instrument2 (str): Name of the second instrument.
        model_type (str): Type of the model used.
        pitch_tensor (torch.Tensor): Tensor containing pitch information.
        loudness_tensor (torch.Tensor): Tensor containing loudness information.
        mean (float): Mean value for normalization.
        std (float): Standard deviation for normalization.
        filename (str): Base name of the test file.
        results_folder (str): Path to the results folder where the audio files should be saved.
        sr (int): Sample rate for the audio files.
        alphas (list): List of interpolation parameters.
    '''
    for alpha in alphas:
        alpha_pct = int(alpha * 100)
        base_name = f"{filename}_interpolated_output_{instrument1}_{instrument2}_{model_type}_alpha_{alpha_pct}"

        with_reverb_path = f"{results_folder}/{base_name}_with_reverb.wav"
        without_reverb_path = f"{results_folder}/{base_name}_without_reverb.wav"

        need_with_reverb = not os.path.exists(with_reverb_path)
        need_without_reverb = not os.path.exists(without_reverb_path)
        if not need_with_reverb and not need_without_reverb:
            continue

        loudness_norm_interp = standardize_loudness_interpolated(
            loudness_tensor, mean, std, instrument1, instrument2, alpha
        )
        output_without_reverb = get_interpolated_output(
            model1, model2, pitch_tensor, loudness_norm_interp, alpha=alpha, reverb=False
        )
        if need_without_reverb:
            save_audio_if_missing(without_reverb_path, output_without_reverb, sr)

        if need_with_reverb:
            output_with_reverb = apply_interpolated_reverb(
                output_without_reverb, model1, model2, alpha
            )
            save_audio_if_missing(with_reverb_path, output_with_reverb, sr)


def generate_interpolated_weights_outputs(state_model_1, state_model_2, instrument1, instrument2, model_type,
                                           pitch_tensor, loudness_tensor, mean, std, filename,
                                           results_folder, sr, config, alphas, device):
    '''
    Generate and save the output of a model whose weights are interpolated weights of two models.
    Save result with and without reverb for each interpolation parameter in the alphas list.
    Args:
        state_model_1 (dict): State dictionary of the first instrument's model.
        state_model_2 (dict): State dictionary of the second instrument's model.
        instrument1 (str): Name of the first instrument.
        instrument2 (str): Name of the second instrument.
        model_type (str): Type of the model used.
        pitch_tensor (torch.Tensor): Tensor containing pitch information.
        loudness_tensor (torch.Tensor): Tensor containing loudness information.
        mean (float): Mean value for normalization.
        std (float): Standard deviation for normalization.
        filename (str): Base name of the test file.
        results_folder (str): Path to the results folder where the audio files should be saved.
        sr (int): Sample rate for the audio files.
        config (dict): Configuration for the models.
        alphas (list): List of interpolation parameters.
        device (torch.device): Device to run the models on.
    '''
    for alpha in alphas:
        alpha_pct = int(alpha * 100)
        base_name = f"{filename}_interpolated_weights_{instrument1}_{instrument2}_{model_type}_alpha_{alpha_pct}"

        with_reverb_path = f"{results_folder}/{base_name}_with_reverb.wav"
        without_reverb_path = f"{results_folder}/{base_name}_without_reverb.wav"

        need_with_reverb = not os.path.exists(with_reverb_path)
        need_without_reverb = not os.path.exists(without_reverb_path)
        if not need_with_reverb and not need_without_reverb:
            continue

        loudness_norm_interp = standardize_loudness_interpolated(
            loudness_tensor, mean, std, instrument1, instrument2, alpha
        )
        interpolated_weights_model = get_model_with_interpolated_weights(
            state_model_1, state_model_2, alpha, config, device=device
        )

        if need_without_reverb:
            output_without_reverb = interpolated_weights_model(
                pitch_tensor, loudness_norm_interp, apply_reverb=False
            )
            save_audio_if_missing(without_reverb_path, output_without_reverb, sr)

        if need_with_reverb:
            output_with_reverb = interpolated_weights_model(
                pitch_tensor, loudness_norm_interp, apply_reverb=True
            )
            save_audio_if_missing(with_reverb_path, output_with_reverb, sr)

        del interpolated_weights_model


def generate_output_sweep(model1, model2, instrument1, instrument2, model_type,
                           pitch_tensor, loudness_tensor, mean, std, filename,
                           results_folder, sr):
    '''
    Generate and save the output for interpolating synth paraeters for two models, sweeping the interpolation parameter.
    If reverb is desired, the reverb IR of both models is averaged and applied, only the dry signal is affected by the
    interpolation parameter.
    The sweep is generated in three parts:
        first third of audio: outputs for the first model
        middle third of audio: interpolated outputs between the two models, sweeping alpha from 0 to 1
        last third of audio: outputs for the second model
    Args:
        model1 (dict): State dictionary of the first instrument's model.
        model2 (dict): State dictionary of the second instrument's model.
        instrument1 (str): Name of the first instrument.
        instrument2 (str): Name of the second instrument.
        model_type (str): Type of the model used.
        pitch_tensor (torch.Tensor): Tensor containing pitch information.
        loudness_tensor (torch.Tensor): Tensor containing loudness information.
        mean (float): Mean value for normalization.
        std (float): Standard deviation for normalization.
        filename (str): Base name of the test file.
        results_folder (str): Path to the results folder where the audio files should be saved.
        sr (int): Sample rate for the audio files.
    '''
    base_name = f"{filename}_sweep_output_{instrument1}_{instrument2}_{model_type}"
    with_reverb_path = f"{results_folder}/{base_name}_with_reverb.wav"
    without_reverb_path = f"{results_folder}/{base_name}_without_reverb.wav"

    need_with_reverb = not os.path.exists(with_reverb_path)
    need_without_reverb = not os.path.exists(without_reverb_path)
    if not need_with_reverb and not need_without_reverb:
        return

    n_steps_no_morph = pitch_tensor.shape[1] // 3

    if need_without_reverb:
        output_without_reverb = get_interpolated_outputs_sweep(
            model1, model2, pitch_tensor, loudness_tensor, mean, std, instrument1, instrument2,
            n_steps_no_morph, reverb=False
        )
        save_audio_if_missing(without_reverb_path, output_without_reverb, sr)

    if need_with_reverb:
        output_with_reverb = get_interpolated_outputs_sweep(
            model1, model2, pitch_tensor, loudness_tensor, mean, std, instrument1, instrument2,
            n_steps_no_morph, reverb=True
        )
        save_audio_if_missing(with_reverb_path, output_with_reverb, sr)


def generate_weights_sweep(state_model_1, state_model_2, instrument1, instrument2, model_type,
                            pitch_tensor, loudness_tensor, mean, std, filename,
                            results_folder, sr, config, device):
    '''
    Generate and save the weights sweep for two instruments using their respective models.
    If reverb is desired, the reverb IR of both models is averaged and applied, only
    the dry signal is interpolated.
    The sweep is generated in three parts:
        first third of audio: weights for the first model
        middle third of audio: interpolated weights between the two models, sweeping alpha from 0 to 1
        last third of audio: weights for the second model
    Args:
        state_model_1 (dict): State dictionary of the first instrument's model.
        state_model_2 (dict): State dictionary of the second instrument's model.
        instrument1 (str): Name of the first instrument.
        instrument2 (str): Name of the second instrument.
        model_type (str): Type of the model used.
        pitch_tensor (torch.Tensor): Tensor containing pitch information.
        loudness_tensor (torch.Tensor): Tensor containing loudness information.
        mean (float): Mean value for normalization.
        std (float): Standard deviation for normalization.
        filename (str): Base name of the test file.
        results_folder (str): Path to the results folder where the audio files should be saved.
        sr (int): Sample rate for the audio files.
        config (dict): Configuration parameters for the interpolation.
        device (torch.device): Device on which to perform computations.
    '''
    base_name = f"{filename}_sweep_weights_{instrument1}_{instrument2}_{model_type}"
    with_reverb_path = f"{results_folder}/{base_name}_with_reverb.wav"
    without_reverb_path = f"{results_folder}/{base_name}_without_reverb.wav"

    need_with_reverb = not os.path.exists(with_reverb_path)
    need_without_reverb = not os.path.exists(without_reverb_path)
    if not need_with_reverb and not need_without_reverb:
        return

    n_steps_no_morph = pitch_tensor.shape[1] // 3

    if need_without_reverb:
        output_without_reverb = get_interpolated_weights_sweep(
            state_model_1, state_model_2, pitch_tensor, loudness_tensor, mean, std, instrument1, instrument2,
            config, n_steps_no_morph, reverb=False, device=device
        )
        save_audio_if_missing(without_reverb_path, output_without_reverb, sr)

    if need_with_reverb:
        output_with_reverb = get_interpolated_weights_sweep(
            state_model_1, state_model_2, pitch_tensor, loudness_tensor, mean, std, instrument1, instrument2,
            config, n_steps_no_morph, reverb=True, device=device
        )
        save_audio_if_missing(with_reverb_path, output_with_reverb, sr)


def process_pair(instrument1, instrument2, instrument_paths, split_data, mean, std,
                  preprocess_config, files_processed_folder, results_folder, sr, config,
                  model_types, alphas, device):
    '''
    Process a pair of instruments by generating audio outputs for each test file in the split data.
    Args:
        instrument1 (str): Name of the first instrument.
        instrument2 (str): Name of the second instrument.
        instrument_paths (dict): Dictionary mapping instrument names to their respective model paths.
        split_data (dict): Dictionary containing the split data for each instrument.
        mean (float): Mean value for normalization.
        std (float): Standard deviation for normalization.
        preprocess_config (dict): Configuration parameters for preprocessing.
        files_processed_folder (str): Path to the folder where processed files are stored.
        results_folder (str): Path to the folder where results should be saved.
        sr (int): Sample rate for the audio files.
        config (dict): Configuration parameters for the interpolation.
        model_types (list): List of model types to use.
        alphas (list): List of alpha values for interpolation.
        device (torch.device): Device on which to perform computations.
    '''
    print(f"working on {instrument1}->{instrument2}")

    instrument_cache_folder = os.path.join(files_processed_folder, instrument1)
    os.makedirs(instrument_cache_folder, exist_ok=True)

    path1 = instrument_paths[instrument1]
    path2 = instrument_paths[instrument2]

    for test_file in split_data[instrument1]["test"]:
        filename, pitch_tensor, loudness_tensor = get_or_process_tensors(
            test_file, instrument_cache_folder, preprocess_config
        )
        pitch_tensor = pitch_tensor.to(device)
        loudness_tensor = loudness_tensor.to(device)

        loudness_norm = standardize_loudness(
            loudness_tensor, mean, std, [instrument1, instrument2]
        )

        track_name = os.path.splitext(filename)[0]
        track_results_folder = os.path.join(results_folder, track_name)
        os.makedirs(track_results_folder, exist_ok=True)

        for model_type in model_types:
            print(f"the model is {model_type}")
            with torch.no_grad():
                state_model_1 = torch.load(path1[model_type], map_location=device, weights_only=True)
                state_model_2 = torch.load(path2[model_type], map_location=device, weights_only=True)
                model1 = build_model_from_state_dict(state_model_1, config, device=device)
                model2 = build_model_from_state_dict(state_model_2, config, device=device)

                save_models_used(
                    path1[model_type], path2[model_type], instrument1, instrument2,
                    model_type, filename, track_results_folder
                )

                generate_extremes(
                    model1, model2, instrument1, instrument2, model_type,
                    pitch_tensor, loudness_norm, filename, track_results_folder, sr
                )

                generate_interpolated_outputs(
                    model1, model2, instrument1, instrument2, model_type,
                    pitch_tensor, loudness_tensor, mean, std, filename, track_results_folder, sr, alphas
                )

                generate_interpolated_weights_outputs(
                    state_model_1, state_model_2, instrument1, instrument2, model_type,
                    pitch_tensor, loudness_tensor, mean, std, filename, track_results_folder, sr, config, alphas,
                    device
                )

                generate_output_sweep(
                    model1, model2, instrument1, instrument2, model_type,
                    pitch_tensor, loudness_tensor, mean, std, filename, track_results_folder, sr
                )

                generate_weights_sweep(
                    state_model_1, state_model_2, instrument1, instrument2, model_type,
                    pitch_tensor, loudness_tensor, mean, std, filename, track_results_folder, sr, config, device
                )

                del model1, model2
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def main():
    args.parse_args()
    config = load_config(args.CONFIG)
    generate_config = load_config(args.GENERATE_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    sr = config["preprocess"]["sampling_rate"]
    processed_folder = config["preprocess"]["out_dir"]
    results_folder = generate_config["generate"]["results_dir"]
    files_processed_folder = f"{processed_folder}/per_track"

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(files_processed_folder, exist_ok=True)

    instrument_paths = generate_config["instrument_paths"]
    instruments = generate_config["generate"]["instruments"]
    model_types = generate_config["generate"]["model_types"]
    alphas = generate_config["generate"]["alphas"]

    mean, std = load_loudness_stats(instrument_paths, processed_folder, instruments)
    split_data = load_split_data(processed_folder)

    for instrument1, instrument2 in permutations(instruments, 2):
        process_pair(
            instrument1, instrument2, instrument_paths, split_data, mean, std,
            config["preprocess"], files_processed_folder, results_folder, sr, config,
            model_types, alphas, device
        )


if __name__ == "__main__":
    main()
