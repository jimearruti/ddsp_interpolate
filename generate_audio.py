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

instrument_paths = {
    "vn": {
        "fix": "runs/fix_LR/20260629_191900/vn/state.pth",
        "exp": "runs/exp_LR/20260701_100835/vn/state.pth",
        "finetuned": "runs/finetune_exp_LR/20260707_002157/vn/state.pth",
        "exported_model": "export/runs_fix_LR_20260629_191900_vn/ddsp_vn_pretrained.ts",
    },
    "fl": {
        "fix": "runs/fix_LR/20260629_191900/fl/state.pth",
        "exp": "runs/from_scratch_exp_LR_after_loudness_fix_512/20260711_133838/fl/state.pth",
        # "exp": "runs/exp_LR/20260703_200324/fl/state.pth",
        "finetuned": "runs/finetune_exp_LR/20260707_002157/fl/state.pth",
        "exported_model": "export/runs_fix_LR_20260629_191900_fl/ddsp_fl_pretrained.ts",
    },
    "tpt": {
        "fix": "runs/fix_LR/20260629_191900/tpt/state.pth",
        "exp": "runs/debug/20260706_123700/tpt/state.pth",
        "finetuned": "runs/finetune_exp_LR/20260707_002157/tpt/state.pth",
        "exported_model": "export/runs_fix_LR_20260629_191900_tpt/ddsp_tpt_pretrained.ts",
    },
}

results_folder = "results_1024"
files_processed_folder = f"{processed_folder}/per_track"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if not os.path.exists(files_processed_folder):
    os.makedirs(files_processed_folder)


for instrument1, instrument2 in [("vn", "fl"), ("fl", "vn")]: #permutations(["vn", "fl", "tpt"], 2):
    print(f"working on {instrument1}->{instrument2}")

    instrument_recording_preprocesed_folder = os.path.join(files_processed_folder, instrument1)
    if not os.path.exists(instrument_recording_preprocesed_folder):
        os.makedirs(instrument_recording_preprocesed_folder)

    path1 = instrument_paths[instrument1]
    path2 = instrument_paths[instrument2]

    split_files_path = f"{processed_folder}/{instrument1}/split_files.json"
    with open(split_files_path, "r") as f:
        validation_files = json.load(f)["val"]

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

        for model_type in [f"exp"]: #["fix", "exp", "finetuned"]:    
            print(f"the model is {model_type}")
            model1 = load_model_from_weights(path1[model_type], config)
            model2 = load_model_from_weights(path2[model_type], config)

            instrument1_config_path = os.path.join(os.path.dirname(path1[model_type]), "config.yaml")
            instrument2_config_path = os.path.join(os.path.dirname(path2[model_type]), "config.yaml")

            with open(instrument1_config_path, "r") as config_file_training:
                instrument1_config = yaml.safe_load(config_file_training)
        
            instrument1_mean_loudness = instrument1_config["data"]["mean_loudness"]
            instrument1_std_loudness = instrument1_config["data"]["std_loudness"]
        
            with open(instrument2_config_path, "r") as config_file_training:
                instrument2_config = yaml.safe_load(config_file_training)
        
            instrument2_mean_loudness = instrument2_config["data"]["mean_loudness"]
            instrument2_std_loudness = instrument2_config["data"]["std_loudness"]

            loudness_norm_instrument1 = (loudness_tensor - instrument1_mean_loudness) / instrument1_std_loudness
            loudness_norm_instrument2 = (loudness_tensor - instrument2_mean_loudness) / instrument2_std_loudness

            # extremes
            instrument_1_audio = f"{results_folder}/{filename}_{instrument1}_{model_type}.wav"
            if not os.path.exists(instrument_1_audio):
                instrument1_output = model1.forward(pitch_tensor, loudness_norm_instrument1).reshape(1, -1).detach().cpu()
                torchaudio.save(f"{results_folder}/{filename}_{instrument1}_{model_type}.wav",  instrument1_output, sample_rate=sr)

            instrument_2_audio = f"{results_folder}/{filename}_{instrument2}_{model_type}.wav"
            if not os.path.exists(instrument_2_audio):
                instrument2_output = model2.forward(pitch_tensor, loudness_norm_instrument2).reshape(1, -1).detach().cpu()
                torchaudio.save(f"{results_folder}/{filename}_{instrument2}_{model_type}.wav",  instrument2_output, sample_rate=sr)

            # # interpolated output middle
            # alpha = 0.5
            # interpolated_output_middle_with_reverb = get_interpolated_output(        
            #     model1,
            #     model2, 
            #     pitch_tensor, 
            #     loudness_norm, 
            #     alpha=alpha
            # )
            # torchaudio.save(f"{results_folder}/{filename}_interpolated_output_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}_with_reverb.wav", 
            #                 interpolated_output_middle_with_reverb.reshape(1, -1).detach().cpu(), sample_rate=sr)

            # interpolated_output_middle_without_reverb = get_interpolated_output(        
            #     model1,
            #     model2, 
            #     pitch_tensor, 
            #     loudness_norm, 
            #     alpha=alpha,
            #     reverb=False
            # )
            # torchaudio.save(f"{results_folder}/{filename}_interpolated_output_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}_without_reverb.wav",  
            #                 interpolated_output_middle_without_reverb.reshape(1, -1).detach().cpu(), sample_rate=sr)
            
            # # interpolated output sweep
            # n_steps_no_morph=2000
            # interpolated_output_sweep_with_reverb = get_interpolated_outputs_sweep(model1, model2, pitch_tensor, loudness_norm, n_steps_no_morph, reverb=True)
            # torchaudio.save(f"{results_folder}/{filename}_interpolated_output_sweep_{instrument1}_{instrument2}_{model_type}_with_reverb.wav", 
            #                 interpolated_output_sweep_with_reverb.reshape(1, -1).detach().cpu(), sample_rate=sr)
            # interpolated_output_sweep_without_reverb = get_interpolated_outputs_sweep(model1, model2, pitch_tensor, loudness_norm, n_steps_no_morph, reverb=False)
            # torchaudio.save(f"{results_folder}/{filename}_interpolated_output_sweep_{instrument1}_{instrument2}_{model_type}_without_reverb.wav",  
            #                 interpolated_output_sweep_without_reverb.reshape(1, -1).detach().cpu(), sample_rate=sr)

            # # interpolated weights middle
            # interpolated_weights_model = get_model_with_interpolated_weights(path1[model_type], path2[model_type], alpha, config)
            # interpolated_weighs_audio = interpolated_weights_model(pitch_tensor, loudness_norm)
            # torchaudio.save(f"{results_folder}/{filename}_interpolated_weights_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}.wav", 
            #                 interpolated_weighs_audio.reshape(1, -1).detach().cpu(), sample_rate=sr)

            #cleanup
            # del model1, model2, interpolated_weights_model
            # torch.cuda.empty_cache()