import IPython
import torch
import json
import torchaudio
from itertools import combinations
from effortless_config import Config
import numpy as np
import yaml

from preprocess import preprocess
from interpolation import (get_interpolated_output, get_interpolated_outputs_sweep, load_model_from_weights,
                           get_model_with_interpolated_weights)


class args(Config):
    CONFIG = "config.yaml"

with open(args.CONFIG, "r") as config_file:
    config = yaml.safe_load(config_file)

sr = config["preprocess"]["sampling_rate"]

instrument_paths = {
    "vn": {
        "fix": "runs/fix_LR/20260629_191900/vn/state.pth",
        "exp": "runs/exp_LR/20260701_100835/vn/state.pth",
        "finetuned": "runs/finetune_exp_LR/20260707_002157/vn/state.pth",
        "exported_model": "export/runs_fix_LR_20260629_191900_vn/ddsp_vn_pretrained.ts",
    },
    "fl": {
        "fix": "runs/fix_LR/20260629_191900/fl/state.pth",
        "exp": "runs/exp_LR/20260703_200324/fl/state.pth",
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


path_stats_file = "preprocessed/mean_std_loudness.yml"
stats = yaml.safe_load(open(path_stats_file, "r")) 
mean_loudness = stats["mean_loudness"]
std_loudness = stats["std_loudness"]


for instrument1, instrument2 in combinations(["vn", "fl", "tpt"], 2):
    print(instrument1, instrument2)

    path1 = instrument_paths[instrument1]
    path2 = instrument_paths[instrument2]

    split_files_path = f"preprocessed/{instrument1}/split_files.json"
    with open(split_files_path, "r") as f:
        validation_files = json.load(f)["val"]

    for model_type in ["fix", "exp", "finetuned"]:
        for validation_file in validation_files:
            filename = validation_file.split("/")[-1]
            x, p, l = preprocess(validation_file, **config["preprocess"])
            pitch_tensor = torch.from_numpy(p).float().view(1, -1, 1)
            loudness_tensor = torch.from_numpy(l).float().view(1, -1, 1)
            loudness_norm = (loudness_tensor - mean_loudness) / std_loudness

            model1 = load_model_from_weights(path1[model_type], config)
            model2 = load_model_from_weights(path2[model_type], config)

            # extremes
            instrument1_output = model1.forward(pitch_tensor, loudness_norm).reshape(-1).detach().cpu().numpy()
            torchaudio.save(f"audios/{filename}_{instrument1}_{model_type}.wav",  instrument1_output, sample_rate=sr)

            instrument2_output = model2.forward(pitch_tensor, loudness_norm).reshape(-1).detach().cpu().numpy()
            torchaudio.save(f"audios/{filename}_{instrument2}_{model_type}.wav",  instrument2_output, sample_rate=sr)

            # interpolated output middle
            alpha = 0.5
            interpolated_output_middle_with_reverb = get_interpolated_output(        
                model1,
                model2, 
                pitch_tensor, 
                loudness_norm, 
                alpha=alpha
            )
            torchaudio.save(f"audios/{filename}_interpolated_output_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}_with_reverb.wav", 
                            interpolated_output_middle_with_reverb, sample_rate=sr)

            interpolated_output_middle_without_reverb = get_interpolated_output(        
                model1,
                model2, 
                pitch_tensor, 
                loudness_norm, 
                alpha=alpha,
                reverb=False
            )
            torchaudio.save(f"audios/{filename}_interpolated_output_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}_without_reverb.wav",  
                            interpolated_output_middle_without_reverb, sample_rate=sr)
            
            # interpolated output sweep
            n_steps_no_morph=2000
            interpolated_output_sweep_with_reverb = get_interpolated_outputs_sweep(model1, model2, pitch_tensor, loudness_norm, n_steps_no_morph, reverb=True)
            torchaudio.save(f"audios/{filename}_interpolated_output_sweep_{instrument1}_{instrument2}_{model_type}_with_reverb.wav",  
                            interpolated_output_sweep_with_reverb, sample_rate=sr)
            interpolated_output_sweep_without_reverb = get_interpolated_outputs_sweep(model1, model2, pitch_tensor, loudness_norm, n_steps_no_morph, reverb=False)
            torchaudio.save(f"audios/{filename}_interpolated_output_sweep_{instrument1}_{instrument2}_{model_type}_without_reverb.wav",  
                            interpolated_output_sweep_without_reverb, sample_rate=sr)

            # interpolated weights middle
            interpolated_weights_model = get_model_with_interpolated_weights(path1[model_type], path2[model_type], alpha, config)
            interpolated_weighs_audio = interpolated_weights_model(pitch_tensor, loudness_norm)
            torchaudio.save(f"audios/{filename}_interpolated_weights_{instrument1}_{instrument2}_{model_type}_alpha_{alpha}.wav", 
                            interpolated_weighs_audio.reshape(-1).detach().cpu().numpy(), sample_rate=sr)
