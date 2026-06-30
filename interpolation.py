import IPython
import torch
from effortless_config import Config
import numpy as np
import yaml

from ddsp.core import (mean_std_loudness, scale_function, remove_above_nyquist, 
                        upsample, harmonic_synth, fft_convolve, amp_to_impulse_response)
from ddsp.model import DDSP
from preprocess import preprocess

import torch.nn as nn


def get_hidden(model, pitch, loudness, mean_l, std_l):
    normalised_loudness = (loudness - mean_l)/std_l
    hidden = torch.cat([
            model.in_mlps[0](pitch),
            model.in_mlps[1](normalised_loudness),
        ], -1)
    hidden = torch.cat([model.gru(hidden)[0], pitch, normalised_loudness], -1)
    hidden = model.out_mlp(hidden)
    return hidden


def get_amplitudes(model, pitch, loudness, mean_l, std_l):
    hidden = get_hidden(model, pitch, loudness, mean_l, std_l)

    param = scale_function(model.proj_matrices[0](hidden))

    total_amp = param[..., :1]
    amplitudes = param[..., 1:]

    amplitudes = remove_above_nyquist(
        amplitudes,
        pitch,
        model.sampling_rate,
    )
    amplitudes /= amplitudes.sum(-1, keepdim=True)
    amplitudes *= total_amp
    return amplitudes

def get_filter_param(model, pitch, loudness, mean_l, std_l):
    hidden = get_hidden(model, pitch, loudness, mean_l, std_l)
    param = scale_function(model.proj_matrices[1](hidden) - 5)
    return param

def get_interpolated_reverb_ir(reverb_a, reverb_b, alpha):
    ir_a = reverb_a.build_impulse()
    ir_b = reverb_b.build_impulse()
    return (1 - alpha) * ir_a + alpha * ir_b

def get_interpolated_output(model1, model2, mean_loudness, std_loudness, pitch, loudness, alpha, reverb=True):
    amplitudes1 = get_amplitudes(model1, pitch, loudness, mean_loudness, std_loudness)
    amplitudes2 = get_amplitudes(model2, pitch, loudness, mean_loudness, std_loudness)
    amplitudes = (1 - alpha) * amplitudes1 + alpha * amplitudes2

    param1 = get_filter_param(model1, pitch, loudness, mean_loudness, std_loudness)
    impulse1 = amp_to_impulse_response(param1, model1.block_size)
    param2 = get_filter_param(model2, pitch, loudness, mean_loudness, std_loudness)
    impulse2 = amp_to_impulse_response(param2, model2.block_size)
    impulse = (1 - alpha) * impulse1 + alpha * impulse2
    
    amplitudes = upsample(amplitudes, model1.block_size)
    pitch = upsample(pitch, model1.block_size)
    harmonic = harmonic_synth(pitch, amplitudes, model1.sampling_rate)

    noise = torch.rand(
        impulse.shape[0],
        impulse.shape[1],
        model1.block_size,
        dtype=impulse.dtype,
        device=impulse.device,
    ).to(impulse) * 2 - 1

    noise = fft_convolve(noise, impulse).contiguous()
    noise = noise.reshape(noise.shape[0], -1, 1)

    signal = harmonic + noise

    # reverb
    if reverb == True:
        reverb1 = model1.reverb
        reverb2 = model2.reverb
        
        len_signal = signal.shape[1]
        ir = get_interpolated_reverb_ir(reverb1, reverb2, alpha)
        ir = nn.functional.pad(ir, (0, 0, 0, len_signal - reverb1.length))
        signal = fft_convolve(signal.squeeze(-1), ir.squeeze(-1)).unsqueeze(-1)
        
    return signal


def get_model_with_interpolated_weights(path_to_weights_1, path_to_weights_2, alpha, config):
    state_model_1 = torch.load(path_to_weights_1)
    state_model_2 = torch.load(path_to_weights_2)

    interp_state = {}
    for key in state_model_1.keys():
        print(key)
        clean_key = key.removeprefix("ddsp.")
        interp_state[clean_key] = (1 - alpha) * state_model_1[key] + alpha * state_model_2[key]

    model_interp = DDSP(**config["model"])
    model_interp.load_state_dict(interp_state, strict=False)

    return model_interp


def load_model_from_weights(path_to_weights, config):
    state_model = torch.load(path_to_weights)
    model = DDSP(**config["model"])
    model.load_state_dict(state_model, strict=False)
    return model