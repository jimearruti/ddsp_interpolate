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


@torch.no_grad()
def get_hidden(model, pitch, loudness):
    hidden = torch.cat([
            model.in_mlps[0](pitch),
            model.in_mlps[1](loudness),
        ], -1)
    hidden = torch.cat([model.gru(hidden)[0], pitch, loudness], -1)
    hidden = model.out_mlp(hidden)
    return hidden


@torch.no_grad()
def get_amplitudes(model, pitch, loudness):
    hidden = get_hidden(model, pitch, loudness)

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


@torch.no_grad()
def get_filter_param(model, pitch, loudness):
    hidden = get_hidden(model, pitch, loudness)
    param = scale_function(model.proj_matrices[1](hidden) - 5)
    return param


@torch.no_grad()
def get_interpolated_reverb_ir(ir_a, ir_b, alpha):
    return (1 - alpha) * ir_a + alpha * ir_b


@torch.no_grad()
def get_interpolated_output(model1, model2, pitch, loudness, 
                            alpha, ir1=None, ir2=None, reverb=True):
    amplitudes1 = get_amplitudes(model1, pitch, loudness)
    amplitudes2 = get_amplitudes(model2, pitch, loudness)
    amplitudes = (1 - alpha) * amplitudes1 + alpha * amplitudes2

    param1 = get_filter_param(model1, pitch, loudness)
    impulse1 = amp_to_impulse_response(param1, model1.block_size)
    param2 = get_filter_param(model2, pitch, loudness)
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

    if reverb == True:
        len_signal = signal.shape[1]
        if ir1 is None:
            ir1 = model1.reverb.build_impulse()
        if ir2 is None:
            ir2 = model2.reverb.build_impulse()
        ir = get_interpolated_reverb_ir(ir1, ir2, alpha)
        ir = nn.functional.pad(ir, (0, 0, 0, len_signal - model1.reverb.length))
        signal = fft_convolve(signal.squeeze(-1), ir.squeeze(-1)).unsqueeze(-1)
    return signal


@torch.no_grad()
def get_model_with_interpolated_weights(path_to_weights_1, path_to_weights_2, alpha, config, device="cpu"):
    state_model_1 = torch.load(path_to_weights_1, map_location=device, weights_only=True)
    state_model_2 = torch.load(path_to_weights_2, map_location=device, weights_only=True)

    interp_state = interpolate_state_dict(state_model_1, state_model_2, alpha)

    model_interp = DDSP(**config["model"])
    model_interp.load_state_dict(interp_state, strict=False)

    return model_interp


@torch.no_grad()
def interpolate_state_dict(state_model_1, state_model_2, alpha):
    interp_state = {}
    for key in state_model_1.keys():
        clean_key = key.removeprefix("ddsp.")
        interp_state[clean_key] = (1 - alpha) * state_model_1[key] + alpha * state_model_2[key]
    return interp_state


@torch.no_grad()
def load_model_from_weights(path_to_weights, config, device="cpu"):
    state_model = torch.load(path_to_weights, map_location=device, weights_only=True)
    model = DDSP(**config["model"])
    model.load_state_dict(state_model, strict=False)
    return model


@torch.no_grad()
def get_output_sweep(model1, model2, pitch, loudness, window_length, hop_length, reverb=True):
    # Just a draft, add COLA-OLA method with hann window
    alpha_values = np.linspace(0, 1, len(pitch))
    output = np.zeros((len(alpha_values), pitch.shape[1], 1))
    for alpha, i in zip(alpha_values, range(len(pitch))):
        output[i] = get_interpolated_output(model1, model2, pitch[i], loudness[i], 
                        alpha, reverb)
    return output