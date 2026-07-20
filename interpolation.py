import torch
import torch.nn as nn
import numpy as np

from ddsp.core import (scale_function, remove_above_nyquist, 
                        upsample, harmonic_synth, fft_convolve, amp_to_impulse_response)
from ddsp.model import DDSP


@torch.no_grad()
def get_hidden(model, pitch, loudness):
    '''
    Get the hidden representation from the model given pitch and loudness inputs.
    '''
    hidden = torch.cat([
            model.in_mlps[0](pitch),
            model.in_mlps[1](loudness),
        ], -1)
    hidden = torch.cat([model.gru(hidden)[0], pitch, loudness], -1)
    hidden = model.out_mlp(hidden)
    return hidden


@torch.no_grad()
def get_amplitudes(model, pitch, loudness):
    '''
    Get the amplitudes from the model given pitch and (normalised) loudness inputs.
    '''
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
    '''
    Get the filter parameters from the model given pitch and (normalised) loudness inputs.
    '''
    hidden = get_hidden(model, pitch, loudness)
    param = scale_function(model.proj_matrices[1](hidden) - 5)
    return param


@torch.no_grad()
def get_interpolated_reverb_ir(ir_a, ir_b, alpha):
    return (1 - alpha) * ir_a + alpha * ir_b


@torch.no_grad()
def get_interpolated_output(model1, model2, pitch, loudness, 
                            alpha, ir1=None, ir2=None, reverb=True):
    '''
    Get the interpolated output from two models given pitch and (normalised) loudness inputs, 
    and an interpolation factor alpha.
    Arguments:
        model1: first DDSP model
        model2: second DDSP model
        pitch: pitch input tensor of shape (batch, time, 1)
        loudness: loudness input tensor of shape (batch, time, 1)
        alpha: interpolation factor
        ir1: reverb impulse response for model1
        ir2: reverb impulse response for model2
        reverb: whether to apply reverb
    Returns:
        signal: interpolated output signal tensor of shape (batch, time, 1)
    '''

    # get amplitudes and filter impulse response from both models
    amplitudes1 = get_amplitudes(model1, pitch, loudness)
    param1 = get_filter_param(model1, pitch, loudness)
    impulse1 = amp_to_impulse_response(param1, model1.block_size)

    amplitudes2 = get_amplitudes(model2, pitch, loudness)
    param2 = get_filter_param(model2, pitch, loudness)
    impulse2 = amp_to_impulse_response(param2, model2.block_size)

    # interpolate amplitudes and impulse responses
    amplitudes = (1 - alpha) * amplitudes1 + alpha * amplitudes2
    impulse = (1 - alpha) * impulse1 + alpha * impulse2
    
    # upsample amplitudes and pitch to match sr (they are at control rate before)
    amplitudes = upsample(amplitudes, model1.block_size)
    pitch = upsample(pitch, model1.block_size)

    # synthesize harmonic output
    harmonic = harmonic_synth(pitch, amplitudes, model1.sampling_rate)

    # generate noise
    noise = torch.rand(
        impulse.shape[0],
        impulse.shape[1],
        model1.block_size,
        dtype=impulse.dtype,
        device=impulse.device,
    ).to(impulse) * 2 - 1

    # filter noise with interpolated impulse response
    noise = fft_convolve(noise, impulse).contiguous()
    noise = noise.reshape(noise.shape[0], -1, 1)

    # combine harmonic and noise to get the final signal
    signal = harmonic + noise

    if reverb == True:
        len_signal = signal.shape[1]
        if ir1 is None:
            ir1 = model1.reverb.build_impulse()
        if ir2 is None:
            ir2 = model2.reverb.build_impulse()
        # interpolate reverb impulse responses
        ir = get_interpolated_reverb_ir(ir1, ir2, alpha)
        ir = nn.functional.pad(ir, (0, 0, 0, len_signal - model1.reverb.length))
        # apply reverb to the signal
        signal = fft_convolve(signal.squeeze(-1), ir.squeeze(-1)).unsqueeze(-1)

    return signal


def update_weights(model, new_weights_dict):
    '''
    Update the model's weights with the weights from new_weights_dict.
    This function modifies the model in-place and returns it.
    '''
    model.load_state_dict(new_weights_dict, strict=False)
    return model


@torch.no_grad()
def interpolate_state_dict(state_model_1, state_model_2, alpha):
    '''
    Interpolate between two state dictionaries.
    Only interpolates the weights of the in_mlps, gru, out_mlp, proj_matrices, and reverb layers.
    Arguments:
        state_model_1: state dictionary of the first model
        state_model_2: state dictionary of the second model
        alpha: interpolation factor
    Returns:
        interp_state: interpolated state dictionary
    '''
    include_prefixes = ("in_mlps.", "gru.", "out_mlp.", "proj_matrices.")
    interp_keys = [k for k in state_model_1.keys() if k.startswith(include_prefixes)]

    reverb_learned = ("reverb.noise", "reverb.decay", "reverb.wet")
    interp_keys += [k for k in reverb_learned if k in state_model_1.keys()]
    
    interp_state = {}
    for key in interp_keys:
        interp_state[key] = (1 - alpha) * state_model_1[key] + alpha * state_model_2[key]
    return interp_state


@torch.no_grad()
def get_model_with_interpolated_weights(path_to_weights_1, path_to_weights_2, alpha, config, device="cpu"):
    '''
    Load two models from their weights and return a new model with interpolated weights.
    Arguments:
        path_to_weights_1: path to the weights of the first model
        path_to_weights_2: path to the weights of the second model
        alpha: interpolation factor
        config: configuration dictionary for the model
        device: device to load the model on
    '''
    # load the state dictionaries of the two models
    state_model_1 = torch.load(path_to_weights_1, map_location=device, weights_only=True)
    state_model_2 = torch.load(path_to_weights_2, map_location=device, weights_only=True)

    # interpolate the state dictionaries of relevant layers
    interp_state = interpolate_state_dict(state_model_1, state_model_2, alpha)

    # create a new model
    model_interp = DDSP(**config["model"])
    # load the model weights from the first model
    model_interp.load_state_dict(state_model_1, strict=False)
    # overwrite relevant layers with the interpolated weights
    model_interp.load_state_dict(interp_state, strict=False)

    return model_interp


@torch.no_grad()
def load_model_from_weights(path_to_weights, config, device="cpu"):
    '''
    Return a DDSP model loaded from the given weights file.
    Arguments:
        path_to_weights: path to the weights file
        config: configuration dictionary for the model
        device: device to load the model on
    '''
    # load the state dictionary of the model
    state_model = torch.load(path_to_weights, map_location=device, weights_only=True)
    model = DDSP(**config["model"])
    # load the model weights from the state dictionary
    missing, unexpected = model.load_state_dict(state_model, strict=False)
    if missing or unexpected:
        print(f"WARNING loading {path_to_weights}:")
        print(f"  missing keys: {missing}")
        print(f"  unexpected keys: {unexpected}")

    # set the model to evaluation mode
    model.eval()
    return model


@torch.no_grad()
def get_interpolated_outputs_sweep(model1, model2, pitch, loudness, n_steps_no_morph, ir1=None, ir2=None, reverb=True):
    block_size = model1.block_size
    n_steps = pitch.shape[1]  
    
    alpha_values = torch.cat([
        torch.zeros(n_steps_no_morph, device=pitch.device, dtype=pitch.dtype),
        torch.linspace(0, 1, n_steps - 2 * n_steps_no_morph, device=pitch.device, dtype=pitch.dtype),
        torch.ones(n_steps_no_morph, device=pitch.device, dtype=pitch.dtype)
    ])
    
    alpha_tensor = alpha_values.view(1, n_steps, 1)

    amplitudes1 = get_amplitudes(model1, pitch, loudness)
    amplitudes2 = get_amplitudes(model2, pitch, loudness)
    
    amplitudes = (1 - alpha_tensor) * amplitudes1 + alpha_tensor * amplitudes2

    param1 = get_filter_param(model1, pitch, loudness)
    impulse1 = amp_to_impulse_response(param1, model1.block_size)
    param2 = get_filter_param(model2, pitch, loudness)
    impulse2 = amp_to_impulse_response(param2, model2.block_size)
    
    impulse = (1 - alpha_tensor) * impulse1 + alpha_tensor * impulse2

    
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
        alpha = 0.5
        ir = get_interpolated_reverb_ir(ir1, ir2, alpha)
        ir = nn.functional.pad(ir, (0, 0, 0, len_signal - model1.reverb.length))
        signal = fft_convolve(signal.squeeze(-1), ir.squeeze(-1)).unsqueeze(-1)
    return signal
    

@torch.no_grad()
def get_interpolated_weights_sweep(path_to_weights_model_1, path_to_weights_model_2,
                                   pitch, loudness, config, window_length):
    interp_model = load_model_from_weights(path_to_weights_model_1, config)

    state_model_1 = torch.load(path_to_weights_model_1, map_location="cpu", weights_only=True)
    state_model_2 = torch.load(path_to_weights_model_2, map_location="cpu", weights_only=True)

    block_size = interp_model.block_size
    n_steps = pitch.shape[1]
    n_audio_samples = n_steps * block_size

    frame_count = int(np.ceil(n_steps / window_length))
    alpha_values = np.linspace(0, 1, frame_count)

    output = torch.zeros(pitch.shape[0], n_audio_samples, 1, dtype=pitch.dtype, device=pitch.device)

    for i in range(frame_count):
        start = i * window_length
        end = min(start + window_length, n_steps)

        alpha = alpha_values[i]
        interpolated_weights_dict = interpolate_state_dict(
            state_model_1, state_model_2, alpha
        )
        frame_output = interp_model.forward_sweep(pitch[:, start:end, :], loudness[:, start:end, :],
                                                  new_weights_dict=interpolated_weights_dict
        )  # (batch, (end-start)*block_size, 1)

        a_start = start * block_size
        a_end = a_start + frame_output.shape[1]
        output[:, a_start:a_end] = frame_output

    return output
