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
def apply_interpolated_reverb(signal, model1, model2, alpha, ir1=None, ir2=None):
    '''
    Apply interpolated reverb (between model1's and model2's reverb) to a dry signal.
    Arguments:
        signal: dry signal tensor of shape (batch, time, 1)
        model1: first DDSP model
        model2: second DDSP model
        alpha: interpolation factor
        ir1: reverb impulse response for model1
        ir2: reverb impulse response for model2
    Returns:
        signal: signal tensor of shape (batch, time, 1) with reverb applied
    '''
    len_signal = signal.shape[1]
    if ir1 is None:
        ir1 = model1.reverb.build_impulse()
    if ir2 is None:
        ir2 = model2.reverb.build_impulse()
    # interpolate reverb impulse responses
    ir = get_interpolated_reverb_ir(ir1, ir2, alpha)
    ir = nn.functional.pad(ir, (0, 0, 0, len_signal - model1.reverb.length))
    # apply reverb to the signal
    return fft_convolve(signal.squeeze(-1), ir.squeeze(-1)).unsqueeze(-1)


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

    amplitudes2 = get_amplitudes(model2, pitch, loudness)
    param2 = get_filter_param(model2, pitch, loudness)

    # interpolate amplitudes and impulse responses
    amplitudes = (1 - alpha) * amplitudes1 + alpha * amplitudes2
    param = (1 - alpha) * param1 + alpha * param2
    
    # upsample amplitudes and pitch to match sr (they are at control rate before)
    amplitudes = upsample(amplitudes, model1.block_size)
    pitch = upsample(pitch, model1.block_size)

    # synthesize harmonic output
    harmonic = harmonic_synth(pitch, amplitudes, model1.sampling_rate)

    # impulse response for noise filter
    impulse = amp_to_impulse_response(param, model2.block_size)
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
        signal = apply_interpolated_reverb(signal, model1, model2, alpha, ir1, ir2)

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
def get_model_with_interpolated_weights(state_model_1, state_model_2, alpha, config, device="cpu"):
    '''
    Build a new model with weights interpolated between two already-loaded state dictionaries.
    Arguments:
        state_model_1: state dictionary of the first model
        state_model_2: state dictionary of the second model
        alpha: interpolation factor
        config: configuration dictionary for the model
        device: device to build the model on
    '''
    # interpolate the state dictionaries of relevant layers
    interp_state = interpolate_state_dict(state_model_1, state_model_2, alpha)

    # create a new model
    model_interp = DDSP(**config["model"]).to(device)
    # load the model weights from the first model
    model_interp.load_state_dict(state_model_1, strict=False)
    # overwrite relevant layers with the interpolated weights
    model_interp.load_state_dict(interp_state, strict=False)

    return model_interp


@torch.no_grad()
def build_model_from_state_dict(state_model, config, device="cpu"):
    '''
    Return a DDSP model built from an already-loaded state dictionary.
    Arguments:
        state_model: state dictionary of the model
        config: configuration dictionary for the model
        device: device to build the model on
    '''
    model = DDSP(**config["model"]).to(device)
    missing, unexpected = model.load_state_dict(state_model, strict=False)
    if missing or unexpected:
        print("WARNING loading state dict:")
        print(f"  missing keys: {missing}")
        print(f"  unexpected keys: {unexpected}")

    # set the model to evaluation mode
    model.eval()
    return model


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
    return build_model_from_state_dict(state_model, config, device)


def build_sweep_alphas(n_steps, n_steps_no_morph, device=None, dtype=None):
    '''
    Build a per-frame alpha tensor for a sweep: flat 0, linear ramp 0->1, flat 1.
    Arguments:
        n_steps: total number of frames in the sweep
        n_steps_no_morph: number of flat (non-interpolated) frames at each end
        device: device to build the tensor on
        dtype: dtype of the tensor
    Returns:
        alpha_values: 1D tensor of length n_steps
    '''
    return torch.cat([
        torch.zeros(n_steps_no_morph, device=device, dtype=dtype),
        torch.linspace(0, 1, n_steps - 2 * n_steps_no_morph, device=device, dtype=dtype),
        torch.ones(n_steps_no_morph, device=device, dtype=dtype)
    ])


@torch.no_grad()
def get_interpolated_outputs_sweep(model1, model2, pitch, loudness, mean, std, instrument1, instrument2,
                                   n_steps_no_morph, ir1=None, ir2=None, reverb=True):
    '''
    Generate the output by interpolating synth parameters between two models.the dry signal is affected by the interpolation parameter.
    The sweep is generated in three parts:
        first third of audio: outputs for the first model
        middle third of audio: interpolated parameters between the two models, sweeping alpha from 0 to 1
        last third of audio: outputs for the second model
    Arguments:
        model1: first DDSP model
        model2: second DDSP model
        pitch: pitch input tensor
        loudness: loudness input tensor
        mean: mean values for loudness normalization
        std: standard deviation values for loudness normalization
        instrument1: identifier for the first instrument
        instrument2: identifier for the second instrument
        n_steps_no_morph: number of steps for non-interpolated sections
        ir1: impulse response for the first model (optional)
        ir2: impulse response for the second model (optional)
        reverb: whether to apply reverb (default: True)
    Returns:
        signal: interpolated output signal tensor of shape (batch, time, 1)
    '''
    # amount of frames
    n_steps = pitch.shape[1]

    # create a tensor of interpolation parameters (alpha) for each frame
    # first a no_morph section, then a morph section, then a no_morph section
    # morph is linearly interpolated between 0 and 1
    alpha_values = build_sweep_alphas(n_steps, n_steps_no_morph, device=pitch.device, dtype=pitch.dtype)

    # reshape alpha_values to match the shape of pitch and loudness
    alpha_tensor = alpha_values.view(1, n_steps, 1)

    # interpolate mean and std for loudness normalization
    interp_mean = (1 - alpha_tensor) * mean[instrument1] + alpha_tensor * mean[instrument2]
    interp_std = (1 - alpha_tensor) * std[instrument1] + alpha_tensor * std[instrument2]
    loudness_norm = (loudness - interp_mean) / interp_std

    # get amplitudes and filter impulse response from both models
    amplitudes1 = get_amplitudes(model1, pitch, loudness_norm)
    amplitudes2 = get_amplitudes(model2, pitch, loudness_norm)

    # interpolate amplitudes
    amplitudes = (1 - alpha_tensor) * amplitudes1 + alpha_tensor * amplitudes2

    # get filter parameters and convert to impulse responses
    param1 = get_filter_param(model1, pitch, loudness_norm)
    impulse1 = amp_to_impulse_response(param1, model1.block_size)
    param2 = get_filter_param(model2, pitch, loudness_norm)
    impulse2 = amp_to_impulse_response(param2, model2.block_size)

    # interpolate impulse responses
    impulse = (1 - alpha_tensor) * impulse1 + alpha_tensor * impulse2

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

    # filter it with the interpolated impulse response
    noise = fft_convolve(noise, impulse).contiguous()
    noise = noise.reshape(noise.shape[0], -1, 1)

    # combine harmonic and noise to get the final signal
    signal = harmonic + noise

    # apply reverb if desired
    if reverb == True:
        len_signal = signal.shape[1]
        if ir1 is None:
            ir1 = model1.reverb.build_impulse()
        if ir2 is None:
            ir2 = model2.reverb.build_impulse()
        # reverb is applied with a fixed alpha of 0.5, as it is not part of the interpolation sweep
        alpha = 0.5
        ir = get_interpolated_reverb_ir(ir1, ir2, alpha)
        ir = nn.functional.pad(ir, (0, 0, 0, len_signal - model1.reverb.length))
        signal = fft_convolve(signal.squeeze(-1), ir.squeeze(-1)).unsqueeze(-1)
    return signal
    

@torch.no_grad()
def get_interpolated_weights_sweep(state_model_1, state_model_2,
                                   pitch, loudness, mean, std, instrument1, instrument2,
                                   config, n_steps_no_morph, reverb=True, device="cpu"):
    '''
    Generate the output by interpolating the weights of two models.
    The dry signal is affected by the interpolation parameter.
    The sweep is generated in three parts:
        first third of audio: outputs for the first model
        middle third of audio: interpolated weights between the two models, sweeping alpha from 0 to 1
        last third of audio: outputs for the second model
    Arguments:
        state_model_1: state dictionary of the first model
        state_model_2: state dictionary of the second model
        pitch: pitch input tensor
        loudness: loudness input tensor
        mean: mean values for loudness normalization
        std: standard deviation values for loudness normalization
        instrument1: identifier for the first instrument
        instrument2: identifier for the second instrument
        config: configuration dictionary for the model
        n_steps_no_morph: number of steps for non-interpolated sections
        reverb: whether to apply reverb (default: True)
        device: device to build the model on
    Returns:
        signal: interpolated output signal tensor of shape (batch, time, 1)
    '''
    interp_model = build_model_from_state_dict(state_model_1, config, device=device)

    block_size = interp_model.block_size
    n_steps = pitch.shape[1]
    n_audio_samples = n_steps * block_size

    alpha_values = build_sweep_alphas(n_steps, n_steps_no_morph, device=pitch.device, dtype=pitch.dtype)

    output = torch.zeros(pitch.shape[0], n_audio_samples, 1, dtype=pitch.dtype, device=pitch.device)

    for i in range(n_steps):

        alpha = alpha_values[i].item()
        interpolated_weights_dict = interpolate_state_dict(
            state_model_1, state_model_2, alpha
        )

        interp_mean = (1 - alpha) * mean[instrument1] + alpha * mean[instrument2]
        interp_std = (1 - alpha) * std[instrument1] + alpha * std[instrument2]
        loudness_norm = (loudness[:, i:i+1, :] - interp_mean) / interp_std

        frame_output = interp_model.forward_sweep(pitch[:, i:i+1, :], loudness_norm,
                                                  new_weights_dict=interpolated_weights_dict
        )  # (batch, (end-start)*block_size, 1)

        a_start = i * block_size
        a_end = a_start + frame_output.shape[1]
        output[:, a_start:a_end] = frame_output

    if reverb:
        reverb_weights_dict = interpolate_state_dict(state_model_1, state_model_2, 0.5)
        interp_model.load_state_dict(reverb_weights_dict, strict=False)
        output = interp_model.reverb(output)

    return output
