import scipy
import numpy as np

from ddsp.core import multiscale_fft, safe_log


def get_scheduler(start_lr, stop_lr, length):
    def schedule(step):
        if step < length:
            t = step / length
            return start_lr * (stop_lr / start_lr) ** t
        else:
            return stop_lr
    return schedule


def spectral_loss(original, reconstructed, scales, overlap):
    """Multiscale spectral loss (linear + log) between two waveforms."""
    ori_stft = multiscale_fft(original, scales, overlap)
    rec_stft = multiscale_fft(reconstructed, scales, overlap)

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        loss += (s_x - s_y).abs().mean()
        loss += (safe_log(s_x) - safe_log(s_y)).abs().mean()
    return loss


def high_pass_filter(signal, cutoff_frequency, fs=16000, taps=129):
    b = scipy.signal.firwin(taps, cutoff_frequency, fs=fs, pass_zero="highpass")
    a = np.zeros_like(b)
    a[0] = 1
    filtered_signal = scipy.signal.lfilter(b, a, signal)
    return filtered_signal

     