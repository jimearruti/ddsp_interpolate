import pathlib
import json

import librosa as li
import pyloudnorm as pyln
import numpy as np
import torch
import yaml
from tqdm import tqdm

from ddsp.core import extract_loudness, extract_pitch
from ddsp.utils import high_pass_filter
from effortless_config import Config


def save_subdataset(signals, pitches, loudness, out_path):
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "signals.npy", signals)
    np.save(out_path / "pitches.npy", pitches)
    np.save(out_path / "loudness.npy", loudness)


class DatasetMultiInstrument(torch.utils.data.Dataset):
    def __init__(self, out_dir, instrument, subset="train"):
        super().__init__()
        data_path = pathlib.Path(out_dir) / instrument / subset
        self.signals = np.load(data_path / "signals.npy")
        self.pitches = np.load(data_path / "pitches.npy")
        self.loudness = np.load(data_path / "loudness.npy")

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitches[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l


def preprocess(
    f, sampling_rate, block_size, signal_length, oneshot, n_fft, **_):
    '''Preprocess a single audio file.
    Args:
        f: Path to the audio file.
        sampling_rate: Sampling rate to load the audio file.
        block_size: Block size for pitch and loudness extraction.
        signal_length: Length of the output signal segments.
        oneshot: If True, only process the first segment of the audio file.
        nfft: Number of FFT bins for loudness extraction.'''

    x, _ = li.load(f, sr=sampling_rate)
    x = pyln.normalize.peak(x, -1.0)
    x = high_pass_filter(x, 80, fs=sampling_rate)

    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size, n_fft=n_fft)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


def process_files(files, config):
    if not files:
        return None
    
    signals: list[np.ndarray] = []
    pitches: list[np.ndarray] = []
    loudness: list[np.ndarray] = []

    progress_bar = tqdm(files)
    for f in progress_bar:
        progress_bar.set_description(str(f))
        x, p, l = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitches.append(p)
        loudness.append(l)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitches = np.concatenate(pitches, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    return (signals, pitches, loudness)


def preprocess_dataset(config):

    root_out_path = pathlib.Path(config["preprocess"]["out_dir"])

    with open(root_out_path / "split_files.json", "r") as f:
        split_data = json.load(f)

    instruments = config["data"]["instruments"]

    for instrument in instruments:
        split_data_instrument = split_data[instrument]
        train_files = split_data_instrument["train"]
        val_files = split_data_instrument["val"]
        test_files = split_data_instrument["test"]

        if not train_files:
            raise ValueError(
                f"No training files for '{instrument}' — dataset may be too small to split."
            )
        out_path = root_out_path / instrument

        for subset, subset_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            result = process_files(subset_files, config)
            if result is None:
                print(f"Skipping '{subset}' for '{instrument}': no files in split.")
                continue
            signals, pitches, loudness = result
            save_subdataset(signals, pitches, loudness, out_path / subset)


if __name__ == "__main__":
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()

    with open(args.CONFIG, "r") as config_file:
        config = yaml.safe_load(config_file)

    preprocess_dataset(config)