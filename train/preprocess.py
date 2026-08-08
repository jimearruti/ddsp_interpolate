import json
import pathlib
import yaml

import librosa as li
import numpy as np
import pyloudnorm as pyln
import torch
from tqdm import tqdm

from ddsp.core import extract_loudness, extract_pitch
from ddsp.utils import high_pass_filter
from effortless_config import Config


def save_subdataset(signals, pitches, loudness, out_path):
    '''
    Save the preprocessed signals, pitches, and loudness to .npy files in the specified output path.
    Args:
        signals (np.ndarray): The preprocessed audio signals.
        pitches (np.ndarray): The extracted pitch features.
        loudness (np.ndarray): The extracted loudness features.
        out_path (pathlib.Path): The output path where the files will be saved.
    '''
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "signals.npy", signals)
    np.save(out_path / "pitches.npy", pitches)
    np.save(out_path / "loudness.npy", loudness)

class DatasetMultiInstrument(torch.utils.data.Dataset):
    '''
    A PyTorch Dataset class for loading preprocessed audio data for multiple instruments.
    Attributes:
        signals (np.ndarray): The preprocessed audio signals.
        pitches (np.ndarray): The extracted pitch features.
        loudness (np.ndarray): The extracted loudness features.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the signals, pitches, and loudness for the sample at the specified index.
    '''
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
    '''Preprocess a single audio file. Normalise to -1 dB, high-pass filter at 80 Hz, 
        and extract pitch and loudness features. Divide the audio into segments of length
        `signal_length` and return the preprocessed signals, pitches, and loudness.

    Args:
        f: Path to the audio file.
        sampling_rate: Sampling rate to load the audio file.
        block_size: Block size for pitch and loudness extraction.
        signal_length: Length of the output signal segments.
        oneshot: If True, only process the first segment of the audio file.
        nfft: Number of FFT bins for loudness extraction.
        
    Returns:
        x: Preprocessed audio signal segments.
        pitch: Extracted pitch features for each segment.
        loudness: Extracted loudness features for each segment.
    '''
    # load the audio file at the config sampling rate
    x, _ = li.load(f, sr=sampling_rate)
    # peak normalize to -1 dB
    x = pyln.normalize.peak(x, -1.0)
    # apply high pass filter at 80 Hz to remove low frequency noise
    x = high_pass_filter(x, 80, fs=sampling_rate)

    # Pad the signal to be a multiple of signal_length
    N = (signal_length - len(x) % signal_length) % signal_length
    # Pad the signal with zeros at the end
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    # Extract pitch and loudness features
    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size, n_fft=n_fft)

    # Split into segments assuming signal_length is a multiple of block_size,
    # x to shape (num_segments, signal_length)
    x = x.reshape(-1, signal_length)
    # pitch and loudness to shape (num_segments, signal_length // block_size)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


def process_files(files, config):
    '''
    Process a list of audio files and return the preprocessed signals, pitches, and loudness.
    Args:
        files: List of paths to the audio files.
        config: Configuration dictionary containing preprocessing parameters.
    Returns:
        signals (np.ndarray): The preprocessed audio signal segments.
        pitches (np.ndarray): The extracted pitch features for each segment.
        loudness (np.ndarray): The extracted loudness features for each segment.
    '''
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
    '''
    Preprocess the dataset for each instrument and split (train, val, test) 
    and save the preprocessed data to .npy files.
    
    Args:
        config: Configuration dictionary containing preprocessing parameters and dataset information.
    Returns:
        None: Saves the preprocessed signals, pitches, and loudness to .npy files in the specified output directory.
    '''

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