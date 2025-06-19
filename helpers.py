import inspect
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import torchaudio
from torchaudio.transforms import Resample
import torch
import tempfile
import shutil

def validate(func):
    sig = inspect.signature(func)
    params = sig.parameters

    # Check number of parameters
    if len(params) != 1:
        raise TypeError(f"Function '{func.__name__}' must accept exactly one argument.")

    # Check parameter type annotation
    param = next(iter(params.values()))
    if param.annotation not in (inspect._empty, str, Path):
        raise TypeError(f"Argument of '{func.__name__}' should be of type 'pathlib.Path', got {param.annotation}")

    # Check return annotation
    if sig.return_annotation not in (inspect._empty, np.ndarray):
        raise TypeError(f"Function '{func.__name__}' should return 'np.ndarray', got {sig.return_annotation}")

    print(f"Function '{func.__name__}' validated successfully.")

def convert_to_wav(input_file, output_file):
    """
    Converts any audio file supported by pydub to WAV.
    """
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")

def preprocess(file_path, resample_rate=16000, overwrite=False):
    """
    Loads audio, converts to WAV if needed, resamples to 16kHz, and ensures mono.

    Args:
    - file_path (str or Path): Path to the original audio file.
    - resample_rate (int): Desired sampling rate.
    - overwrite (bool): If True, allow overwriting the original file.

    Returns:
    - audio (torch.Tensor): Processed waveform.
    - sr (int): Sampling rate.
    - used_path (Path): Path of the file actually loaded.
    """
    file_path = Path(file_path)
    original_suffix = file_path.suffix.lower()
    needs_conversion = original_suffix != ".wav"

    if needs_conversion:
        # Choose target path
        if overwrite:
            output_path = file_path.with_suffix(".wav")
        else:
            temp_dir = Path(tempfile.mkdtemp())
            output_path = temp_dir / (file_path.stem + ".wav")

        convert_to_wav(file_path, output_path)
        used_path = output_path
    else:
        used_path = file_path

    audio, sr = torchaudio.load(used_path)

    # Resample if needed
    if sr != resample_rate:
        resampler = Resample(orig_freq=sr, new_freq=resample_rate)
        audio = resampler(audio)
        sr = resample_rate

    # Convert to mono if needed
    if audio.size(0) > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    return audio, sr, used_path



