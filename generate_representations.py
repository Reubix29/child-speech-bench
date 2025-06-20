"""
An example script to generate representations for a dataset using mHuBERT, HuBERT discrete, and MFCCs.

Author: Reuben Smit
Date: 2025
"""
from pathlib import Path
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, HubertModel
import helpers
import editdistance

rep_registry, dist_registry, models = {}, {}, {}

def register_rep_fn(name):
    """Decorator to register a representation function."""
    def decorator(func):
        helpers.validate_rep_fn(func)
        rep_registry[name] = func
        return func
    return decorator

def register_dist_fn(name):
    """Decorator to register a distance function."""
    def decorator(func):
        helpers.validate_dist_fn(func)
        dist_registry[name] = func
        return func
    return decorator


def initialize_models():
    """Initialize and store models and processors globally."""
    global models

    # Add models and processors to the global models dictionary

    models["mhubert_processor"] = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    models["mhubert_model"] = HubertModel.from_pretrained("utter-project/mHuBERT-147")
    
    print("Models initialized successfully.")


def preemphasis(signal, coeff=0.97):
    """Perform preemphasis on the input `signal`."""    
    return np.append(signal[0], signal[1:] - coeff*signal[:-1])

"""
=============================================
Representation Functions
=============================================
"""

@register_rep_fn("mhubert")
def generate_mhubert_representations(audio):
    """Generate mHuBERT representations for the audio tensor."""
    processor = models.get("mhubert_processor")
    mhubert = models.get("mhubert_model")

    if processor is None or mhubert is None:
        raise RuntimeError("Models not initialized. Call 'initialize_models()' first.")

    with torch.inference_mode():
        outputs = mhubert(audio, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    
    return hidden_states

@register_rep_fn("mfccs")
def generate_mfccs(audio):
    sample_rate = 16000
    waveform = audio.squeeze(0).numpy()
    signal = preemphasis(waveform)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=24,
        n_fft=int(np.floor(0.025*sample_rate)),
        hop_length=int(np.floor(0.01*sample_rate)),
        fmin=64,
        fmax=8000
        )

    mfccs = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_spectrogram),
        sr=sample_rate,
        n_mfcc=13
        )
    
    return mfccs

@register_rep_fn("hubert_discrete")
def generate_speech_units(audio):
    #TODO
    pass

@register_rep_fn("whisper")
def generate_whisper_representations(audio, processor=None, model=None):
    #TODO
    pass

def get_representation_function(name):
    """Retrieve a registered representation function by name."""
    if name in rep_registry:
        return rep_registry[name]
    else:
        raise ValueError(f"Representation function '{name}' not found in registry. Available functions: {list(rep_registry.keys())}")

"""
=============================================
Distance Functions
=============================================
"""

@register_dist_fn("dtw")
def dtw_distance(x, y):
    """
    Compute Dynamic Time Warping (DTW) distance between two sequences.
    
    Args:
        x (np.ndarray): First sequence.
        y (np.ndarray): Second sequence.
    
    Returns:
        float: DTW distance between x and y.
    """
    #TODO
    pass

@register_dist_fn("ned")
def ned(x, y):
    """
    Compute Normalized Edit Distance (NED) between two sequences.
    
    Args:
        x (np.ndarray): First sequence.
        y (np.ndarray): Second sequence.
    
    Returns:
        float: NED between x and y.
    """
    return editdistance.eval(x, y) / max(len(x), len(y))

def get_distance_function(name):
    """Retrieve a registered distance function by name."""
    if name in dist_registry:
        return dist_registry[name]
    else:
        raise ValueError(f"Distance function '{name}' not found in registry. Available functions: {list(rep_registry.keys())}")
    
