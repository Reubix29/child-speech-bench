"""
An example script to generate representations for a dataset using mHuBERT, HuBERT discrete, and MFCCs.

Author: Reuben Smit
Date: 2025
"""
from pathlib import Path
import torch
import numpy as np
from transformers import AutoProcessor, HubertModel
import helpers
rep_registry = {}
dist_registry = {}

def register_rep_fn(name):
    """Decorator to register a representation function."""
    def decorator(func):
        helpers.validate(func)
        rep_registry[name] = func
        return func
    return decorator

def register_dist_fn(name):
    """Decorator to register a distance function."""
    def decorator(func):
        helpers.validate(func)
        dist_registry[name] = func
        return func
    return decorator


"""
===============================================================================================================================
Representation Functions
===============================================================================================================================
"""

@register_rep_fn("mhubert")
def generate_mhubert_representations(audio, processor=None, mhubert=None):
    """Generate mHuBERT representations for the audio tensor."""

    if processor is None or mhubert is None:
        processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        mhubert = HubertModel.from_pretrained("utter-project/mHuBERT-147")

    with torch.inference_mode():
        outputs = mhubert(audio, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    
    return hidden_states

@register_rep_fn("mfccs")
def generate_mfccs(audio):
    #TODO
    pass

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
===============================================================================================================================
Distance Functions
===============================================================================================================================
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
    #TODO
    pass

def get_distance_function(name):
    """Retrieve a registered distance function by name."""
    if name in dist_registry:
        return dist_registry[name]
    else:
        raise ValueError(f"Distance function '{name}' not found in registry. Available functions: {list(rep_registry.keys())}")
    
