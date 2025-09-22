"""
An example script to generate representations for a dataset using mHuBERT, HuBERT discrete, and MFCCs.

Author: Reuben Smit
Date: 2025
"""
import re
from pathlib import Path
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, HubertModel, AutoModelForSpeechSeq2Seq, pipeline
from sklearn.linear_model import LogisticRegression
import helpers
import editdistance
from speech_dtw.qbe import parallel_dtw_sweep_min as dtw

rep_registry, rep_type_registry, dist_registry, models = {}, {}, {}, {}

def register_rep_fn(name, rep_type):
    """Decorator to register a representation function."""
    def decorator(func):
        helpers.validate_rep_fn(func)
        rep_registry[name] = func
        rep_type_registry[name] = rep_type
        return func
    return decorator

def register_dist_fn(name):
    """Decorator to register a distance function."""
    def decorator(func):
        helpers.validate_dist_fn(func)
        dist_registry[name] = func
        return func
    return decorator

def initialize_models(rep_fn_names=None):
    """Initialize and store models and processors globally."""
    global models
    # Add models and processors to the global models dictionary
    if "mhubert" in rep_fn_names:
        # mHuBERT
        models["mhubert_processor"] = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        models["mhubert_model"] = HubertModel.from_pretrained("utter-project/mHuBERT-147")
    elif "hubert_discrete" in rep_fn_names:
        # HuBERT Discrete (https://github.com/bshall/dusted)
        hubert, encode = torch.hub.load("bshall/dusted:main", "hubert", language="english", trust_repo=True)
        kmeans, segment = torch.hub.load("bshall/dusted:main", "kmeans", language="english", trust_repo=True)
        models["hubert_discrete"] = hubert
        models["hubert_discrete_kmeans"] = kmeans
        models["hubert_discrete_encode"] = encode
        models["hubert_discrete_segment"] = segment 
    elif "whisper" in rep_fn_names:
        # Whisper ASR
        whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-small", torch_dtype=torch.float32, low_cpu_mem_usage=False, use_safetensors=True
            )
        whisper_processor = AutoProcessor.from_pretrained("openai/whisper-small")
        whisper_processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
        whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=whisper,
                tokenizer=whisper_processor.tokenizer,
                feature_extractor=whisper_processor.feature_extractor,
                torch_dtype=torch.float32,
                device="cpu",
            )
        models["whisper"] = whisper_pipe
        models["whisper_processor"] = whisper_processor
    elif "mfccs" in rep_fn_names:
        pass
    else:
        raise ValueError("No valid representation function names provided for model initialization.")
    print("Models initialized successfully.")


def preemphasis(signal, coeff=0.97):
    """Perform preemphasis on the input signal."""    
    return np.append(signal[0], signal[1:] - coeff*signal[:-1])

"""
=============================================
Representation Functions
=============================================
"""

@register_rep_fn("mhubert", "continuous")
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

@register_rep_fn("mfccs", "continuous")
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
    
    mfccs = mfccs.T 
    
    return [mfccs]

@register_rep_fn("hubert_discrete", "discrete")
def generate_speech_units(audio):
    hubert = models.get("hubert_discrete")
    encode = models.get("hubert_discrete_encode")
    kmeans = models.get("hubert_discrete_kmeans")
    segment = models.get("hubert_discrete_segment")
    if hubert is None or encode is None or kmeans is None or segment is None:
        raise RuntimeError("Models not initialized. Call 'initialize_models()' first.")

 
    audio = audio.unsqueeze(0)  

    # Pass audio through the encode function
    x = encode(hubert, audio).squeeze().cpu().numpy()

    # Pass encoded output through the segment function
    units, _ = segment(x, kmeans.cluster_centers_, gamma=0.2)  # gamma=0.2 recommended in paper

    return units

@register_rep_fn("whisper", "discrete")
def generate_whisper_representations(audio):
    whisper = models.get("whisper")

    transcription = whisper(audio.numpy().squeeze(), generate_kwargs={"language": "english", "forced_decoder_ids": None})["text"]
    
    # Decode to text
    cleaned_transcription = re.sub(r"[^\w\s]", "", transcription).lower()
    return cleaned_transcription

def get_representation_function(name):
    """Retrieve a registered representation function by name."""
    if name in rep_registry:
        return rep_registry[name]
    else:
        raise ValueError(f"Representation function '{name}' not found in registry. Available functions: {list(rep_registry.keys())}")
    
def get_representation_function_type(name):
    """Retrieve a registered representation function type."""
    if name in rep_type_registry:
        return rep_type_registry[name]
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
    dist = dtw(x, y)
    return dist
    

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
    
