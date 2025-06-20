import inspect
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import torchaudio
from torchaudio.transforms import Resample
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import torch
import tempfile
from generate_representations import initialize_models


def validate_rep_fn(func):
    sig = inspect.signature(func)
    params = sig.parameters

    # Check number of parameters
    if len(params) != 1:
        raise TypeError(f"Function '{func.__name__}' must accept exactly one argument.")

    # Check parameter type annotation
    param = next(iter(params.values()))
    if param.annotation not in (inspect._empty, torch.Tensor):
        raise TypeError(f"Argument of '{func.__name__}' should be of type 'torch.Tensor', got {param.annotation}")

    # Check return annotation
    if sig.return_annotation not in (inspect._empty, np.ndarray):
        raise TypeError(f"Function '{func.__name__}' should return 'np.ndarray', got {sig.return_annotation}")

    print(f"Function '{func.__name__}' validated successfully.")

def validate_rep_fn(func):
    sig = inspect.signature(func)
    params = sig.parameters

    # Check number of parameters
    if len(params) != 2:
        raise TypeError(f"Function '{func.__name__}' must accept exactly two arguments.")

    # Check return annotation
    if sig.return_annotation not in (inspect._empty, float):
        raise TypeError(f"Function '{func.__name__}' should return 'float', got {sig.return_annotation}")

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

def calculate_metrics(file_path, template_dirname, rep_fn, dist_fn, template_ranking):
    """
    Calculate classification metrics for a given dataset, using a representation function and a distance function.

    Args:
    - file_path (str or Path): Path to the dataset.
    - rep_fn (callable): Function to generate representations.
    - dist_fn (callable): Function to calculate distances.

    Returns:
    - metrics (dict): Dictionary containing calculated metrics.
    """
    audio, sr, _ = preprocess(file_path)
    template_path = file_path / "templates" / template_dirname
    # First calculate the distances for the dev and test sets
    all_distances = {"dev": [], "test": []}
    groundtruths = {"dev": [], "test": []}

    for set in ["dev", "test"]:
        for bucket in (file_path / "query" / set).iterdir():
            if bucket.is_dir():
                # Calculate representations for templates
                template_features, bucket_distances = [], []
                templates = list((template_path / bucket.name).glob("*.wav"))
                if not templates:
                    raise FileNotFoundError(f"No templates found in {template_path / bucket.name}")
                else:
                    for template_file in templates:
                        template_audio, _, _ = preprocess(template_file)
                        template_features.append(rep_fn(template_audio))
                for audio_file in bucket.glob("*.wav"):
                    audio, sr, _ = preprocess(audio_file)
                    query_features = rep_fn(audio)
                    # Calculate distances using the distance function
                    distances = [dist_fn(query_features, template_feature) for template_feature in template_features]
                    if template_ranking == "avg":
                        distances = np.mean(distances, axis=0)
                    elif template_ranking == "min":
                        distances = np.min(distances, axis=0)
                    bucket_distances.append(distances)
                all_distances[set].append(bucket_distances)
                # Append ground truth labels
                groundtruths[set].append([1 if bucket.name == query_file.stem else 0 for query_file in bucket.glob("*.wav")])
                

    # Calculate the threshold that maximises the balanced accuracy on the dev set
    dev_distances = np.array(all_distances["dev"])
    dev_groundtruths = np.array(groundtruths["dev"])

    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0
    best_bacc = 0

    for threshold in thresholds:
        predictions = (dev_distances < threshold).astype(int)
        bacc = balanced_accuracy_score(dev_groundtruths.flatten(), predictions.flatten())
        if bacc > best_bacc:
            best_bacc = bacc
            best_threshold = threshold
    print(f"Best threshold: {best_threshold}, Balanced Accuracy on dev set: {best_bacc}")

    # Using the best threshold, calculate distances on the test set
    test_distances = np.array(all_distances["test"]).flatten()
    test_groundtruths = np.array(groundtruths["test"]).flatten()
    test_predictions = (test_distances < best_threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(test_groundtruths, test_distances),
        "precision": precision_score(test_groundtruths, test_predictions),
        "recall": recall_score(test_groundtruths, test_predictions),
        "f1": f1_score(test_groundtruths, test_predictions),
        "balanced_accuracy": balanced_accuracy_score(test_groundtruths, test_predictions)
    }
    return metrics


    
