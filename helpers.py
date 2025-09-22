import inspect
import editdistance
import random
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import torchaudio
from torchaudio.transforms import Resample
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import torch
from tqdm import tqdm
import tempfile
from speech_dtw import _dtw


def cache_representations(rep_fn, dataset_path, template_dirname, rep_type="continuous"):
    """
    Helper function to cache representations for a dataset.
    """

    root_path = Path("data/cache/{rep_fn.__name__}")
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Check whether the representations are already cached
    if (root_path / "templates" / template_dirname).exists() and (root_path / "query" / "dev").exists() and (root_path / "query" / "test").exists():
        print(f"Representations for {rep_fn.__name__} already cached. Skipping caching step.")
        return

    # Cache the template representations
    cache_template_path = root_path / "templates"
    cache_template_path.mkdir(parents=True, exist_ok=True)

    for cls in tqdm(list((dataset_path / "templates" / template_dirname)), desc="Caching templates"):
        for template_file in cls.glob("*.wav"):
            audio, _, _ = preprocess(template_file)
            representation = rep_fn(audio)
            save_path = cache_template_path / cls.name
            save_path.mkdir(parents=True, exist_ok=True)
            if rep_type == "discrete":
                # For discrete representations, save as a textfile
                with open(save_path / (template_file.stem + ".pt"), "w") as f:
                    f.write(" ".join(map(str, representation)))
            else:
                torch.save(representation, save_path / (template_file.stem + ".pt"))

    for set in ["dev", "test"]:
        for cls in tqdm(list((dataset_path / "query" / set)), desc=f"Caching {set} set"):
            for audio_file in cls.glob("*.wav"):
                audio, _, _ = preprocess(audio_file)
                representation = rep_fn(audio)
                save_path = root_path / "query" / set / cls.name
                save_path.mkdir(parents=True, exist_ok=True)
                if rep_type == "discrete":
                # For discrete representations, save as a textfile
                    with open(save_path / (template_file.stem + ".pt"), "w") as f:
                        f.write(" ".join(map(str, representation))) 
                else:
                    torch.save(representation, save_path / (template_file.stem + ".pt"))

def load_cached_representation(rep_fn, dataset_path):
    """
    Helper function to load cached representations for a dataset.
    """
    #TODO
    pass

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

def validate_dist_fn(func):
    sig = inspect.signature(func)
    params = sig.parameters

    # Check number of parameters
    if len(params) != 2:
        raise TypeError(f"Function '{func.__name__}' must accept exactly two arguments.")

    # Check return annotation
    if sig.return_annotation not in (inspect._empty, float):
        raise TypeError(f"Function '{func.__name__}' should return 'float', got {sig.return_annotation}")


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

def generate_neighbours(s, codebook_size=100):
    codebook = range(codebook_size)
    s = list(s)
    neighbors = set()
    for i in range(len(s)):
        # Deletion
        neighbors.add(tuple(s[:i] + s[i+1:]))
        # Substitution
        for c in codebook:
            neighbors.add(tuple(s[:i] + [c] + s[i+1:]))
    for i in range(len(s)+1):
        # Insertion
        for c in codebook:
            neighbors.add(tuple(s[:i] + [c] + s[i:]))
    return list(neighbors)

def edb(sequences, max_iter=100, codebook_size=100):
    """
    Edit Distance Barycenter Averaging (EDBA) for discrete sequences.
    """
    median_length = int(np.median([len(s) for s in sequences]))
    median_sequences = [s for s in sequences if len(s) == median_length]
    current = random.choice(median_sequences) if median_sequences else random.choice(sequences)

    best_score = sum(editdistance.eval(current, s) for s in sequences)

    for _ in range(max_iter):
        neighbors = generate_neighbours(current, codebook_size)
        improved = False
        for n in tqdm(neighbors, desc="Searching through neighbours", leave=False):
            score = sum(editdistance.eval(n, s) for s in sequences)
            if score < best_score:
                current = n
                best_score = score
                improved = True
                break
        if not improved:
            break

    return list(current)

def dba(embedding_sequences, n_iter=1):
    """
    Perform Dynamic Time Warping (DTW) Barycenter Averaging (DBA) on a set of embeddings.

    Args:
    - embeddings_sequences (list): List of sequences of embeddings.
    - n_iter (int): Number of iterations to perform.

    Returns:
    - barycenter (torch.Tensor): Barycenter of the embeddings.
    """
    dtw_path_function = _dtw.multivariate_dtw

    # Get the median length of sequences
    # if embedding sequences is a list, make it a np array
    if isinstance(embedding_sequences, list):
        embedding_sequences = np.array(embedding_sequences, dtype=object)
    seq_lengths = [seq.shape[1] for seq in embedding_sequences]

    median_length = int(np.median(seq_lengths))
    closest_length = min(seq_lengths, key=lambda x: abs(x - median_length)) 
    reference_index = seq_lengths.index(closest_length)
    reference_sequence = np.asarray(embedding_sequences[reference_index], dtype=np.float64).squeeze()

    # Perform the DBA iterations
    for n in range(n_iter):
        # Compute the DTW path between the reference sequence and each sequence
        dtw_paths = []
        for sequence in embedding_sequences:
            path = dtw_path_function(reference_sequence, np.array(sequence, dtype=np.float64).squeeze())[0]
            dtw_paths.append(path)
        
        # Compute the DBA sequence
        dba_sequence = reference_sequence.copy() # Start with the reference sequence vectors
        dba_counts = np.zeros(reference_sequence.shape[1]) # Keep track of the number of vectors summed for each coordinate
        for j in range(len(embedding_sequences)):
            path = dtw_paths[j]
            sequence = embedding_sequences[j].squeeze()
            for k in range(len(path)):
                i, j = path[k]
                if sequence[j, :].dtype == torch.float32:
                    sequence_np = sequence[j, :].detach().numpy()
                else:
                    sequence_np = sequence[j, :]
                dba_sequence[i, :] += sequence_np
                dba_counts[i] += 1
        
        # Avoid division by zero
        dba_counts[dba_counts == 0] = 1  
        dba_sequence = dba_sequence / dba_counts

        # Update the reference sequence
        reference_sequence = dba_sequence

    return [dba_sequence]

def calculate_metrics(file_path, template_dirname, rep_fn, dist_fn, template_ranking, rep_type="continuous"):
    """
    Calculate classification metrics for a given dataset, using a representation function and a distance function.

    Args:
    - file_path (str or Path): Path to the dataset.
    - template_dirname (str): Name of the template subdirectory.
    - rep_fn (callable): Function to generate representations.
    - dist_fn (callable): Function to calculate distances.
    - template_ranking (str): "avg" or "min".

    Returns:
    - metrics (dict): Dictionary containing calculated metrics.
    """
    template_path = file_path / "templates" / template_dirname
    all_distances = {"dev": {}, "test": {}}
    groundtruths = {"dev": {}, "test": {}}

    # ---- Collect distances and ground truths ----
    for split in ["dev", "test"]:
        querypath = file_path / "query" / split
        for bucket in tqdm(list(querypath.iterdir()), desc=f"Processing {split} set"):
            if bucket.is_dir():
                # Load template features
                template_features = []
                templates = list((template_path / bucket.name).glob("*.wav"))
                if not templates:
                    raise FileNotFoundError(f"No templates found in {template_path / bucket.name}")
                for template_file in templates:
                    template_audio, _, _ = preprocess(template_file)
                    template_features.append(rep_fn(template_audio))

                if template_ranking == "barycentre":
                    if rep_type == "continuous":
                        template_features = dba(template_features)
                    else:
                        template_features = edb(template_features)
                # Query distances
                bucket_distances = []
                query_files = sorted(bucket.glob("*.wav"))
                query_names = [f.stem for f in query_files]
                for audio_file in query_files:
                    audio, sr, _ = preprocess(audio_file)
                    query_features = rep_fn(audio)
                    if template_ranking == "avg":
                        distance_val = float(np.mean([dist_fn(query_features, tf) for tf in template_features]))
                    elif template_ranking == "min":
                        distance_val = float(np.min([dist_fn(query_features, tf) for tf in template_features]))
                    elif template_ranking == "barycentre":
                        if rep_type == "discrete":
                            distance_val = dist_fn(query_features, template_features)
                        else:  
                            distance_val = dist_fn(query_features, template_features)
                            distance_val = distance_val[0][0]
                    else:
                        raise ValueError(f"Unknown template_ranking: {template_ranking}")
                    bucket_distances.append(distance_val)

                all_distances[split][bucket.name] = bucket_distances
                groundtruths[split][bucket.name] = [1 if bucket.name == qn.split("_")[0] else 0 for qn in query_names]

    # ---- Find best threshold on dev (max balanced accuracy) ----
    dev_distances = np.concatenate([np.array(all_distances["dev"][cls]) for cls in all_distances["dev"]])
    dev_groundtruth = np.concatenate([np.array(groundtruths["dev"][cls]) for cls in groundtruths["dev"]])
    thresholds = np.linspace(0, 1, 200)
    baccs, recs, precs, f1s = {}, {}, {}, {}
    for th in thresholds:
        preds = (dev_distances < th).astype(int)
        baccs[th] = balanced_accuracy_score(dev_groundtruth, preds)
        recs[th] = recall_score(dev_groundtruth, preds, zero_division=0)
        precs[th] = precision_score(dev_groundtruth, preds, zero_division=0)
        f1s[th] = f1_score(dev_groundtruth, preds, zero_division=0)
    best_threshold = max(baccs, key=baccs.get)
 

    # ---- Evaluate on test, per class, then average ----
    test_precision, test_recall, test_f1, test_roc, test_far, test_mr = [], [], [], [], [], []

    for cls in all_distances["test"]:
        distances = np.array(all_distances["test"][cls])
        groundtruth = np.array(groundtruths["test"][cls])
        predictions = (distances < best_threshold).astype(int)

        # Core metrics
        if len(np.unique(groundtruth)) > 1:
            test_roc.append(roc_auc_score(groundtruth, 1 - distances))
        else:
            # If only one class, AUC is undefined → skip
            test_roc.append(np.nan)

        test_precision.append(precision_score(groundtruth, predictions, zero_division=0))
        test_recall.append(recall_score(groundtruth, predictions, zero_division=0))
        test_f1.append(f1_score(groundtruth, predictions, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(groundtruth, predictions, labels=[0,1]).ravel()
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        test_far.append(false_alarm_rate)
        test_mr.append(miss_rate)

    # Aggregate across classes
    metrics = {
        "Recall": np.nanmean(test_recall),
        "Precision": np.nanmean(test_precision),
        "F1": np.nanmean(test_f1),
        "ROC AUC": np.nanmean(test_roc),
        "False Alarm Rate": np.nanmean(test_far),
        "Miss Rate": np.nanmean(test_mr),
        "Balanced Accuracy": balanced_accuracy_score(
            np.concatenate([groundtruths["test"][cls] for cls in groundtruths["test"]]),
            (np.concatenate([all_distances["test"][cls] for cls in all_distances["test"]]) < best_threshold).astype(int),
        ),
    }

    return metrics

    
