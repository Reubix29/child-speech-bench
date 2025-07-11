"""
Functions for performing DTW Barycenter Averaging (DBA) on a sequence of embeddings. DBA finds an approximate centroid for a set of sequences by iteratively aligning the sequences with DTW and averaging the aligned vectors.

Author: Reuben Smit
Contact: reubensmit.1@gmail.com
Date: 2025
"""

from os import path
import numpy as np
import sys

basedir = path.join(path.dirname(__file__), "..")
sys.path.append(basedir)

from speech_dtw import _dtw

dtw_path_function = _dtw.multivariate_dtw


def dba(sequences, n_iter=10):
    """
    Perform DTW Barycenter Averaging (DBA) on a sequence of embeddings.

    Parameters:
        sequences: list of numpy arrays
            List of sequences to average.
        n_iter: int
            Number of iterations to perform.
        dur_normalize: bool
            Normalize by the duration of the sequences.

    Returns:
        dba_sequence: numpy array
            The DBA sequence.
    """
    sequences = np.array(sequences).shape
    # Get the median length of sequences
    seq_lengths = [seq.shape[1] for seq in sequences]
    median_length = int(np.median(seq_lengths))

    # Select the first median length sequence as the reference sequence
    reference_index = seq_lengths.index(median_length)
    reference_sequence = sequences[reference_index]

    # Perform the DBA iterations
    for i in range(n_iter):
        # Compute the DTW path between the reference sequence and each sequence
        dtw_paths = []
        for sequence in sequences:
            path = dtw_path_function(reference_sequence, sequence)[0] 
            dtw_paths.append(path) 
        
        # Compute the DBA sequence
        dba_sequence = reference_sequence.copy() # Start with the reference sequence vectors
        dba_counts = np.zeros(reference_sequence.shape[1]) # Keep track of the number of vectors summed for each coordinate
        for j in range(len(sequences)):
            path = dtw_paths[j]
            sequence = sequences[j]
            for k in range(len(path)):
                dba_sequence[:, path[k][0]] = np.sum([dba_sequence[:, path[k][0]], sequence[:, path[k][1]]], axis=0)
                dba_counts[path[k][0]] += 1
        dba_sequence = dba_sequence / dba_counts

        # Update the reference sequence
        reference_sequence = dba_sequence

    return dba_sequence