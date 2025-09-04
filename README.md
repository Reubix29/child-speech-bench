# Child speech benchmark
A benchmark for evaluating speech representation methods on child speech, with a focus on low-resource and few-shot learning scenarios.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2507.12217)

## Overview

Imagine a child in a low-resource setting is shown a word on a screen and asked to read it aloud. The audio they produce is recorded and then compared to **adult template recordings** of that same word. By measuring the distance between the child’s recording and the templates, we decide whether the word they said matches the expected one.

![Few-shot paper example](fig/system.png)

This benchmark evaluates how well different **speech representation** and **distance calculation** methods can classify the spoken word as a match or not. The process works as follows:

1. A **binary classifier** makes match/no-match decisions based on the computed distances.

2. A **threshold** is selected to maximize **balanced accuracy** on the development set.

3. This threshold is then applied to classify samples in the test set.

4. The system reports performance using **recall**, **precision**, **F1 score**, **balanced accuracy**, and **AUC** on the test set.

The goal is to find the combination of representation and distance method that gives the best **balanced accuracy**, especially in low-resource speech settings.

## Quick Start
### Programmatic Usage

1. Install the dependencies:

    - Make the setup file executable by running ``chmod +x setup.sh``, then run ``setup.sh``.

    Alternatively, manually install the dependencies:
    - Run ``conda env create -f environment.yml``
    - Run ``cd dtw`` and then ``make`` to compile the cython code for the dynamic time warping module used in the benchmark.


2. Download an applicable few-shot isolated word dataset, like [ISACS](https://reubix29.github.io/isolated-afrikaans-child-speech/), or [use your own](#use-your-own-dataset).

3. In `map.yml`:
    - Add the local path to the chosen dataset.
    - Select a representation method:
        - **mhubert**: a multilingual HuBERT model.
        - **hubert_discrete**: discrete speech codes.
        - **mfccs**: mel-frequency cepstral coefficients.
        - **whisper**: English Whisper transcriptions.
        - or [create your own](#test-a-custom-representation-or-distance-method)
    - Select a distance function:
        - **dtw**: dynamic time warping. This is the default for continuous representations. 
        - **ned**: normalised edit distance. This is the default for discrete representations.
        - or [create your own](#test-a-custom-representation-or-distance-method)
    - Choose whether to average (**avg**) the distances from the query to the template representations, or select the minimum distance (**min**).
    - Choose the speaker you would like to use. In [ISACS](https://reubix29.github.io/isolated-afrikaans-child-speech/), there are four speakers available:
        - sp_1
        - sp_2
        - sp_3
        - child

5. In your terminal, run `python run_benchmark.py`, which should result in a list of metrics.

### Use your own dataset

If you want to test methods on your own data, the dataset should be structured as follows:

```dataset/
├── query/
│   ├── dev/
│   │   ├── word_1/
│   │   │   ├── pos_sample1.wav
│   │   │   ├── pos_sample2.wav
│   │   │   ├── neg_sample1.wav
│   │   │   └── neg_sample2.wav
│   │   └── word_2/
│   │       ├── ...
│   └── test/
│       ├── word_1/
│       │   ├── pos_sample1.wav
│       │   ├── ...
│       └── word_2/
│           ├── ...
├── templates/
│   ├── speaker_1/
│   │   ├── word_1/
│   │   │   ├── template1.wav
│   │   │   ├── template2.wav
│   │   └── word_2/
│   │       ├── ...
│   ├── speaker_2/
│   │   ├── word_1/
│   │   └── word_2/
│   └── ...
```

> Note:
> - For both the dev and test sets, each word must have an equal number of positive and negative examples. This ensures that evaluation metrics like precision and recall remain fair and balanced.
> - Ensure that speakers in the dev and test sets are unique (i.e., they do not appear in both sets).
> - For each template word, make sure to include the same number of samples. This keeps comparisons across words consistent.

For an example of correct structure, look at the structure of [ISACS](https://reubix29.github.io/isolated-afrikaans-child-speech/).

### Use a custom representation or distance method

A working representation function should take in a `torchaudio` [Tensor](https://docs.pytorch.org/audio/stable/generated/torchaudio.load.html#torchaudio.load) as a parameter, and return a continuous or discrete representation of that audio.

Next, you must add a `@register_rep_fn("rep_fn_alias")` decorator above the custom representation function to register it. You may then select it in `map.yml` by setting `representation_function : "rep_fn_alias"`.

By default, continuous representations can be evaluated with dynamic time warping, and discrete representations can be evaluated with normalised edit distance, but you may need to create a custom distance method depending on your representation function.

To register a distance function, you may likewise add a `@register_dist_fn("dist_fn_alias")` above your distance function to register it. Then select it in `map.yml` by setting `distance_function : "dist_fn_alias"`.


## Results

| Representation| Recall | Precision | F1 | ROC AUC | Balanced Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **mHuBERT continuous**| 58.0 | 65.2 | 59.8 | 66.0 | **65.6** | 
| **MFCCs** | **71.6** | 63.3 | **64.4** | 70.5 | 65.4 |
| **Whisper English ASR** | 47.7 | **73.6** | 53.7 | **70.7** | 64.8| 
| **HuBERT discrete** | 66.2 | 60.1 | 61.3 | 65.3 | 60.6 |
---

