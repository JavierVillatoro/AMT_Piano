# High-Resolution HPPNet for Piano Transcription

This repository contains a PyTorch implementation of a High-Resolution Harmonic Pitch Prediction Network (HPPNet) designed for Automatic Music Transcription (AMT). The system converts raw audio recordings of piano performances into symbolic MIDI files, predicting note onsets, offsets, frame activation, and velocity with high precision.

The model is trained on the **MAESTRO v3.0.0** dataset and utilizes a custom architecture involving Harmonic Dilated Convolutions (HDConv) and Fine-Grained Bi-Directional LSTMs.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
    - [Preprocessing](#1-preprocessing)
    - [Training](#2-training)
    - [Inference](#3-inference)
- [Configuration](#configuration)
- [Metrics and Loss Functions](#metrics-and-loss-functions)
- [License](#license)

## Overview

Automatic Music Transcription (AMT) is the process of converting an acoustic musical signal into some form of musical notation. This project focuses on piano transcription, aiming to output MIDI files that accurately capture the timing and dynamics of a performance.

Key features of this implementation:
* **High-Resolution Input:** Uses a Constant-Q Transform (CQT) with 48 bins per octave, resulting in an input size of 352 frequency bins.
* **Multi-Task Learning:** Simultaneously predicts Onsets, Frames (duration), Offsets, and Velocity.
* **Post-Processing:** Implements a logic-based decoding strategy that utilizes onset peak picking and explicit offset detection to determine note duration.

## Model Architecture

The architecture is based on HPPNet (Harmonic Pitch Prediction Network) with several modifications for stability and performance:

1.  **Acoustic Model:**
    * **Input:** High-Resolution CQT spectrogram (Time, 352, 1).
    * **HDConv (Harmonic Dilated Convolutions):** Captures harmonic structures by using dilated convolutions calculated based on the distance between harmonics in the log-frequency domain.
    * **Residual Blocks:** Uses Instance Normalization and skip connections to facilitate deep training.
    * **Pooling:** Reduces the frequency dimension from 352 bins to 88 bins (one per piano key) before entering the context layers.

2.  **Sequence Modeling:**
    * **FG-LSTM:** Fine-Grained Bidirectional LSTMs process the features for each head (Onset, Frame, Offset, Velocity) to capture temporal dependencies.

3.  **Output Heads:**
    * **Onset:** Detects the start of a note.
    * **Frame:** Detects the active duration of a note.
    * **Offset:** Detects the end of a note.
    * **Velocity:** Regresses the MIDI velocity (dynamics) of the note.

## Dataset

This model is designed to be trained on the **MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization)** dataset, version 3.0.0.

* **Source:** Google Magenta MAESTRO Dataset
* **Format:** The preprocessing script expects the dataset structure to contain `.wav` and `.midi` pairs.

## Requirements

* Python 3.8+
* PyTorch (CUDA recommended)
* Librosa
* PrettyMIDI
* Numpy
* Pandas
* Scikit-learn
* Mir_eval
* Tqdm
* Matplotlib / Seaborn

To install the necessary dependencies, you can use pip:

    pip install torch numpy librosa pretty_midi pandas scikit-learn mir_eval tqdm matplotlib seaborn

## Usage

### 1. Preprocessing

The preprocessing pipeline converts raw audio into CQT spectrograms and generates ground-truth labels for training.

    python preprocess.py

**Output:**
The script generates `.npy` files in the `processed_data_HighRes` directory, organized into:
* `inputs_hcqt`: Input spectrograms.
* `targets_onset`: Note start targets.
* `targets_offset`: Note end targets.
* `targets_frame`: Active note duration targets.
* `targets_velocity`: Normalized velocity values.

### 2. Training

The training script supports multi-GPU DataParallel training and mixed-precision (AMP). It utilizes a curriculum learning approach where loss weights may be adjusted.

    python train.py

**Key Training Features:**
* **Loss Weights:** Dynamically balances Onset, Frame, and Offset importance.
* **Curriculum:** The provided configuration emphasizes Frame and Offset accuracy in later stages.
* **Validation:** Periodically evaluates the model using `mir_eval` metrics.

### 3. Inference

To transcribe an audio file to MIDI using a trained checkpoint:

    python inference.py

The script will prompt for the model checkpoint path and the input audio file path. It outputs a standard `.mid` file.

**Decoding Logic:**
1.  **Peak Picking:** Identifies local maxima in the Onset probability map.
2.  **Sustain Verification:** Checks the Frame probability to confirm the note is active.
3.  **Offset Detection:** Searches for a high Offset probability or a drop in Frame probability to terminate the note.

## Configuration

The default hyperparameters are set for a standard GPU environment (e.g., T4 or P100):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Sample Rate** | 16000 Hz | Audio sampling rate |
| **Hop Length** | 320 | ~20ms temporal resolution |
| **Bins/Octave** | 48 | High spectral resolution |
| **Input Bins** | 352 | Total frequency bins |
| **Batch Size** | 16 | Adjusted for GPU memory |
| **Learning Rate** | 1e-4 | For fine-tuning/Phase 2 |

## Metrics and Loss Functions

The model performance is evaluated using the following metrics:

* **Loss Functions:**
    * **Focal Loss:** Used for Onset and Offset detection to handle class imbalance (sparse events vs. silence).
    * **Combo Loss (BCE + Dice):** Used for Frame detection to ensure both pixel-wise accuracy and global shape consistency.
    * **MSE Loss:** Used for Velocity regression.
* **Evaluation:**
    * **Precision, Recall, F1-Score:** Calculated using `mir_eval.transcription` with standard tolerances (50ms for onsets).
    * **Note Offset Accuracy:** Evaluates if both the start and end of the note are correct (within 20% of duration).

## License

This project is licensed under the MIT License.



