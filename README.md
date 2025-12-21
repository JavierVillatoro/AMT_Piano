# AMT Piano Sheet Music - Transcription Project (Deep Learning)

This project implements a **custom Automatic Music Transcription (AMT)** system for piano, designed to work with the **MAESTRO dataset**. 

The core neural network is a specialized architecture **based on HPPNet (Harmonic Pitch Prediction Network)** principles. Unlike standard CRNNs, this model utilizes **Harmonic Dilated Convolutions (HDConv)** and **Frequency-Grouped LSTMs** to explicitly capture the harmonic relationships between piano notes, resulting in high-precision polyphonic transcription.

---

## 🚀 Execution Pipeline

Follow these steps in order to reproduce the training and inference pipeline.

### 1. Data Preparation

* **Script:** `01_rename_midi.py`
    * **Purpose:** Normalizes all MIDI file extensions to `.mid`.
* **Script:** `02_resample_in_place.py`
    * **Purpose:** Resamples the entire MAESTRO dataset to **16,000 Hz (Mono)**.
    * *Note:* This overwrites original files to optimize disk space and training speed.

---

### 2. Preprocessing (Ultralight HCQT)

* **Script:** `03_preprocess.py`
    * **Method:** **HCQT (Harmonic Constant-Q Transform)**.
    * **Input Features:** A 3-Channel tensor capturing specific harmonic structures:
        1.  **0.5x** (Sub-harmonic / Octave below)
        2.  **1.0x** (Fundamental frequency)
        3.  **2.0x** (First harmonic / Octave above)
    * **Storage Optimization:**
        * Data is saved using **`float16`** (spectrograms) and **`int8`** (binary targets) to drastically reduce disk usage and allow larger batch sizes.
    * **Safety:** Handles the Nyquist limit automatically to prevent crashes at 16kHz.

---

### 3. Training (Custom HPPNet-based Model)

* **Script:** `06_training.py`
    * **Engine:** PyTorch with **Automatic Mixed Precision (AMP)** for faster training on consumer GPUs (e.g., GTX 1060 Ti).
    * **Hyperparameter Tuning:** Integrated with **Optuna**. The script runs a preliminary search (`N_TRIALS`) to find the optimal Learning Rate and LSTM hidden size before starting the main training.
    * **Output Heads (Multi-Task Learning):**
        1.  **Onset:** Note attack detection.
        2.  **Frame:** Note duration/sustain detection.
        3.  **Offset:** Note release detection.
        4.  **Velocity:** Dynamics regression (MSE Loss masked by active frames).

    * **Class Balancing:** Applies dynamic `pos_weight` to the Loss functions to handle the imbalance between silence and active notes.

---

### 4. Inference (Audio → MIDI)

* **Script:** `07_inferencel.py`
    * **Purpose:** Generates the final MIDI file from raw audio.
    * **Process:**
        1.  Converts audio to the specific **3-Channel HCQT** format used in training.
        2.  Passes data through the custom network.
        3.  Decodes `Onset`, `Frame`, and `Offset` probabilities into millisecond-accurate note events.
        4.  Assigns dynamics using the `Velocity` head.
    * **Output:** MIDI files are saved in the `midi_output` directory.

---

## 🧠 Custom Model Architecture

This architecture is designed to understand audio physics rather than treating spectrograms as simple images.

| Component | Type | Description |
| :--- | :--- | :--- |
| **Input** | `(Batch, 3, Time, 88)` | 3-Channel HCQT Spectrogram (Sub-harmonic, Fundamental, 1st Harmonic). |
| **Stem** | Conv2d Blocks | Initial feature extraction with Instance Normalization. |
| **Harmonic Block** | **HDConv** | **Harmonic Dilated Convolution**. Filters are dilated based on musical intervals (12 bins/octave) to capture dependencies between a note and its harmonics. |
| **Context Block** | **DilatedFreqBlock** | Convolutions dilated along the *frequency axis* to understand the context of the entire piano range. |
| **Output Heads** | **FG-LSTM** | **Frequency-Grouped BiLSTMs**. Instead of one global LSTM, it processes frequency bands to maintain pitch invariance. |

---

## 🛠 Common Issues & Fixes

### 1. Nyquist Error during Preprocessing
* **Symptom:** Error when computing harmonics for high notes (> 8 kHz).
* **Solution:** The `03_preprocess...` script automatically fills frequencies above the Nyquist limit with zeros instead of crashing.

### 2. CUDA Out of Memory
* **Cause:** The 3-channel input increases VRAM usage.
* **Solution:** The training script uses `SEGMENT_FRAMES = 640` (approx 10s) and `gradient_accumulation` to keep the physical batch size small (e.g., 4) while simulating a larger batch.

### 3. OSError with MP3 files (Windows)
* **Symptom:** File paths contain `&` or quotes when dragged into the terminal.
* **Solution:** The scripts include a path cleaner to sanitize Windows inputs automatically.

---

## 📦 Requirements

Install the necessary dependencies. **Optuna** is now required for the training loop.

```bash
# 1. PyTorch (with CUDA 11.8 support recommended)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 2. Audio & Data Processing
pip install numpy librosa pretty_midi soundfile tqdm scikit-learn matplotlib

# 3. Optimization
pip install optuna



