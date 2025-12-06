# AMT Piano Sheet Music - Transcription Project (Deep Learning)

This project implements an **Automatic Music Transcription (AMT)** system for piano using a hybrid **CRNN (Convolutional Recurrent Neural Network)** architecture.

The model converts audio (`.wav`) into MIDI (`.mid`), predicting not only pitch but also precise **note durations** and **velocity**.

---

## Execution Pipeline (Correct Order)

### 1. Data Preparation

* **Script:** `01_rename_midi.py`
    * **Function:** Normalizes `.midi` extensions to `.mid`.

* **Script:** `02_resample_in_place.py`
    * **Function:** Converts the entire MAESTRO dataset to **16,000 Hz (Mono)**. Overwrites originals to save disk space.

---

### 2. Preprocessing (CQT / HCQT)

* **Script:** `03_preprocess_multichannel_fixed.py` (**Recommended**)
    * **Function:** Converts audio into spectrograms.
    * **Modes:**
        * **Option 1 (CQT):** 1 channel. Standard.
        * **Option 2 (HCQT):** 3 channels (Fundamental, 2nd Harmonic, 3rd Harmonic). *Higher quality.*

    * **Fix:** Includes zero-padding for frequencies above the Nyquist limit (8 kHz).

---

### 3. Training (Full-Stack)

* **Script:** `06_training_full.py`
    * **Architecture:** U-Net (Encoder/Decoder) + BiLSTM (Temporal Context).
    * **Output Heads (4 heads):**
        1. `Onset`: Does a note start?
        2. `Frame`: Is the note sustained?
        3. `Offset`: Does the note end? (For accurate duration)
        4. `Velocity`: Note dynamics (0–127)

    * **Optimizations:**
        * **Smart Sampling:** Ignores silent fragments.
        * **Masked Loss:** Velocity is trained only when a note is present.
        * **Pos_weight:** Class balancing (10× weight for Onsets).

---

### 4. Inference (Audio → MIDI)

* **Script:** `07_inference_full.py`
    * **Function:** Generates the final MIDI file.
    * **Features:**
        * Auto-detects format (MP3/WAV) and converts to 16 kHz.
        * Applies **padding** to avoid U-Net size mismatches.
        * Uses `Onset` + `Frame` + `Offset` for millisecond-accurate note durations.
        * Uses the **predicted velocity** (no randomness).

    * **Output:** Results are saved in `mis_midis_generados`.

---

## Model Architecture (`PianoCRNN`)

The neural network processes ~10-second windows (320 frames).

| Component | Configuration |
| :--- | :--- |
| **Input** | `(Batch, 1, 320, 88)` or `(Batch, 3, 320, 88)` |
| **Encoder** | 3 Convolutional Blocks with MaxPool (time-only `2×1`) |
| **Bottleneck** | Bidirectional LSTM with 128 hidden units |
| **Decoder** | 3 Blocks with Skip Connections (U-Net style) |
| **Output** | 4 probability matrices `(Time, 88)` |

---

## Common Error Fixes

1. **Nyquist Error in Preprocessing**
    * *Cause:* Trying to compute 3rd harmonic for high notes (> 8 kHz).
    * *Solution:* The `fixed` script fills those frequencies with zeros.

2. **RuntimeError: Sizes of tensors must match... (Inference)**
    * *Cause:* Audio length mismatches due to MaxPool downsampling.
    * *Solution:* `07_inference_full.py` automatically pads the audio to be a multiple of 4.

3. **OSError with MP3 (Windows)**
    * *Cause:* Dragging files into PowerShell adds `&` and quotes.
    * *Solution:* The inference script automatically cleans the file path.

---

## Required Libraries

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy librosa pretty_midi soundfile tqdm scikit-learn matplotlib seaborn

# Optional for MP3 on Windows:
# winget install ffmpeg

