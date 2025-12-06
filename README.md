# AMT Piano Sheet Music - Proyecto de Transcripción (Deep Learning)

Este proyecto implementa un sistema de **Automatic Music Transcription (AMT)** para piano utilizando una arquitectura híbrida **CRNN (Convolutional Recurrent Neural Network)**. 

El modelo convierte audio (`.wav`) a MIDI (`.mid`) prediciendo no solo las notas, sino también su duración exacta y dinámica (velocity).

---

## Pipeline de Ejecución (Orden Correcto)

### 1. Preparación de Datos
* **Script:** `01_rename_midi.py`
    * **Función:** Normaliza las extensiones de `.midi` a `.mid`.
* **Script:** `02_resample_in_place.py`
    * **Función:** Convierte todo el dataset MAESTRO a **16.000 Hz (Mono)**. Sobrescribe los originales para ahorrar espacio.

### 2. Preprocesamiento (CQT / HCQT)
* **Script:** `03_preprocess_multichannel_fixed.py` (Recomendado)
    * **Función:** Convierte audio en espectrogramas.
    * **Modos:**
        * `Opción 1 (CQT)`: 1 Canal. Estándar.
        * `Opción 2 (HCQT)`: 3 Canales (Fundamental, 2º Armónico, 3º Armónico). *Mejor calidad.*
    * **Fix:** Incluye padding con ceros para frecuencias que superan el límite de Nyquist (8kHz).

### 3. Entrenamiento (Full-Stack)
* **Script:** `06_training_full.py`
    * **Arquitectura:** U-Net (Encoder/Decoder) + BiLSTM (Contexto Temporal).
    * **Cabezas de Salida (4 Heads):**
        1.  `Onset`: ¿Empieza una nota?
        2.  `Frame`: ¿Se mantiene la nota?
        3.  `Offset`: ¿Acaba la nota? (Para precisión en la duración).
        4.  `Velocity`: Dinámica/Volumen (0-127).
    * **Optimizaciones:**
        * `Smart Sampling`: El Dataset ignora fragmentos de silencio.
        * `Masked Loss`: Solo entrenamos la velocity cuando realmente hay una nota.
        * `Pos_weight`: Balanceo de clases (10x peso a los Onsets).

### 4. Inferencia (Audio -> MIDI)
* **Script:** `07_inference_full.py`
    * **Función:** Genera el archivo MIDI final.
    * **Características:**
        * Autodetecta formato (MP3/WAV) y convierte a 16kHz.
        * Aplica **Padding** para evitar errores de tamaño en la U-Net.
        * Usa `Onset` + `Frame` + `Offset` para cortar las notas con precisión milimétrica.
        * Asigna la **Velocity real** predicha por el modelo (no random).
    * **Salida:** Guarda los resultados en la carpeta `mis_midis_generados`.

---

## Arquitectura del Modelo (`PianoCRNN`)

La red neuronal procesa ventanas de audio de ~10 segundos (320 frames).

| Componente | Configuración |
| :--- | :--- |
| **Input** | `(Batch, 1, 320, 88)` o `(Batch, 3, 320, 88)` |
| **Encoder** | 3 Bloques Convolucionales con MaxPool (solo en tiempo `2x1`). |
| **Bottleneck** | BiLSTM (Bidireccional) con 128 unidades ocultas. |
| **Decoder** | 3 Bloques con Skip Connections (estilo U-Net). |
| **Salida** | 4 Matrices de Probabilidad `(Time, 88)`. |

---

## Solución de Errores Comunes

1.  **Error de Nyquist en Preprocess:**
    * *Causa:* Intentar calcular el 3er armónico de notas agudas (>8kHz).
    * *Solución:* El script `fixed` rellena esas frecuencias con ceros.

2.  **RuntimeError: Sizes of tensors must match... (Inferencia):**
    * *Causa:* El audio tiene una longitud impar que no cuadra con el MaxPool.
    * *Solución:* `07_inference_full.py` aplica padding automático para que sea múltiplo de 4.

3.  **OSError con MP3 (Windows):**
    * *Causa:* Arrastrar archivos a PowerShell añade `&` y comillas.
    * *Solución:* El script de inferencia limpia automáticamente la ruta del archivo.

---

## Librerías Necesarias
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install numpy librosa pretty_midi soundfile tqdm scikit-learn matplotlib seaborn
# Opcional para MP3 en Windows:
# winget install ffmpeg


Inference_full.py