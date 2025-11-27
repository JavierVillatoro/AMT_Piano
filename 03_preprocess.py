import os
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN DE AUDIO (Hiperparámetros) ---
# Estos valores deben ser LOS MISMOS cuando entrenes la IA
SR = 16000           # Sample Rate (16kHz es suficiente para piano)
HOP_LENGTH = 512     # Cuánto avanzamos en cada "frame". 512 muestras ≈ 32ms
N_MELS = 229         # Cantidad de frecuencias (altura de la imagen de input)
N_FFT = 2048         # Ventana de análisis de Fourier
FMIN = 30            # Frecuencia mínima (aprox A0)
MIN_MIDI = 21        # La nota más grave del piano (A0) es el MIDI 21
MAX_MIDI = 108       # La nota más aguda (C8) es el MIDI 108

def compute_log_mel(audio_path):
    """Genera el Espectrograma Log-Mel (Input de la red)."""
    y, _ = librosa.load(audio_path, sr=SR)
    
    # Generar Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=FMIN
    )
    
    # Convertir a decibelios (Log scale) y transponer a [Tiempo, Frecuencia]
    log_mel = librosa.power_to_db(mel, ref=np.max).T
    
    # Normalizar entre 0 y 1 (opcional pero recomendado para redes neuronales)
    # log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
    
    return log_mel.astype(np.float32)

def compute_labels(midi_path, num_frames):
    """
    Genera las matrices de Frames y Onsets (Labels).
    num_frames: Es vital para forzar que el label mida LO MISMO que el audio.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error leyendo MIDI {midi_path}: {e}")
        return None, None

    # Matrices vacías: [Tiempo, 88 teclas]
    # Usamos int8 para ahorrar espacio (solo son 0s y 1s)
    frame_matrix = np.zeros((num_frames, 88), dtype=np.int8)
    onset_matrix = np.zeros((num_frames, 88), dtype=np.int8)

    # Tiempo que dura un frame en segundos
    time_per_frame = HOP_LENGTH / SR

    for note in pm.instruments[0].notes:
        # Ignorar notas fuera del rango del piano (por si acaso)
        if note.pitch < MIN_MIDI or note.pitch > MAX_MIDI:
            continue
            
        # Ajustar índice (0 = tecla 21 MIDI)
        pitch_idx = note.pitch - MIN_MIDI

        # Calcular en qué frame empieza y termina la nota
        start_frame = int(note.start / time_per_frame)
        end_frame = int(note.end / time_per_frame)

        # Seguridad: no salirnos del array
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

        if start_frame >= num_frames:
            continue

        # 1. Rellenar FRAMES (Cuerpo de la nota)
        # Ponemos un 1 desde el inicio hasta el final
        frame_matrix[start_frame:end_frame, pitch_idx] = 1

        # 2. Rellenar ONSETS (Ataque)
        # Ponemos un 1 solo en el frame de inicio
        # (A veces se expande a start_frame +/- 1, pero empecemos simple)
        if start_frame < num_frames:
            onset_matrix[start_frame, pitch_idx] = 1

    return frame_matrix, onset_matrix

def procesar_dataset():
    # Rutas
    root_path = Path("data/maestro-v3.0.0")
    output_base = Path("processed_data")
    
    # Crear estructura de carpetas de salida
    (output_base / "inputs_spectrogram").mkdir(parents=True, exist_ok=True)
    (output_base / "targets/frames").mkdir(parents=True, exist_ok=True)
    (output_base / "targets/onsets").mkdir(parents=True, exist_ok=True)

    # Buscar todos los wavs recursivamente
    wav_files = list(root_path.rglob("*.wav"))
    
    print(f"🚀 Iniciando procesamiento de {len(wav_files)} archivos...")
    print(f"⚙️  Configuración: SR={SR}, Hop={HOP_LENGTH}, Mels={N_MELS}")

    for wav_path in tqdm(wav_files):
        # Encontrar el MIDI correspondiente (mismo nombre, extensión .mid)
        # Asumimos que ya corriste el script de renombrado a .mid
        midi_path = wav_path.with_suffix(".mid")
        
        if not midi_path.exists():
            # Intento de fallback por si no se renombró o es .midi
            midi_path = wav_path.with_suffix(".midi")
            if not midi_path.exists():
                print(f"⚠️ MIDI no encontrado para: {wav_path.name}")
                continue

        # Generar ID único para el archivo (ej: 2004_filename)
        # Usamos el nombre de la carpeta año + nombre archivo para evitar duplicados
        file_id = f"{wav_path.parent.name}_{wav_path.stem}"

        # 1. Procesar AUDIO
        log_mel = compute_log_mel(wav_path)
        
        # 2. Procesar MIDI (Usando el largo del audio para alinear)
        frames, onsets = compute_labels(midi_path, num_frames=log_mel.shape[0])

        if frames is None:
            continue

        # 3. Guardar como .npy
        np.save(output_base / "inputs_spectrogram" / f"{file_id}.npy", log_mel)
        np.save(output_base / "targets/frames" / f"{file_id}.npy", frames)
        np.save(output_base / "targets/onsets" / f"{file_id}.npy", onsets)

    print("\n✅ ¡Procesamiento completado!")
    print(f"📂 Los datos están en: {output_base.resolve()}")

if __name__ == "__main__":
    procesar_dataset()