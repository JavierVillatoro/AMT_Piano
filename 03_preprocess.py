import os
import random
import numpy as np
import librosa
import librosa.display
import pretty_midi
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN DE AUDIO ---
SR = 16000           
HOP_LENGTH = 512     
# CQT CONFIG (Optimizado para piano 88 teclas)
MIN_NOTE = 'A0'      
N_BINS = 88          
BINS_PER_OCTAVE = 12 
MIN_MIDI = 21        
MAX_MIDI = 108       
NUM_CLASSES = 88     

def ask_percentage():
    while True:
        try:
            val = input("\n📊 ¿Qué porcentaje del dataset quieres procesar? (0.1 - 100): ")
            percent = float(val)
            if 0 < percent <= 100:
                return percent
        except ValueError: pass

def compute_cqt(audio_path):
    # Cargar audio
    y, _ = librosa.load(str(audio_path), sr=SR)
    
    # Calcular CQT
    cqt = librosa.cqt(
        y=y, 
        sr=SR, 
        hop_length=HOP_LENGTH, 
        fmin=librosa.note_to_hz(MIN_NOTE), 
        n_bins=N_BINS, 
        bins_per_octave=BINS_PER_OCTAVE
    )
    
    # Convertir a dB y Normalizar
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).T.astype(np.float32)
    
    # Normalización simple (0 a 1)
    cqt_db = (cqt_db + 80.0) / 80.0
    cqt_db = np.clip(cqt_db, 0, 1)
    
    return y, cqt_db

def add_soft_label(matrix, exact_frame, pitch_idx, num_frames, width=3):
    """
    Aplica etiquetas suaves (soft labels) decrecientes alrededor del frame exacto.
    Basado en la técnica de High-Resolution (Kong et al / hFT-Transformer).
    """
    start_idx = int(np.floor(exact_frame - width))
    end_idx = int(np.ceil(exact_frame + width))

    for i in range(start_idx, end_idx + 1):
        if 0 <= i < num_frames:
            dist = abs(i - exact_frame)
            val = 1.0 - (dist / width)
            if val > 0:
                # Usamos max para no sobrescribir si ya había una nota más cercana
                matrix[i, pitch_idx] = max(matrix[i, pitch_idx], val)

def compute_labels(midi_path, num_frames):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error MIDI {midi_path}: {e}")
        return None, None, None, None

    # Inicializar matrices 
    # NOTA: Onset y Offset ahora son float32 para soportar decimales (0.9, 0.5, etc)
    onset_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    offset_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    frame_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.int8)
    velocity_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)

    time_per_frame = HOP_LENGTH / SR
    J_WIDTH = 3 # Ancho de la ventana de suavizado (paper standard)

    for note in pm.instruments[0].notes:
        if note.pitch < MIN_MIDI or note.pitch > MAX_MIDI:
            continue
            
        pitch_idx = note.pitch - MIN_MIDI
        
        # Tiempos exactos en frames (con decimales)
        start_frame_float = note.start / time_per_frame
        end_frame_float = note.end / time_per_frame
        
        # Tiempos enteros para la matriz de FRAME (sostenimiento)
        start_frame_int = int(round(start_frame_float))
        end_frame_int = int(round(end_frame_float))
        
        # Limites seguros
        start_frame_int = max(0, min(start_frame_int, num_frames - 1))
        end_frame_int = max(0, min(end_frame_int, num_frames - 1))

        if end_frame_int <= start_frame_int:
            end_frame_int = start_frame_int + 1
        
        # 1. FRAME (Binario: Sostenimiento)
        frame_matrix[start_frame_int:end_frame_int, pitch_idx] = 1
        
        # 2. ONSET (High-Res Soft Label)
        add_soft_label(onset_matrix, start_frame_float, pitch_idx, num_frames, width=J_WIDTH)
        
        # 3. OFFSET (High-Res Soft Label)
        add_soft_label(offset_matrix, end_frame_float, pitch_idx, num_frames, width=J_WIDTH)
        
        # 4. VELOCITY (En el frame de inicio entero)
        if start_frame_int < num_frames:
            velocity_matrix[start_frame_int, pitch_idx] = note.velocity / 127.0

    return onset_matrix, offset_matrix, frame_matrix, velocity_matrix

def save_verification_plot(audio, cqt, onsets, frames, velocities, file_id, save_path):
    # Definir duración exacta en segundos
    duration_sec = len(audio) / SR
    extent = [0, duration_sec, 0, 88]

    # Cortamos a 10 segundos
    zoom_sec = 10
    if duration_sec > zoom_sec:
        max_sample = int(zoom_sec * SR)
        max_frame = int(zoom_sec * SR / HOP_LENGTH)
        
        audio_cut = audio[:max_sample]
        duration_cut = len(audio_cut) / SR
        extent_cut = [0, duration_cut, 0, 88]
        
        cqt_cut = cqt[:max_frame].T
        frames_cut = frames[:max_frame].T
        onsets_cut = onsets[:max_frame].T
        vel_cut = velocities[:max_frame].T
    else:
        audio_cut = audio
        extent_cut = extent
        cqt_cut = cqt.T
        frames_cut = frames.T
        onsets_cut = onsets.T
        vel_cut = velocities.T

    fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    plt.subplots_adjust(hspace=0.2)
    
    ax[0].set_title(f"1. Audio Waveform - {file_id}")
    librosa.display.waveshow(audio_cut, sr=SR, ax=ax[0], color='#444444')
    ax[0].set_ylabel("Amplitude")
    
    ax[1].set_title("2. Input CQT")
    ax[1].imshow(cqt_cut, aspect='auto', origin='lower', extent=extent_cut, cmap='magma', interpolation='nearest')
    
    ax[2].set_title("3. Target: Frames (Binary)")
    ax[2].imshow(frames_cut, aspect='auto', origin='lower', extent=extent_cut, cmap='binary', interpolation='nearest')
    
    ax[3].set_title("4. Target: Onsets (High-Res Soft Labels)")
    ax[3].imshow(onsets_cut, aspect='auto', origin='lower', extent=extent_cut, cmap='Reds', interpolation='nearest')
    
    ax[4].set_title("5. Target: Velocity")
    im = ax[4].imshow(vel_cut, aspect='auto', origin='lower', extent=extent_cut, cmap='viridis', interpolation='nearest')
    
    ax[4].set_xlabel("Time (seconds)")
    
    output_file = save_path / f"debug_viz_{file_id}.png"
    plt.savefig(output_file)
    plt.close()
    print(f"   📸 Verificación guardada: {output_file.name}")

def procesar_dataset():
    root_path = Path("data/maestro-v3.0.0") 
    output_base = Path("processed_data_more")
    
    folders = ["inputs_cqt", "targets_onset", "targets_offset", "targets_frame", "targets_velocity"]
    for sub in folders:
        (output_base / sub).mkdir(parents=True, exist_ok=True)
    
    viz_path = output_base / "debug_visualizations"
    viz_path.mkdir(parents=True, exist_ok=True)

    all_wavs = list(root_path.rglob("*.wav"))
    
    if not all_wavs:
        print("❌ No se encontraron archivos .wav")
        return

    percent = ask_percentage()
    num_to_process = int(len(all_wavs) * (percent / 100))
    if num_to_process < 1: num_to_process = 1
    
    random.shuffle(all_wavs)
    selected_wavs = all_wavs[:num_to_process]

    print(f"\n🚀 Procesando {num_to_process} archivos (Modo High-Resolution)...")
    print(f"📂 Guardando en: {output_base.resolve()}")

    viz_count = 0 
    MAX_VIZ = 3

    for wav_path in tqdm(selected_wavs):
        midi_path = wav_path.with_suffix(".mid")
        if not midi_path.exists(): midi_path = wav_path.with_suffix(".midi")
        if not midi_path.exists(): continue

        file_id = f"{wav_path.parent.name}_{wav_path.stem}"

        # 1. Audio + CQT
        audio_raw, cqt = compute_cqt(wav_path)
        
        # 2. Labels (Con lógica High-Res)
        onsets, offsets, frames, vels = compute_labels(midi_path, cqt.shape[0])

        if onsets is None: continue

        # 3. Guardar Datos
        np.save(output_base / "inputs_cqt" / f"{file_id}.npy", cqt)
        np.save(output_base / "targets_onset" / f"{file_id}.npy", onsets)
        np.save(output_base / "targets_offset" / f"{file_id}.npy", offsets)
        np.save(output_base / "targets_frame" / f"{file_id}.npy", frames)
        np.save(output_base / "targets_velocity" / f"{file_id}.npy", vels)

        # 4. Visualización
        if viz_count < MAX_VIZ:
            save_verification_plot(audio_raw, cqt, onsets, frames, vels, file_id, viz_path)
            viz_count += 1

    print("\n✅ ¡Proceso High-Res completado correctamente!")

if __name__ == "__main__":
    procesar_dataset()