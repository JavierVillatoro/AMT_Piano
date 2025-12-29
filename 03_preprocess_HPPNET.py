import os
import random
import numpy as np
import librosa
import librosa.display
import pretty_midi
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ==========================================
# ⚙️ CONFIGURACIÓN FINAL OPTIMIZADA
# ==========================================
SR = 16000           # 16 kHz
HOP_LENGTH = 512     

# Configuración del Piano
MIN_NOTE = 'A0'      
N_BINS = 88          
BINS_PER_OCTAVE = 12 
MIN_MIDI = 21        
MAX_MIDI = 108       
NUM_CLASSES = 88     

# 🔹 HCQT CONFIG (ULTRALIGHT)
# Solo 3 canales. Esto volará en tu ordenador.
HARMONICS = [0.5, 1, 2] 

# 🔹 HIGH-RESOLUTION LABELS
J_WIDTH = 1 

def ask_percentage():
    while True:
        try:
            val = input("\n📊 ¿Qué porcentaje del dataset quieres procesar? (0.1 - 100): ")
            percent = float(val)
            if 0 < percent <= 100:
                return percent
        except ValueError: pass

def compute_hcqt(audio_path):
    """
    Genera un tensor 3D (Time, 88, 3) ROBUSTO contra el límite de Nyquist.
    """
    y, _ = librosa.load(str(audio_path), sr=SR)
    
    base_fmin_hz = librosa.note_to_hz(MIN_NOTE)
    nyquist = SR / 2  # 8000 Hz
    
    cqt_layers = []

    for h in HARMONICS:
        current_fmin = base_fmin_hz * h
        
        # --- LÓGICA DE PROTECCIÓN (Evita el crash en 16k) ---
        if current_fmin >= nyquist:
            # Si la frecuencia de inicio ya se pasa del límite, capa vacía.
            n_bins_possible = 0
        else:
            # Vemos hasta dónde podemos llegar antes de chocar con 8000 Hz
            max_freq_target = current_fmin * (2 ** ((N_BINS - 1) / BINS_PER_OCTAVE))
            
            if max_freq_target < nyquist:
                n_bins_possible = N_BINS # Todo cabe bien
            else:
                # Calculamos cuántos bins caben matemáticamente
                limit_bins = int(np.floor(BINS_PER_OCTAVE * np.log2(nyquist / current_fmin)))
                n_bins_possible = max(0, limit_bins)
        # ----------------------------------------------------

        if n_bins_possible > 0:
            cqt = librosa.cqt(
                y=y, 
                sr=SR, 
                hop_length=HOP_LENGTH, 
                fmin=current_fmin, 
                n_bins=n_bins_possible, # Pedimos solo lo que cabe
                bins_per_octave=BINS_PER_OCTAVE
            )
            
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).T.astype(np.float32)
            cqt_db = (cqt_db + 80.0) / 80.0
            cqt_db = np.clip(cqt_db, 0, 1)

            # Rellenamos lo que falta con ceros (padding)
            if n_bins_possible < N_BINS:
                missing = N_BINS - n_bins_possible
                # Rellenamos a la derecha en el eje de frecuencias
                cqt_db = np.pad(cqt_db, ((0, 0), (0, missing)), 'constant')
                
        else:
            # Capa totalmente negra si no cabe nada
            n_frames = librosa.time_to_frames(librosa.get_duration(y=y, sr=SR), sr=SR, hop_length=HOP_LENGTH)
            cqt_db = np.zeros((n_frames, N_BINS), dtype=np.float32)
        
        cqt_layers.append(cqt_db)
    
    # Ajuste fino de longitud temporal (a veces varía 1 frame)
    min_len = min(layer.shape[0] for layer in cqt_layers)
    cqt_layers = [layer[:min_len, :] for layer in cqt_layers]
    
    # Stack final: (Time, 88, 3)
    hcqt = np.stack(cqt_layers, axis=-1)
    
    return y, hcqt

def add_soft_label(matrix, exact_frame, pitch_idx, num_frames, width=1):
    start_idx = int(np.floor(exact_frame - width))
    end_idx = int(np.ceil(exact_frame + width))

    for i in range(start_idx, end_idx + 1):
        if 0 <= i < num_frames:
            dist = abs(i - exact_frame)
            val = 1.0 - (dist / width)
            if val > 0:
                matrix[i, pitch_idx] = max(matrix[i, pitch_idx], val)

def compute_labels(midi_path, num_frames):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error MIDI {midi_path}: {e}")
        return None, None, None, None

    onset_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    offset_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    frame_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.int8) 
    velocity_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)

    time_per_frame = HOP_LENGTH / SR

    for note in pm.instruments[0].notes:
        if note.pitch < MIN_MIDI or note.pitch > MAX_MIDI:
            continue
            
        pitch_idx = note.pitch - MIN_MIDI
        
        start_frame_float = note.start / time_per_frame
        end_frame_float = note.end / time_per_frame
        
        start_frame_int = int(round(start_frame_float))
        end_frame_int = int(round(end_frame_float))
        
        start_frame_int = max(0, min(start_frame_int, num_frames - 1))
        end_frame_int = max(0, min(end_frame_int, num_frames - 1))

        if end_frame_int <= start_frame_int:
            end_frame_int = start_frame_int + 1
        
        # 1. FRAME
        frame_matrix[start_frame_int:end_frame_int, pitch_idx] = 1
        
        # 2. ONSET
        add_soft_label(onset_matrix, start_frame_float, pitch_idx, num_frames, width=J_WIDTH)
        
        # 3. OFFSET
        add_soft_label(offset_matrix, end_frame_float, pitch_idx, num_frames, width=J_WIDTH)
        
        # 4. VELOCITY (Full Duration)
        norm_vel = note.velocity / 127.0
        velocity_matrix[start_frame_int:end_frame_int, pitch_idx] = norm_vel

    return onset_matrix, offset_matrix, frame_matrix, velocity_matrix

def save_verification_plot(audio, hcqt, onsets, frames, velocities, file_id, save_path):
    duration_sec = len(audio) / SR
    zoom_sec = 10
    
    if duration_sec > zoom_sec:
        max_frame = int(zoom_sec * SR / HOP_LENGTH)
        max_sample = int(zoom_sec * SR)
        audio_cut = audio[:max_sample]
        hcqt_cut = hcqt[:max_frame] 
        frames_cut = frames[:max_frame]
        onsets_cut = onsets[:max_frame]
        vel_cut = velocities[:max_frame]
        extent_cut = [0, zoom_sec, 0, 88]
    else:
        audio_cut = audio
        hcqt_cut = hcqt
        frames_cut = frames
        onsets_cut = onsets
        vel_cut = velocities
        extent_cut = [0, duration_sec, 0, 88]

    # Visualizamos el Canal 1 (Fundamental). En la lista [0.5, 1, 2], el índice 1 es la fundamental.
    hcqt_viz = hcqt_cut[:, :, 1].T 

    fig, ax = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    ax[0].set_title(f"1. Audio - {file_id}")
    librosa.display.waveshow(audio_cut, sr=SR, ax=ax[0], color='#333333')
    
    ax[1].set_title(f"2. HCQT (Canal Fundamental) - Input Shape: {hcqt.shape}")
    ax[1].imshow(hcqt_viz, aspect='auto', origin='lower', extent=extent_cut, cmap='magma')
    
    ax[2].set_title("3. Target: Frames")
    ax[2].imshow(frames_cut.T, aspect='auto', origin='lower', extent=extent_cut, cmap='binary')
    
    ax[3].set_title("4. Target: Onsets")
    ax[3].imshow(onsets_cut.T, aspect='auto', origin='lower', extent=extent_cut, cmap='Reds')
    
    ax[4].set_title("5. Target: Velocity")
    im = ax[4].imshow(vel_cut.T, aspect='auto', origin='lower', extent=extent_cut, cmap='inferno')
    
    cbar = fig.colorbar(im, ax=ax[4], orientation='horizontal', fraction=0.05, pad=0.25)
    cbar.set_label("Velocity")
    
    ax[4].set_xlabel("Time (s)")
    
    out_file = save_path / f"verify_{file_id}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"   📸 Debug guardado: {out_file.name}")

def procesar_dataset():
    root_path = Path("data/maestro-v3.0.0") 
    output_base = Path("processed_data_HPPNET_10") 
    
    folders = ["inputs_hcqt", "targets_onset", "targets_offset", "targets_frame", "targets_velocity"]
    for sub in folders:
        (output_base / sub).mkdir(parents=True, exist_ok=True)
    
    viz_path = output_base / "debug_visualizations"
    viz_path.mkdir(parents=True, exist_ok=True)

    all_wavs = list(root_path.rglob("*.wav"))
    if not all_wavs:
        print("❌ ERROR: No se encontraron archivos .wav en data/maestro-v3.0.0")
        return

    percent = ask_percentage()
    num_to_process = int(len(all_wavs) * (percent / 100))
    if num_to_process < 1: num_to_process = 1
    
    random.shuffle(all_wavs)
    selected_wavs = all_wavs[:num_to_process]

    print(f"\n🚀 PROCESAMIENTO LIGHT (SR={SR}, 3 Harmonics) INICIADO")
    print(f"🔹 Archivos: {num_to_process}")
    print(f"🔹 Output: {output_base.resolve()}")

    viz_count = 0 
    MAX_VIZ = 3

    for wav_path in tqdm(selected_wavs):
        midi_path = wav_path.with_suffix(".mid")
        if not midi_path.exists(): midi_path = wav_path.with_suffix(".midi")
        if not midi_path.exists(): continue

        file_id = f"{wav_path.parent.name}_{wav_path.stem}"

        # 1. Calcular HCQT (3 Canales: 0.5, 1, 2)
        audio_raw, hcqt = compute_hcqt(wav_path)
        
        # 2. Calcular Etiquetas
        onsets, offsets, frames, vels = compute_labels(midi_path, hcqt.shape[0])

        if onsets is None: continue

        # 3. Guardar
        np.save(output_base / "inputs_hcqt" / f"{file_id}.npy", hcqt)
        np.save(output_base / "targets_onset" / f"{file_id}.npy", onsets)
        np.save(output_base / "targets_offset" / f"{file_id}.npy", offsets)
        np.save(output_base / "targets_frame" / f"{file_id}.npy", frames)
        np.save(output_base / "targets_velocity" / f"{file_id}.npy", vels)

        if viz_count < MAX_VIZ:
            save_verification_plot(audio_raw, hcqt, onsets, frames, vels, file_id, viz_path)
            viz_count += 1

    print("\n🏆 ¡Proceso finalizado!")
    print("⚠️  IMPORTANTE: Ahora tu modelo debe tener in_channels=3.")

if __name__ == "__main__":
    procesar_dataset()