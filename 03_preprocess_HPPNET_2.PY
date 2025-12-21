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
# ⚙️ CONFIGURACIÓN FINAL OPTIMIZADA (V2)
# ==========================================
SR = 16000           # 16 kHz
HOP_LENGTH = 512     

# Configuración del Piano
MIN_NOTE = 'A0'      
MIN_MIDI = 21        
MAX_MIDI = 108       
NUM_CLASSES = 88     

# 🔥 CAMBIO 1: ALTA RESOLUCIÓN
BINS_PER_OCTAVE = 48         # Antes 12. Ahora 4 bins por semitono.
NUM_OCTAVES = 88 / 12        # ~7.33 octavas
N_BINS = int(np.ceil(NUM_OCTAVES * BINS_PER_OCTAVE)) # ~352 bins totales

# 🔥 CAMBIO 2: ARMÓNICOS DE OCTAVA (MEJORES PARA CNN)
# [1, 2, 4] crea un patrón geométrico regular sin ruido de subarmónicos (0.5)
# ni confusión de quintas (3).
HARMONICS = [1, 2, 4] 

# 🔹 HIGH-RESOLUTION LABELS
# Reducimos un poco el ancho del label suave para ser más precisos con la nueva resolución
J_WIDTH = 2 

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
    Genera un tensor 3D (Time, 352, 3) ROBUSTO contra el límite de Nyquist.
    """
    y, _ = librosa.load(str(audio_path), sr=SR)
    
    base_fmin_hz = librosa.note_to_hz(MIN_NOTE)
    nyquist = SR / 2  # 8000 Hz
    
    cqt_layers = []

    for h in HARMONICS:
        current_fmin = base_fmin_hz * h
        
        # --- LÓGICA DE PROTECCIÓN (Evita el crash en 16k con 48 bins) ---
        if current_fmin >= nyquist:
            n_bins_possible = 0
        else:
            # Calculamos frecuencia máxima teórica para este armónico
            max_freq_target = current_fmin * (2 ** ((N_BINS - 1) / BINS_PER_OCTAVE))
            
            if max_freq_target < nyquist:
                n_bins_possible = N_BINS # Todo cabe bien
            else:
                # Si se pasa, calculamos cuántos bins caben antes de chocar con 8000Hz
                limit_bins = int(np.floor(BINS_PER_OCTAVE * np.log2(nyquist / current_fmin)))
                n_bins_possible = max(0, limit_bins)
        # ----------------------------------------------------

        if n_bins_possible > 0:
            cqt = librosa.cqt(
                y=y, 
                sr=SR, 
                hop_length=HOP_LENGTH, 
                fmin=current_fmin, 
                n_bins=n_bins_possible, 
                bins_per_octave=BINS_PER_OCTAVE
            )
            
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).T.astype(np.float32)
            cqt_db = (cqt_db + 80.0) / 80.0
            cqt_db = np.clip(cqt_db, 0, 1)

            # Rellenamos lo que falta con ceros (padding) para mantener la forma (Time, 352)
            if n_bins_possible < N_BINS:
                missing = N_BINS - n_bins_possible
                # Padding en el eje de frecuencia (axis 1)
                cqt_db = np.pad(cqt_db, ((0, 0), (0, missing)), 'constant')
                
        else:
            # Capa totalmente negra si no cabe nada (ej. armónico 4 de notas muy agudas)
            n_frames = librosa.time_to_frames(librosa.get_duration(y=y, sr=SR), sr=SR, hop_length=HOP_LENGTH)
            cqt_db = np.zeros((n_frames, N_BINS), dtype=np.float32)
        
        cqt_layers.append(cqt_db)
    
    # Ajuste fino de longitud temporal
    min_len = min(layer.shape[0] for layer in cqt_layers)
    cqt_layers = [layer[:min_len, :] for layer in cqt_layers]
    
    # Stack final: (Time, 352, 3)
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

    # Las etiquetas siguen siendo de 88 clases (notas de piano), NO 352 bins
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
        
        # 4. VELOCITY (Normalizado 0-1)
        # Nota: velocity=1.0 es MIDI 127, muy raro. 0.8 es normal.
        if start_frame_int < num_frames:
            # Ponemos velocidad en todo el frame de duración para facilitar learning
            velocity_matrix[start_frame_int:end_frame_int, pitch_idx] = note.velocity / 127.0

    return onset_matrix, offset_matrix, frame_matrix, velocity_matrix

def save_verification_plot(audio, hcqt, onsets, frames, velocities, file_id, save_path):
    # Definir límites de tiempo para visualización (10 segundos o duración real)
    duration_sec = len(audio) / SR
    zoom_sec = 10
    
    if duration_sec > zoom_sec:
        # Recortar datos a 10 segundos
        max_frame = int(zoom_sec * SR / HOP_LENGTH)
        max_sample = int(zoom_sec * SR)
        
        audio_cut = audio[:max_sample]
        hcqt_cut = hcqt[:max_frame] 
        frames_cut = frames[:max_frame]
        onsets_cut = onsets[:max_frame]
        vel_cut = velocities[:max_frame]
        
        # Definir los "Extents" (Límites de los ejes en la gráfica)
        # [X_min, X_max, Y_min, Y_max]
        # X va de 0 a 10 segundos
        extent_labels = [0, zoom_sec, 0, 88]      # Para targets (88 teclas)
        extent_hcqt = [0, zoom_sec, 0, N_BINS]    # Para input (352 bins)
    else:
        # Usar audio completo
        audio_cut = audio
        hcqt_cut = hcqt
        frames_cut = frames
        onsets_cut = onsets
        vel_cut = velocities
        
        extent_labels = [0, duration_sec, 0, 88]
        extent_hcqt = [0, duration_sec, 0, N_BINS]

    # Visualizamos el Canal 0 (Fundamental = Armónico 1). 
    hcqt_viz = hcqt_cut[:, :, 0].T 

    # Crear figura
    fig, ax = plt.subplots(5, 1, figsize=(12, 16), sharex=True) # sharex=True fuerza a alinear
    plt.subplots_adjust(hspace=0.2) # Menos espacio entre gráficas
    
    # 1. AUDIO
    ax[0].set_title(f"1. Audio Waveform - {file_id}", fontsize=10)
    librosa.display.waveshow(audio_cut, sr=SR, ax=ax[0], color='#333333')
    ax[0].set_ylabel("Amplitude")
    
    # 2. HCQT (Input)
    # 🔥 AQUI ESTA EL FIX: Usamos extent=extent_hcqt para que use SEGUNDOS en X
    ax[1].set_title(f"2. HCQT (Canal 1) - Shape: {hcqt.shape} (Bins={N_BINS})", fontsize=10)
    ax[1].imshow(hcqt_viz, aspect='auto', origin='lower', cmap='magma', extent=extent_hcqt)
    ax[1].set_ylabel("Frequency Bins")
    
    # 3. TARGET FRAMES
    ax[2].set_title("3. Target: Frames (88 keys)", fontsize=10)
    ax[2].imshow(frames_cut.T, aspect='auto', origin='lower', cmap='binary', extent=extent_labels)
    ax[2].set_ylabel("Piano Keys")
    
    # 4. TARGET ONSETS
    ax[3].set_title("4. Target: Onsets (High-Res)", fontsize=10)
    ax[3].imshow(onsets_cut.T, aspect='auto', origin='lower', cmap='Reds', extent=extent_labels)
    ax[3].set_ylabel("Piano Keys")
    
    # 5. TARGET VELOCITY
    ax[4].set_title("5. Target: Velocity", fontsize=10)
    im = ax[4].imshow(vel_cut.T, aspect='auto', origin='lower', cmap='inferno', extent=extent_labels)
    ax[4].set_ylabel("Piano Keys")
    ax[4].set_xlabel("Time (seconds)")
    
    # Colorbar para velocity
    cbar = fig.colorbar(im, ax=ax[4], orientation='horizontal', fraction=0.05, pad=0.3)
    cbar.set_label("Velocity Intensity")
    
    out_file = save_path / f"verify_{file_id}.png"
    plt.savefig(out_file, dpi=100)
    plt.close()
    print(f"   📸 Debug guardado (Corregido): {out_file.name}")

def procesar_dataset():
    root_path = Path("data/maestro-v3.0.0") 
    output_base = Path("processed_data_HPPNET_2") 
    
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

    print(f"\n🚀 PROCESAMIENTO HIGH-RES (SR={SR}, Bins={BINS_PER_OCTAVE}, Harmonics={HARMONICS})")
    print(f"🔹 Archivos: {num_to_process}")
    print(f"🔹 Output: {output_base.resolve()}")

    viz_count = 0 
    MAX_VIZ = 3

    for wav_path in tqdm(selected_wavs):
        midi_path = wav_path.with_suffix(".mid")
        if not midi_path.exists(): midi_path = wav_path.with_suffix(".midi")
        if not midi_path.exists(): continue

        file_id = f"{wav_path.parent.name}_{wav_path.stem}"

        # 1. Calcular HCQT (3 Canales: 1, 2, 4 | 352 Bins)
        audio_raw, hcqt = compute_hcqt(wav_path)
        
        # 2. Calcular Etiquetas (88 Clases)
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
    print(f"⚠️  NOTA: Tus inputs ahora tienen shape (Time, {N_BINS}, 3) y targets (Time, 88).")
    print("    El modelo HPPNet-Separate se encargará de hacer el pooling de 352 -> 88.")

if __name__ == "__main__":
    procesar_dataset()