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
HARMONICS = [0.5, 1, 2, 3] 

# 🔹 HIGH-RESOLUTION LABELS
J_WIDTH = 5  

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
    Genera un tensor 3D (Time, 88, 4) con los armónicos definidos.
    """
    y, _ = librosa.load(str(audio_path), sr=SR)
    
    base_fmin_hz = librosa.note_to_hz(MIN_NOTE)
    nyquist = SR / 2 
    
    cqt_layers = []

    for h in HARMONICS:
        current_fmin = base_fmin_hz * h
        
        # --- LÓGICA DE PROTECCIÓN (Evita el crash en 16k) ---
        if current_fmin >= nyquist:
            n_bins_possible = 0
        else:
            max_freq_target = current_fmin * (2 ** ((N_BINS - 1) / BINS_PER_OCTAVE))
            if max_freq_target < nyquist:
                n_bins_possible = N_BINS 
            else:
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

            # Rellenamos padding
            if n_bins_possible < N_BINS:
                missing = N_BINS - n_bins_possible
                cqt_db = np.pad(cqt_db, ((0, 0), (0, missing)), 'constant')
                
        else:
            n_frames = librosa.time_to_frames(librosa.get_duration(y=y, sr=SR), sr=SR, hop_length=HOP_LENGTH)
            cqt_db = np.zeros((n_frames, N_BINS), dtype=np.float32)
        
        cqt_layers.append(cqt_db)
    
    # Ajuste fino de longitud temporal
    min_len = min(layer.shape[0] for layer in cqt_layers)
    cqt_layers = [layer[:min_len, :] for layer in cqt_layers]
    
    # Stack final: (Time, 88, 4)
    hcqt = np.stack(cqt_layers, axis=-1)
    
    return y, hcqt

def add_soft_label(matrix, exact_frame, col_idx, num_frames, width=1):
    """
    Función auxiliar para etiquetas suaves (onset/offset).
    Soporta matrices de (Time, 88) o (Time, 1) para el pedal.
    """
    start_idx = int(np.floor(exact_frame - width))
    end_idx = int(np.ceil(exact_frame + width))

    for i in range(start_idx, end_idx + 1):
        if 0 <= i < num_frames:
            dist = abs(i - exact_frame)
            val = 1.0 - (dist / width)
            if val > 0:
                matrix[i, col_idx] = max(matrix[i, col_idx], val)

def compute_labels(midi_path, num_frames):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error MIDI {midi_path}: {e}")
        return None, None, None, None, None, None, None

    # === A. PREPARACIÓN DE MATRICES DE NOTAS (88 Clases) ===
    onset_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    offset_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    frame_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.int8) 
    velocity_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)

    time_per_frame = HOP_LENGTH / SR

    # --- Procesamiento de NOTAS ---
    for note in pm.instruments[0].notes:
        if note.pitch < MIN_MIDI or note.pitch > MAX_MIDI:
            continue
            
        pitch_idx = note.pitch - MIN_MIDI
        
        start_frame_float = note.start / time_per_frame
        end_frame_float = (note.end / time_per_frame) 
        
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
        # 4. VELOCITY
        norm_vel = note.velocity / 127.0
        velocity_matrix[start_frame_int:end_frame_int, pitch_idx] = norm_vel

    # === B. PREPARACIÓN DE MATRICES DE PEDAL (1 Clase) ===
    # Shape: (Time, 1) - Solo necesitamos saber si "El" pedal de sustain está activo
    pedal_frame = np.zeros((num_frames, 1), dtype=np.int8)
    pedal_onset = np.zeros((num_frames, 1), dtype=np.float32)
    pedal_offset = np.zeros((num_frames, 1), dtype=np.float32)

    # --- Lógica de Extracción de PEDAL (CC 64) ---
    # Convertimos eventos puntuales CC en intervalos de tiempo
    cc_events = sorted([cc for cc in pm.instruments[0].control_changes if cc.number == 64], key=lambda x: x.time)
    
    pedal_intervals = []
    is_pedal_on = False
    current_onset_time = 0.0

    # Máquina de estados para reconstruir intervalos On-Off del pedal
    for cc in cc_events:
        if cc.value >= 64 and not is_pedal_on:
            is_pedal_on = True
            current_onset_time = cc.time
        elif cc.value < 64 and is_pedal_on:
            is_pedal_on = False
            pedal_intervals.append((current_onset_time, cc.time))
    
    # Si el archivo termina con el pedal pisado, lo cerramos al final del archivo
    if is_pedal_on:
        pedal_intervals.append((current_onset_time, pm.get_end_time()))

    # --- Procesamiento de Intervalos de PEDAL ---
    for start_t, end_t in pedal_intervals:
        start_f_float = start_t / time_per_frame
        end_f_float = end_t / time_per_frame # El offset del pedal suele ser exacto al soltar

        start_f_int = int(round(start_f_float))
        end_f_int = int(round(end_f_float))

        start_f_int = max(0, min(start_f_int, num_frames - 1))
        end_f_int = max(0, min(end_f_int, num_frames - 1))

        if end_f_int <= start_f_int: 
            end_f_int = start_f_int + 1

        # 1. PEDAL FRAME
        pedal_frame[start_f_int:end_f_int, 0] = 1
        # 2. PEDAL ONSET
        add_soft_label(pedal_onset, start_f_float, 0, num_frames, width=J_WIDTH)
        # 3. PEDAL OFFSET
        add_soft_label(pedal_offset, end_f_float, 0, num_frames, width=J_WIDTH)

    return onset_matrix, offset_matrix, frame_matrix, velocity_matrix, pedal_frame, pedal_onset, pedal_offset

def save_verification_plot(audio, hcqt, onsets, offsets, frames, velocities, p_frame, p_onset, p_offset, file_id, save_path):
    duration_sec = len(audio) / SR
    zoom_sec = 12 # Un poco más de zoom para ver el pedal
    
    if duration_sec > zoom_sec:
        max_frame = int(zoom_sec * SR / HOP_LENGTH)
        max_sample = int(zoom_sec * SR)
        
        # Cortes
        audio_cut = audio[:max_sample]
        hcqt_cut = hcqt[:max_frame] 
        frames_cut = frames[:max_frame]
        onsets_cut = onsets[:max_frame]
        offsets_cut = offsets[:max_frame]
        vel_cut = velocities[:max_frame]
        
        # Pedal Cortes
        p_frame_cut = p_frame[:max_frame]
        p_onset_cut = p_onset[:max_frame]
        p_offset_cut = p_offset[:max_frame]
        
        extent_cut = [0, zoom_sec, 0, 88]
        time_axis = np.linspace(0, zoom_sec, max_frame)
    else:
        # Full
        max_frame = frames.shape[0]
        audio_cut = audio
        hcqt_cut = hcqt
        frames_cut = frames
        onsets_cut = onsets
        offsets_cut = offsets
        vel_cut = velocities
        
        p_frame_cut = p_frame
        p_onset_cut = p_onset
        p_offset_cut = p_offset
        
        extent_cut = [0, duration_sec, 0, 88]
        time_axis = np.linspace(0, duration_sec, max_frame)

    # Visualizamos el Canal 1 (Fundamental).
    hcqt_viz = hcqt_cut[:, :, 1].T 

    # Creamos 7 subplots
    fig, ax = plt.subplots(7, 1, figsize=(12, 22), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    
    # 1. AUDIO
    ax[0].set_title(f"1. Audio - {file_id}")
    librosa.display.waveshow(audio_cut, sr=SR, ax=ax[0], color='#333333', alpha=0.6)
    
    # 2. HCQT
    ax[1].set_title(f"2. HCQT (Fundamental) - Shape: {hcqt.shape}")
    ax[1].imshow(hcqt_viz, aspect='auto', origin='lower', extent=extent_cut, cmap='magma')
    
    # 3. FRAMES (Notas)
    ax[2].set_title("3. Target: Note Frames (Roll)")
    ax[2].imshow(frames_cut.T, aspect='auto', origin='lower', extent=extent_cut, cmap='binary')
    
    # 4. ONSETS (Notas)
    ax[3].set_title("4. Target: Note Onsets")
    ax[3].imshow(onsets_cut.T, aspect='auto', origin='lower', extent=extent_cut, cmap='Reds')
    
    # 5. OFFSETS (Notas) - NUEVO
    ax[4].set_title("5. Target: Note Offsets (NEW)")
    ax[4].imshow(offsets_cut.T, aspect='auto', origin='lower', extent=extent_cut, cmap='Blues')
    
    # 6. VELOCITY (Notas)
    ax[5].set_title("6. Target: Velocity")
    im = ax[5].imshow(vel_cut.T, aspect='auto', origin='lower', extent=extent_cut, cmap='inferno')
    
    # 7. PEDAL (Sustain) - NUEVO
    ax[6].set_title("7. Target: Sustain Pedal (CC 64)")
    # Dibujamos el Frame como área gris
    ax[6].fill_between(time_axis, 0, p_frame_cut.flatten(), color='gray', alpha=0.3, label='Pedal Frame (Active)')
    # Dibujamos Onset (Verde) y Offset (Azul) como líneas
    ax[6].plot(time_axis, p_onset_cut.flatten(), color='green', linewidth=1.5, label='Pedal Onset')
    ax[6].plot(time_axis, p_offset_cut.flatten(), color='blue', linewidth=1.5, label='Pedal Offset', linestyle='--')
    ax[6].set_ylim(0, 1.1)
    ax[6].legend(loc='upper right', fontsize='small')
    ax[6].set_xlabel("Time (s)")

    # Colorbar para velocity
    cbar = fig.colorbar(im, ax=ax[5], orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label("Vel")
    
    out_file = save_path / f"verify_{file_id}.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"   📸 Debug guardado: {out_file.name}")

def procesar_dataset():
    root_path = Path("data/maestro-v3.0.0") 
    output_base = Path("processed_data_HPPNET_4_h_pedal_5") 
    
    # Agregamos carpetas para OFFSETS y PEDAL
    folders = [
        "inputs_hcqt", 
        "targets_onset", 
        "targets_offset", 
        "targets_frame", 
        "targets_velocity",
        "targets_pedal_frame",  # Nuevo
        "targets_pedal_onset",  # Nuevo
        "targets_pedal_offset"  # Nuevo
    ]
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

    print(f"\n🚀 PROCESAMIENTO ACTUALIZADO (Con Pedal y Offsets)")
    print(f"🔹 Archivos: {num_to_process}")
    print(f"🔹 Output: {output_base.resolve()}")

    viz_count = 0 
    MAX_VIZ = 5 # Aumentamos un poco para ver ejemplos con pedal

    for wav_path in tqdm(selected_wavs):
        midi_path = wav_path.with_suffix(".mid")
        if not midi_path.exists(): midi_path = wav_path.with_suffix(".midi")
        if not midi_path.exists(): continue

        file_id = f"{wav_path.parent.name}_{wav_path.stem}"

        # 1. Calcular HCQT
        audio_raw, hcqt = compute_hcqt(wav_path)
        
        # 2. Calcular Etiquetas (Notas + Pedal)
        onsets, offsets, frames, vels, p_frame, p_onset, p_offset = compute_labels(midi_path, hcqt.shape[0])

        if onsets is None: continue

        # 3. Guardar todo
        np.save(output_base / "inputs_hcqt" / f"{file_id}.npy", hcqt)
        
        # Notas
        np.save(output_base / "targets_onset" / f"{file_id}.npy", onsets)
        np.save(output_base / "targets_offset" / f"{file_id}.npy", offsets)
        np.save(output_base / "targets_frame" / f"{file_id}.npy", frames)
        np.save(output_base / "targets_velocity" / f"{file_id}.npy", vels)
        
        # Pedal
        np.save(output_base / "targets_pedal_frame" / f"{file_id}.npy", p_frame)
        np.save(output_base / "targets_pedal_onset" / f"{file_id}.npy", p_onset)
        np.save(output_base / "targets_pedal_offset" / f"{file_id}.npy", p_offset)

        if viz_count < MAX_VIZ:
            save_verification_plot(
                audio_raw, hcqt, 
                onsets, offsets, frames, vels, 
                p_frame, p_onset, p_offset, 
                file_id, viz_path
            )
            viz_count += 1

    print("\n🏆 ¡Proceso finalizado!")
    print("⚠️ IMPORTANTE: Recuerda que 'targets_pedal_frame' es shape (Time, 1).")
    print("   Asegúrate de que tu Dataset Loader lo lea correctamente.")

if __name__ == "__main__":
    procesar_dataset()