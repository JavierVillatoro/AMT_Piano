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
# ⚙️ CONFIGURACIÓN FINAL (HIGH-RES CQT + PEDAL)
# ==========================================
SR = 16000            
HOP_LENGTH = 320     # 20ms de resolución temporal

# Configuración del Piano (Targets Notas)
MIN_NOTE = 'A0'       
MIN_MIDI = 21        
MAX_MIDI = 108        
NUM_CLASSES = 88     

# 🔹 CONFIGURACIÓN DE ENTRADA (Input Spectrogram)
BINS_PER_OCTAVE = 48 
N_BINS = 352         

# 🔹 LABEL SMOOTHING
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
    y, _ = librosa.load(str(audio_path), sr=SR)
    base_fmin = librosa.note_to_hz(MIN_NOTE)
    
    cqt = librosa.cqt(
        y=y, 
        sr=SR, 
        hop_length=HOP_LENGTH, 
        fmin=base_fmin, 
        n_bins=N_BINS, 
        bins_per_octave=BINS_PER_OCTAVE
    )
    
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).T.astype(np.float32)
    cqt_db = (cqt_db + 80.0) / 80.0
    cqt_db = np.clip(cqt_db, 0, 1)

    return y, cqt_db[:, :, np.newaxis]

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
        return None, None, None, None, None, None, None

    # Matrices de Notas (Time, 88)
    onset_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    offset_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    frame_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.int8) 
    velocity_matrix = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)

    # Matrices de Pedal (Time, 1)
    pedal_onset_matrix = np.zeros((num_frames, 1), dtype=np.float32)
    pedal_offset_matrix = np.zeros((num_frames, 1), dtype=np.float32)
    pedal_frame_matrix = np.zeros((num_frames, 1), dtype=np.int8)

    time_per_frame = HOP_LENGTH / SR

    # --- 1. PROCESAMIENTO DE NOTAS ---
    for note in pm.instruments[0].notes:
        if note.pitch < MIN_MIDI or note.pitch > MAX_MIDI:
            continue
            
        pitch_idx = note.pitch - MIN_MIDI
        start_frame_float = note.start / time_per_frame
        end_frame_float = (note.end / time_per_frame) + 1.0
        
        start_frame_int = max(0, min(int(round(start_frame_float)), num_frames - 1))
        end_frame_int = max(0, min(int(round(end_frame_float)), num_frames - 1))

        if end_frame_int <= start_frame_int:
            end_frame_int = start_frame_int + 1
        
        frame_matrix[start_frame_int:end_frame_int, pitch_idx] = 1
        add_soft_label(onset_matrix, start_frame_float, pitch_idx, num_frames, width=J_WIDTH)
        add_soft_label(offset_matrix, end_frame_float, pitch_idx, num_frames, width=J_WIDTH)
        norm_vel = note.velocity / 127.0
        velocity_matrix[start_frame_int:end_frame_int, pitch_idx] = norm_vel

    # --- 2. PROCESAMIENTO DEL PEDAL DE SOSTENIDO (CC 64) ---
    pedal_events = [cc for cc in pm.instruments[0].control_changes if cc.number == 64]
    
    is_pedal_on = False
    pedal_start_time = 0.0

    for cc in pedal_events:
        if cc.value >= 64 and not is_pedal_on:
            is_pedal_on = True
            pedal_start_time = cc.time
            # Añadir Onset Soft Label
            start_frame_float = pedal_start_time / time_per_frame
            add_soft_label(pedal_onset_matrix, start_frame_float, 0, num_frames, width=J_WIDTH)

        elif cc.value < 64 and is_pedal_on:
            is_pedal_on = False
            pedal_end_time = cc.time
            
            # Rellenar Frame
            start_frame_int = max(0, min(int(round(pedal_start_time / time_per_frame)), num_frames - 1))
            end_frame_int = max(0, min(int(round(pedal_end_time / time_per_frame)), num_frames - 1))
            if end_frame_int <= start_frame_int:
                end_frame_int = start_frame_int + 1
            pedal_frame_matrix[start_frame_int:end_frame_int, 0] = 1
            
            # Añadir Offset Soft Label
            end_frame_float = pedal_end_time / time_per_frame
            add_soft_label(pedal_offset_matrix, end_frame_float, 0, num_frames, width=J_WIDTH)

    # Caso borde: si la canción termina con el pedal pisado
    if is_pedal_on:
        start_frame_int = max(0, min(int(round(pedal_start_time / time_per_frame)), num_frames - 1))
        pedal_frame_matrix[start_frame_int:, 0] = 1

    return onset_matrix, offset_matrix, frame_matrix, velocity_matrix, pedal_onset_matrix, pedal_offset_matrix, pedal_frame_matrix

def save_verification_plot(audio, hcqt, onsets, offsets, frames, velocities, ped_on, ped_off, ped_fr, file_id, save_path):
    duration_sec = len(audio) / SR
    zoom_sec = 10
    
    if duration_sec > zoom_sec:
        max_frame = int(zoom_sec * SR / HOP_LENGTH)
        max_sample = int(zoom_sec * SR)
        audio_cut = audio[:max_sample]
        hcqt_cut = hcqt[:max_frame] 
        frames_cut = frames[:max_frame]
        onsets_cut = onsets[:max_frame]
        offsets_cut = offsets[:max_frame] 
        vel_cut = velocities[:max_frame]
        
        # Cortes de pedal
        ped_on_cut = ped_on[:max_frame]
        ped_off_cut = ped_off[:max_frame]
        ped_fr_cut = ped_fr[:max_frame]
        
        extent_time = zoom_sec
    else:
        # (Omitido por brevedad, igual que arriba pero sin cortar)
        audio_cut, hcqt_cut, frames_cut, onsets_cut, offsets_cut, vel_cut = audio, hcqt, frames, onsets, offsets, velocities
        ped_on_cut, ped_off_cut, ped_fr_cut = ped_on, ped_off, ped_fr
        extent_time = duration_sec

    hcqt_viz = hcqt_cut[:, :, 0].T 

    # 🔹 Aumentamos a 7 filas para acomodar el pedal
    fig, ax = plt.subplots(7, 1, figsize=(14, 22), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    
    extent_spec = [0, extent_time, 0, N_BINS]      
    extent_piano = [0, extent_time, 0, NUM_CLASSES] 
    extent_pedal = [0, extent_time, 0, 1] # Solo 1 canal para el pedal
    
    ax[0].set_title(f"1. Audio - {file_id}")
    librosa.display.waveshow(audio_cut, sr=SR, ax=ax[0], color='#333333')
    
    ax[1].set_title(f"2. High-Res CQT (Input) - Shape: {hcqt.shape}")
    ax[1].imshow(hcqt_viz, aspect='auto', origin='lower', extent=extent_spec, cmap='magma')
    
    ax[2].set_title("3. Target NOTAS: Frames")
    ax[2].imshow(frames_cut.T, aspect='auto', origin='lower', extent=extent_piano, cmap='binary')
    
    ax[3].set_title("4. Target NOTAS: Onsets")
    ax[3].imshow(onsets_cut.T, aspect='auto', origin='lower', extent=extent_piano, cmap='Reds')
    
    ax[4].set_title("5. Target NOTAS: Offsets")
    ax[4].imshow(offsets_cut.T, aspect='auto', origin='lower', extent=extent_piano, cmap='Blues')
    
    ax[5].set_title("6. Target NOTAS: Velocity")
    im = ax[5].imshow(vel_cut.T, aspect='auto', origin='lower', extent=extent_piano, cmap='inferno')
    
    # --- NUEVO PLOT: PEDAL DE SOSTENIDO ---
    ax[6].set_title("7. Target PEDAL: Frame (Negro), Onset (Verde), Offset (Rojo)")
    # Superponemos Frame, Onset y Offset en la misma pista para ahorrar espacio
    ax[6].imshow(ped_fr_cut.T, aspect='auto', origin='lower', extent=extent_pedal, cmap='binary', alpha=0.5)
    ax[6].plot(np.linspace(0, extent_time, len(ped_on_cut)), ped_on_cut, color='green', label='Onset', linewidth=1.5)
    ax[6].plot(np.linspace(0, extent_time, len(ped_off_cut)), ped_off_cut, color='red', label='Offset', linewidth=1.5)
    ax[6].set_yticks([]) # Ocultamos el eje Y porque es binario
    ax[6].legend(loc='upper right')
    ax[6].set_xlabel("Time (s)")
    
    out_file = save_path / f"verify_{file_id}.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"   📸 Debug guardado: {out_file.name}")

def procesar_dataset():
    root_path = Path("data/maestro-v3.0.0") 
    output_base = Path("processed_data_cqt_pedal") 
    
    # Nuevas carpetas para los targets del pedal
    folders = [
        "inputs_hcqt", 
        "targets_onset", "targets_offset", "targets_frame", "targets_velocity",
        "targets_pedal_onset", "targets_pedal_offset", "targets_pedal_frame"
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

    print(f"\n🚀 PROCESAMIENTO HIGH-RES + PEDAL INICIADO")
    
    viz_count = 0 
    MAX_VIZ = 3

    for wav_path in tqdm(selected_wavs):
        midi_path = wav_path.with_suffix(".mid")
        if not midi_path.exists(): midi_path = wav_path.with_suffix(".midi")
        if not midi_path.exists(): continue

        file_id = f"{wav_path.parent.name}_{wav_path.stem}"

        audio_raw, hcqt = compute_hcqt(wav_path)
        
        # 🔹 Ahora extraemos 7 variables (4 notas + 3 pedal)
        t_on, t_off, t_fr, t_vel, p_on, p_off, p_fr = compute_labels(midi_path, hcqt.shape[0])

        if t_on is None: continue

        # Guardar Notas
        np.save(output_base / "inputs_hcqt" / f"{file_id}.npy", hcqt)
        np.save(output_base / "targets_onset" / f"{file_id}.npy", t_on)
        np.save(output_base / "targets_offset" / f"{file_id}.npy", t_off)
        np.save(output_base / "targets_frame" / f"{file_id}.npy", t_fr)
        np.save(output_base / "targets_velocity" / f"{file_id}.npy", t_vel)
        
        # Guardar Pedal
        np.save(output_base / "targets_pedal_onset" / f"{file_id}.npy", p_on)
        np.save(output_base / "targets_pedal_offset" / f"{file_id}.npy", p_off)
        np.save(output_base / "targets_pedal_frame" / f"{file_id}.npy", p_fr)

        if viz_count < MAX_VIZ:
            save_verification_plot(audio_raw, hcqt, t_on, t_off, t_fr, t_vel, p_on, p_off, p_fr, file_id, viz_path)
            viz_count += 1

    print("\n🏆 ¡Proceso finalizado con éxito!")

if __name__ == "__main__":
    procesar_dataset()