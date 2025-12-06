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
# CQT CONFIG
MIN_NOTE = 'A0'       
N_BINS = 88           
BINS_PER_OCTAVE = 12 
MIN_MIDI = 21         
MAX_MIDI = 108        
NUM_CLASSES = 88      

def ask_user_config():
    print("\n🎛️  CONFIGURACIÓN DEL PREPROCESADO")
    print("-" * 40)
    
    # 1. Preguntar Porcentaje
    percent = 100.0
    while True:
        try:
            val = input("📊 ¿Qué porcentaje del dataset procesar? (0.1 - 100): ")
            percent = float(val)
            if 0 < percent <= 100: break
        except ValueError: pass
        
    # 2. Preguntar Modo (CQT vs HCQT)
    mode = "CQT"
    print("\nSelecciona el tipo de entrada:")
    print("  [1] CQT Estándar (1 Canal) -> Rápido, menos RAM/VRAM.")
    print("  [2] HCQT (3 Canales: 1x, 2x, 3x armónicos) -> Mejor calidad (Paper HPPNet).")
    while True:
        sel = input("👉 Elige opción (1 o 2): ").strip()
        if sel == "1": 
            mode = "CQT"
            break
        if sel == "2": 
            mode = "HCQT"
            break
            
    return percent, mode

def compute_cqt_layer(y, harmonics_multiplier):
    """
    Calcula una capa de CQT. Si la frecuencia excede Nyquist (SR/2),
    calcula menos bins y rellena el resto con ceros (padding).
    """
    base_fmin = librosa.note_to_hz(MIN_NOTE)
    target_fmin = base_fmin * harmonics_multiplier
    
    # Límite de Nyquist
    nyquist = SR / 2
    
    # Calculamos cuántos bins caben antes de romper Nyquist
    # Frecuencia del bin k: f_k = fmin * 2^(k/bpo)
    # Queremos f_k < nyquist
    # k < bpo * log2(nyquist / fmin)
    max_bins_possible = int(np.floor(BINS_PER_OCTAVE * np.log2(nyquist / target_fmin)))
    
    # Si podemos calcular los 88 bins, perfecto. Si no, recortamos.
    actual_bins = min(N_BINS, max_bins_possible)
    
    if actual_bins <= 0:
        # Caso extremo: fmin ya es mayor que nyquist (no debería pasar con piano y 16k)
        return np.zeros((88, int(len(y)/HOP_LENGTH) + 1), dtype=np.float32).T
    
    # Calculamos CQT con los bins seguros
    cqt = librosa.cqt(
        y=y, 
        sr=SR, 
        hop_length=HOP_LENGTH, 
        fmin=target_fmin, 
        n_bins=actual_bins, 
        bins_per_octave=BINS_PER_OCTAVE
    )
    
    # Procesado estándar: Amplitud -> dB -> Normalizar
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).astype(np.float32)
    cqt_db = (cqt_db + 80.0) / 80.0
    cqt_db = np.clip(cqt_db, 0, 1)
    
    # --- PADDING ---
    # Si calculamos menos de 88 bins, rellenamos los agudos con ceros
    if actual_bins < N_BINS:
        time_steps = cqt_db.shape[1]
        full_matrix = np.zeros((N_BINS, time_steps), dtype=np.float32)
        full_matrix[:actual_bins, :] = cqt_db # Copiar datos calculados abajo
        # Los bins superiores (N_BINS - actual_bins) se quedan en 0
        cqt_db = full_matrix
        
    return cqt_db.T # Devolver transpuesto (Time, 88)

def compute_input_representation(audio_path, mode="CQT"):
    # Cargar audio una sola vez
    y, _ = librosa.load(str(audio_path), sr=SR)
    
    if mode == "CQT":
        # Modo Clásico: 1 sola matriz (Time, 88)
        cqt = compute_cqt_layer(y, 1)
        return y, cqt
        
    elif mode == "HCQT":
        # Modo Armónico: 3 matrices apiladas (Time, 88, 3)
        layers = []
        for h in [1, 2, 3]:
            layers.append(compute_cqt_layer(y, h))
            
        # Asegurar que todas las capas tengan la misma longitud temporal 
        # (a veces librosa varía por 1 frame por redondeo)
        min_time = min([L.shape[0] for L in layers])
        layers = [L[:min_time, :] for L in layers]
        
        # Apilar en el eje de canales (último eje)
        # Resultado shape: (Time, 88, 3)
        hcqt = np.stack(layers, axis=-1)
        return y, hcqt

def add_soft_label(matrix, exact_frame, pitch_idx, num_frames, width=3):
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
    J_WIDTH = 3 

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
        
        frame_matrix[start_frame_int:end_frame_int, pitch_idx] = 1
        add_soft_label(onset_matrix, start_frame_float, pitch_idx, num_frames, width=J_WIDTH)
        add_soft_label(offset_matrix, end_frame_float, pitch_idx, num_frames, width=J_WIDTH)
        
        if start_frame_int < num_frames:
            velocity_matrix[start_frame_int, pitch_idx] = note.velocity / 127.0

    return onset_matrix, offset_matrix, frame_matrix, velocity_matrix

def save_verification_plot(audio, inputs, onsets, file_id, save_path, mode):
    # Si es HCQT, cogemos solo el primer canal para visualizar (Canal Fundamental)
    if mode == "HCQT":
        viz_input = inputs[:, :, 0].T # (88, Time)
    else:
        viz_input = inputs.T # (88, Time)

    # Recorte rápido para visualización
    MAX_LEN = 800 # Frames
    if viz_input.shape[1] > MAX_LEN:
        viz_input = viz_input[:, :MAX_LEN]
        onsets_cut = onsets[:MAX_LEN].T
        audio_cut = audio[:MAX_LEN * HOP_LENGTH]
    else:
        onsets_cut = onsets.T
        audio_cut = audio

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    
    ax[0].set_title(f"Audio ({mode})")
    librosa.display.waveshow(audio_cut, sr=SR, ax=ax[0])
    
    ax[1].set_title(f"Input Spectrogram (Canal 0 - Fundamental)")
    ax[1].imshow(viz_input, aspect='auto', origin='lower', cmap='magma')
    
    ax[2].set_title("Target Onsets")
    ax[2].imshow(onsets_cut, aspect='auto', origin='lower', cmap='Reds')
    
    plt.tight_layout()
    plt.savefig(save_path / f"viz_{mode}_{file_id}.png")
    plt.close()

def procesar_dataset():
    root_path = Path("data/maestro-v3.0.0") 
    output_base = Path("processed_data_hcqt") # Nueva carpeta para no mezclar
    
    folders = ["inputs_cqt", "targets_onset", "targets_offset", "targets_frame", "targets_velocity"]
    for sub in folders:
        (output_base / sub).mkdir(parents=True, exist_ok=True)
    
    viz_path = output_base / "debug_visualizations"
    viz_path.mkdir(parents=True, exist_ok=True)

    all_wavs = list(root_path.rglob("*.wav"))
    if not all_wavs:
        print("❌ No se encontraron archivos .wav")
        return

    # --- PREGUNTAR AL USUARIO ---
    percent, mode = ask_user_config()
    
    num_to_process = int(len(all_wavs) * (percent / 100))
    if num_to_process < 1: num_to_process = 1
    
    random.shuffle(all_wavs)
    selected_wavs = all_wavs[:num_to_process]

    print(f"\n🚀 Procesando {num_to_process} archivos en MODO: {mode}")
    print(f"📂 Guardando en: {output_base.resolve()}")

    viz_count = 0 
    MAX_VIZ = 3

    for wav_path in tqdm(selected_wavs):
        midi_path = wav_path.with_suffix(".mid")
        if not midi_path.exists(): midi_path = wav_path.with_suffix(".midi")
        if not midi_path.exists(): continue

        file_id = f"{wav_path.parent.name}_{wav_path.stem}"

        # 1. Calcular Input (CQT o HCQT)
        try:
            audio_raw, input_data = compute_input_representation(wav_path, mode)
        except Exception as e:
            print(f"Error calculando CQT en {wav_path.name}: {e}")
            continue
        
        # 2. Calcular Labels (Usando la longitud temporal del input)
        num_frames = input_data.shape[0]
        onsets, offsets, frames, vels = compute_labels(midi_path, num_frames)

        if onsets is None: continue

        # 3. Guardar Datos
        np.save(output_base / "inputs_cqt" / f"{file_id}.npy", input_data)
        np.save(output_base / "targets_onset" / f"{file_id}.npy", onsets)
        np.save(output_base / "targets_offset" / f"{file_id}.npy", offsets)
        np.save(output_base / "targets_frame" / f"{file_id}.npy", frames)
        np.save(output_base / "targets_velocity" / f"{file_id}.npy", vels)

        # 4. Visualización de prueba
        if viz_count < MAX_VIZ:
            save_verification_plot(audio_raw, input_data, onsets, file_id, viz_path, mode)
            viz_count += 1

    print("\n✅ ¡Proceso completado!")
    print(f"ℹ️ IMPORTANTE: Has procesado los datos en modo {mode}.")
    print(f"📂 Los datos están en 'processed_data_hcqt'. Acuérdate de cambiar la ruta en el training.")

if __name__ == "__main__":
    procesar_dataset()