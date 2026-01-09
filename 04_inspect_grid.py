import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import random

# --- CONFIGURACIÓN ---
SR = 16000
HOP_LENGTH = 512
FRAME_TIME = HOP_LENGTH / SR  # 0.032s

def midi_to_name(midi_number):
    return librosa.midi_to_note(midi_number + 21)

def plot_paper_style_grid():
    base_path = Path("processed_data_CQT_48")
    if not base_path.exists():
        print("❌ Ejecuta primero el preprocesamiento.")
        return

    # 1. Cargar una canción al azar
    files = list((base_path / "inputs_hcqt").glob("*.npy"))
    if not files:
        print("No hay datos procesados.")
        return

    chosen_file = random.choice(files)
    fid = chosen_file.name
    print(f"Analizando: {fid}")

    # Cargar matrices
    try:
        onsets = np.load(base_path / "targets_onset" / fid)
        frames = np.load(base_path / "targets_frame" / fid)
        offsets = np.load(base_path / "targets_offset" / fid)
        vels = np.load(base_path / "targets_velocity" / fid)
    except FileNotFoundError:
        print("❌ Faltan archivos targets.")
        return

    # 2. Buscar un "momento interesante"
    activity = frames.sum(axis=1)
    active_idxs = np.where(activity > 0)[0]
    
    if len(active_idxs) == 0:
        print("Canción vacía (silencio).")
        return

    center_idx = random.choice(active_idxs)
    
    # 3. Definir ventana de ZOOM
    WINDOW = 14 
    start = max(0, center_idx - WINDOW // 2)
    end = min(len(frames), start + WINDOW)
    
    # Recortar datos y Transponer
    sl_onset = onsets[start:end].T
    sl_frame = frames[start:end].T
    sl_offset = offsets[start:end].T
    sl_vel = vels[start:end].T

    # 4. Filtrar teclas inactivas
    total_activity = sl_onset + sl_frame + sl_offset
    active_keys_idx = np.where(total_activity.sum(axis=1) > 0)[0]
    
    if len(active_keys_idx) == 0:
        print("Slice vacío, reintentando...")
        return plot_paper_style_grid()

    sl_onset = sl_onset[active_keys_idx]
    sl_frame = sl_frame[active_keys_idx]
    sl_offset = sl_offset[active_keys_idx]
    sl_vel = sl_vel[active_keys_idx]
    
    note_names = [librosa.midi_to_note(idx + 21) for idx in active_keys_idx]

    # --- GRAFICADO ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    
    times = np.arange(start, end) * FRAME_TIME
    time_labels = [f"{t:.3f}s" for t in times]

    def plot_matrix(ax, data, title, cmap, is_float=False):
        ax.set_ylabel(title, fontsize=12, fontweight='bold', rotation=0, labelpad=40)
        ax.set_yticks(np.arange(len(note_names)))
        ax.set_yticklabels(note_names)
        
        # Grid
        ax.set_xticks(np.arange(len(time_labels)))
        ax.set_xticklabels(time_labels, rotation=45)
        ax.set_xticks(np.arange(-0.5, len(time_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(note_names), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Poner valores numéricos
        for i in range(data.shape[0]): # Pitch
            for j in range(data.shape[1]): # Time
                val = data[i, j]
                if val > 0:
                    text_color = 'black'
                    # Si es float, mostramos 2 decimales. Si no, entero.
                    txt = f"{val:.2f}" if is_float else f"{int(val)}"
                    # Para floats pequeños (ej. 0.05), quitamos el '0' del principio para ahorrar espacio si quieres
                    if is_float and val < 1: txt = f"{val:.2f}".lstrip('0') 
                    
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=cmap, alpha=0.3 + (val*0.4)) # Alpha dinámico según intensidad
                    ax.add_patch(rect)
                    ax.text(j, i, txt, ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)
                else:
                    ax.text(j, i, "0", ha='center', va='center', color='#cccccc', fontsize=8)

        ax.set_xlim(-0.5, len(time_labels)-0.5)
        ax.set_ylim(-0.5, len(note_names)-0.5)

    # 1. Onset (AHORA ES FLOAT)
    plot_matrix(axes[0], sl_onset, "ONSET\n(High-Res)", "orange", is_float=True)
    # 2. Velocity (Float)
    plot_matrix(axes[1], sl_vel, "VELOCITY", "cyan", is_float=True)
    # 3. Offset (AHORA ES FLOAT)
    plot_matrix(axes[2], sl_offset, "OFFSET\n(High-Res)", "yellow", is_float=True)
    # 4. Frame (Sigue siendo Binario 0/1)
    plot_matrix(axes[3], sl_frame, "FRAME", "green", is_float=False)

    axes[0].set_title(f"Visualización High-Resolution (Paper Style) - {fid}", pad=20)
    plt.xlabel("Tiempo (Segundos)")
    
    save_path = base_path / "debug_visualizations" / f"paper_grid_HR_{fid}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico High-Resolution guardado en: {save_path}")

if __name__ == "__main__":
    plot_paper_style_grid()