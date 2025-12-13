import os
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN ---
MAESTRO_ROOT_DIR = Path("./data/maestro-v3.0.0") 
CSV_FILE = MAESTRO_ROOT_DIR / "maestro-v3.0.0.csv"
OUTPUT_DIR = Path("./dataset/chopin_extracted")

def extract_chopin():
    print("--- 🎹 EXTRACTOR DE CHOPIN (Fuerza extensión .mid) ---")

    if not CSV_FILE.exists():
        print(f"❌ Error CRÍTICO: No encuentro el CSV en: {CSV_FILE}")
        return

    print("Leyendo base de datos...")
    df = pd.read_csv(CSV_FILE)
    
    chopin_df = df[df['canonical_composer'].str.contains("Chopin", case=False, na=False)]
    count = len(chopin_df)
    print(f"✅ Se encontraron {count} piezas de Chopin.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nIniciando copia (Audio + MIDI)...")
    
    files_copied = 0
    missing_midis = 0

    for index, row in tqdm(chopin_df.iterrows(), total=count):
        # 1. Obtener rutas del CSV
        midi_rel_path = row['midi_filename']
        audio_rel_path = row['audio_filename']
        
        # 2. Definir ruta origen de Audio
        src_audio = MAESTRO_ROOT_DIR / audio_rel_path

        # 3. LÓGICA CORREGIDA PARA EL MIDI (.midi vs .mid)
        # ---------------------------------------------------------
        src_midi = MAESTRO_ROOT_DIR / midi_rel_path
        
        # Si el archivo NO existe tal cual viene en el CSV, probamos cambiando a .mid
        if not src_midi.exists():
            # Intentamos forzar la extensión .mid
            posible_src_midi = src_midi.with_suffix('.mid')
            if posible_src_midi.exists():
                src_midi = posible_src_midi
        # ---------------------------------------------------------

        # 4. Definir nombres de destino
        safe_title = str(row['canonical_title']).replace("/", "_").replace("\\", "_").replace(":", "").replace('"', '').strip()
        dst_filename_base = f"Chopin_{safe_title}_{row['year']}_id{index}"
        
        # Guardaremos el midi destino siempre como .mid para ser consistentes
        dst_midi = OUTPUT_DIR / f"{dst_filename_base}.mid"
        dst_wav = OUTPUT_DIR / f"{dst_filename_base}.wav"

        try:
            # --- COPIAR MIDI ---
            if src_midi.exists():
                shutil.copy2(src_midi, dst_midi)
            else:
                # Debug: Imprimir qué ruta intentó buscar y falló
                tqdm.write(f"⚠️ FALTA MIDI: Buscado en {src_midi}")
                missing_midis += 1

            # --- COPIAR AUDIO ---
            if src_audio.exists():
                shutil.copy2(src_audio, dst_wav)
            
            if src_midi.exists() or src_audio.exists():
                files_copied += 1
            
        except Exception as e:
            tqdm.write(f"❌ Error: {e}")

    print("-" * 30)
    print(f"🎉 ¡Proceso terminado! Procesados: {files_copied}")
    if missing_midis > 0:
        print(f"⚠️ Aún faltaron {missing_midis} MIDIs. Verifica la carpeta original.")
    print(f"📂 Carpeta: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_chopin()