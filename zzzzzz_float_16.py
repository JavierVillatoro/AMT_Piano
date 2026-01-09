import numpy as np
from pathlib import Path
from tqdm import tqdm

# Lista de carpetas que contienen float32 y queremos bajar a float16
folders_to_compress = [
    "processed_data_HPPNET_J_2/inputs_hcqt",
    "processed_data_HPPNET_J_2/targets_onset",
    "processed_data_HPPNET_J_2/targets_offset",
    "processed_data_HPPNET_J_2/targets_velocity",
    "processed_data_HPPNET_J_2/targets_frame"
]

# NO incluimos 'targets_frame' porque ya es int8 (muy ligero)

print(f"📉 INICIANDO COMPRESIÓN MASIVA (float32 -> float16)")
print("⚠️ Esto modificará los archivos en tu disco para liberar espacio.\n")

total_saved = 0

for folder_str in folders_to_compress:
    folder_path = Path(folder_str)
    if not folder_path.exists():
        print(f"saltando {folder_str} (no existe)")
        continue
        
    files = list(folder_path.glob("*.npy"))
    print(f"📂 Procesando carpeta: {folder_path.name} ({len(files)} archivos)...")
    
    for f in tqdm(files):
        try:
            # 1. Cargar
            data = np.load(f)
            
            # 2. Verificar si vale la pena comprimir
            if data.dtype == np.float32 or data.dtype == np.float64:
                # Calcular ahorro estimado
                original_size = data.nbytes
                
                # 3. Convertir
                data_fp16 = data.astype(np.float16)
                
                # 4. Sobrescribir
                np.save(f, data_fp16)
                
                saved = original_size - data_fp16.nbytes
                total_saved += saved
            
            # Si ya es float16 o int8, no hacemos nada
            
        except Exception as e:
            print(f"Error en {f.name}: {e}")

# Convertir bytes a GB para mostrar
saved_gb = total_saved / (1024**3)
print(f"\n✅ ¡PROCESO TERMINADO!")
print(f"🎉 Has liberado aproximadamente: {saved_gb:.2f} GB de espacio en disco.")
print("   Ahora deberías tener espacio para comprimir y subir a Kaggle.")