import os
import random
import numpy as np
from pathlib import Path

def inspect_random_song():
    # 1. Configuración de rutas
    base_path = Path("processed_data")
    
    # Verificamos que existan las carpetas
    if not base_path.exists():
        print("❌ No encuentro la carpeta 'processed_data'. ¿Ejecutaste el paso 3?")
        return

    # 2. Obtener lista de archivos disponibles (mirando en inputs_cqt)
    input_dir = base_path / "inputs_cqt"
    files = list(input_dir.glob("*.npy"))
    
    if not files:
        print("❌ No hay archivos .npy en inputs_cqt.")
        return
        
    # 3. Seleccionar uno al azar
    chosen_file = random.choice(files)
    file_id = chosen_file.name # Ej: "2004_song.npy"
    
    print(f"\n🎵 CANCIÓN SELECCIONADA: {file_id}")
    print("=" * 60)
    
    # 4. Definir qué vamos a cargar
    vectors_to_load = {
        "INPUT (CQT)":      base_path / "inputs_cqt" / file_id,
        "TARGET (Onset)":   base_path / "targets_onset" / file_id,
        "TARGET (Offset)":  base_path / "targets_offset" / file_id,
        "TARGET (Frame)":   base_path / "targets_frame" / file_id,
        "TARGET (Velocity)": base_path / "targets_velocity" / file_id
    }
    
    # 5. Cargar e imprimir detalles
    for name, path in vectors_to_load.items():
        if not path.exists():
            print(f"⚠️ Falta el archivo para {name}")
            continue
            
        # Cargar matriz
        data = np.load(path)
        
        print(f"\n🔹 {name}")
        print(f"   📂 Ruta: .../{path.parent.name}/{path.name}")
        print(f"   📐 Forma (Shape): {data.shape}  -> [Frames de Tiempo, 88 Teclas]")
        print(f"   💾 Tipo de Dato: {data.dtype}")
        
        # Análisis de valores
        min_val = data.min()
        max_val = data.max()
        mean_val = data.mean()
        
        print(f"   📊 Rango de valores: Min [{min_val:.4f}] | Max [{max_val:.4f}] | Media [{mean_val:.4f}]")
        
        # Imprimir una muestra real (el primer frame que no sea puro silencio)
        # Buscamos un frame donde haya algo > 0
        non_zero_indices = np.where(data.sum(axis=1) > 0)[0]
        
        if len(non_zero_indices) > 0:
            sample_idx = non_zero_indices[0] + 100 # Tomamos uno 100 frames después del inicio para ver acción
            if sample_idx >= len(data): sample_idx = non_zero_indices[0]
            
            row = data[sample_idx]
            
            # Mostramos solo los valores > 0 para que sea legible
            active_notes = np.where(row > 0.01)[0] # Filtramos ruido
            
            print(f"   👀 Vistazo al Frame #{sample_idx} (Solo valores activos):")
            if len(active_notes) > 0:
                print(f"      Teclas activas (Índices 0-87): {active_notes}")
                values = row[active_notes]
                print(f"      Valores en esas teclas: {np.round(values, 3)}")
            else:
                print("      (Frame de silencio/vacío)")
        else:
            print("      (La matriz parece estar vacía/silencio total)")

    print("\n" + "=" * 60)
    print("✅ Inspección completada. Estos son los números que verá tu IA.")

if __name__ == "__main__":
    inspect_random_song()