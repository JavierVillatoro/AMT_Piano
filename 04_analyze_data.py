import os
import numpy as np
import glob
import sys


BASE_DIR = "processed_data"

FOLDERS = {
    "cqt":      os.path.join(BASE_DIR, "inputs_cqt"),
    "onset":    os.path.join(BASE_DIR, "targets_onset"),     
    "offset":   os.path.join(BASE_DIR, "targets_offset"),   
    "frame":    os.path.join(BASE_DIR, "targets_frame"),     
    "velocity": os.path.join(BASE_DIR, "targets_velocity")
}

def analyze_dataset():
    print(f"--- ANALIZANDO DATOS EN: {BASE_DIR} ---\n")

    # 1. Obtener lista de archivos CQT (Input principal)
    # Buscamos todos los .npy en la carpeta de inputs
    cqt_files = sorted(glob.glob(os.path.join(FOLDERS["cqt"], "*.npy")))
    
    if not cqt_files:
        print(f"ERROR: No se encontraron archivos .npy en {FOLDERS['cqt']}")
        return

    print(f"Total de archivos de entrada (CQT) encontrados: {len(cqt_files)}")
    print("-" * 60)

    # Variables para estadísticas globales
    total_frames = 0
    errors_found = 0
    
    # Iteramos sobre cada archivo
    for i, cqt_path in enumerate(cqt_files):
        filename = os.path.basename(cqt_path)
        
        # Intentamos cargar todos los archivos correspondientes a este track
        try:
            # Asumimos que el nombre del archivo es igual en todas las carpetas
            # Si en cqt se llama "track1.npy", en onset buscamos "track1.npy"
            paths = {
                "cqt": cqt_path,
                "onset": os.path.join(FOLDERS["onset"], filename),
                "offset": os.path.join(FOLDERS["offset"], filename),
                "frame": os.path.join(FOLDERS["frame"], filename),
                "velocity": os.path.join(FOLDERS["velocity"], filename)
            }
            
            # Cargar datos
            data = {}
            for key, p in paths.items():
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Falta archivo: {p}")
                data[key] = np.load(p)

            # --- ANÁLISIS INDIVIDUAL ---
            
            # 1. Chequeo de Shapes (Dimensiones)
            # Asumimos shape [Time, Freq] o [Freq, Time]. 
            # Detectamos cuál es tiempo: usualmente es la dimensión variable (la más larga o distinta de 88)
            cqt_shape = data["cqt"].shape
            
            # Buscamos la dimensión de tiempo en el CQT (asumiendo que freq es 88)
            if cqt_shape[0] == 88:
                time_dim_idx = 1
                freq_dim_idx = 0
            else:
                time_dim_idx = 0
                freq_dim_idx = 1
            
            time_frames = cqt_shape[time_dim_idx]
            freq_bins = cqt_shape[freq_dim_idx]
            
            total_frames += time_frames

            # Verificar consistencia temporal entre Input y Targets
            # Todos los targets deben tener el mismo número de frames que el CQT
            mismatches = []
            for key in ["onset", "offset", "frame", "velocity"]:
                # Asumiendo targets siempre son [Time, 88] o [88, Time], buscamos la dim que coincida con time_frames
                target_shape = data[key].shape
                if time_frames not in target_shape:
                    mismatches.append(f"{key}: {target_shape}")
            
            if mismatches:
                print(f"[ERROR] {filename} - Desajuste de tiempo: CQT tiene {time_frames} frames, pero: {mismatches}")
                errors_found += 1
                continue

            # 2. Imprimir detalles solo del primer archivo (para inspección visual)
            if i == 0:
                print(f"\n>>> INFORME DETALLADO DEL PRIMER ARCHIVO: {filename} <<<")
                print(f"{'TIPO':<10} | {'SHAPE':<15} | {'DTYPE':<10} | {'MIN':<8} | {'MAX':<8} | {'MEAN':<8}")
                print("-" * 75)
                for key, arr in data.items():
                    print(f"{key:<10} | {str(arr.shape):<15} | {str(arr.dtype):<10} | {arr.min():.2f}     | {arr.max():.2f}     | {arr.mean():.4f}")
                print("-" * 75)
                
                # Advertencias inteligentes
                if data["cqt"].max() > 10.0:
                    print("⚠️ ADVERTENCIA: Los valores del CQT parecen muy altos (no están normalizados entre 0 y 1). Esto puede dificultar el entrenamiento.")
                if data["onset"].max() > 1:
                    print("⚠️ ADVERTENCIA: Los targets de Onset no son binarios (0 o 1). ¿Quizás son índices midi?")
                if freq_bins != 88:
                    print(f"⚠️ ADVERTENCIA: Se detectaron {freq_bins} bins de frecuencia. El modelo UNet estándar espera 88.")
                print("\nAnalizando el resto de archivos (solo se mostrarán errores)...")

        except FileNotFoundError as e:
            print(f"[FALTA ARCHIVO] Para {filename}: {e}")
            errors_found += 1
        except Exception as e:
            print(f"[ERROR DE LECTURA] En {filename}: {e}")
            errors_found += 1

    # --- RESUMEN FINAL ---
    print("\n" + "="*30)
    print("RESUMEN DEL DATASET")
    print("="*30)
    print(f"Total Tracks procesados: {len(cqt_files)}")
    print(f"Total Frames (Tiempo):   {total_frames}")
    if len(cqt_files) > 0:
        print(f"Promedio Frames/Track:   {total_frames / len(cqt_files):.0f}")
    
    if errors_found == 0:
        print("\n✅ ¡TODO PARECE CORRECTO! Las dimensiones coinciden.")
    else:
        print(f"\n❌ SE ENCONTRARON {errors_found} ERRORES. Revisa los logs de arriba.")

if __name__ == "__main__":
    analyze_dataset()