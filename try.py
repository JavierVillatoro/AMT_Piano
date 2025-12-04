import pathlib
import glob
import os
import sys

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Detectar ruta automáticamente
current_dir = pathlib.Path(__file__).parent.absolute()
if (current_dir / "processed_data").exists():
    DATA_DIR = current_dir / "processed_data"
elif (current_dir.parent / "processed_data").exists():
    DATA_DIR = current_dir.parent / "processed_data"
else:
    print("¡ERROR! No se encuentra la carpeta 'processed_data'")
    sys.exit(1)

print(f"--- ANALIZANDO DATASET EN: {DATA_DIR} ---\n")

# 1. Obtener todos los Inputs
input_path = DATA_DIR / "inputs_cqt"
input_files = sorted(glob.glob(str(input_path / "*.npy")))
total_inputs = len(input_files)

if total_inputs == 0:
    print("¡ERROR! No hay archivos en inputs_cqt. Revisa tu preprocesamiento.")
    sys.exit(1)

print(f"Total archivos de entrada encontrados: {total_inputs}")
print("-" * 60)
print(f"{'NOMBRE DEL ARCHIVO (INPUT)':<50} | {'ONSETS':<8} | {'FRAMES':<8} | {'OFFSETS':<8} | {'VEL':<8}")
print("-" * 60)

# 2. Definir carpetas de targets a revisar
# NOTA: Ajusta los nombres si tus carpetas se llaman diferente (ej: targets_onset vs targets_onsets)
target_folders = {
    "ONSETS":   DATA_DIR / "targets_onset",
    "FRAMES":   DATA_DIR / "targets_frame",
    "OFFSETS":  DATA_DIR / "targets_offset",
    "VEL":      DATA_DIR / "targets_velocity"
}

# Verificar que las carpetas existan antes de empezar
for key, path in target_folders.items():
    if not path.exists():
        print(f"[ALERTA] La carpeta '{path.name}' NO EXISTE. Todos los archivos darán error en esta columna.")

# 3. Bucle Principal: Revisar archivo por archivo
files_ok = 0
files_error = 0

for cqt_file in input_files:
    filename = os.path.basename(cqt_file)
    
    # Estado de cada target para este archivo
    status = {}
    is_fully_complete = True
    
    for key, folder_path in target_folders.items():
        # Construir ruta esperada
        target_file = folder_path / filename
        
        if target_file.exists():
            status[key] = "OK"
        else:
            status[key] = "MISSING"
            is_fully_complete = False
            
            # (Opcional) Intentar buscar nombres alternativos para dar pistas
            # Por ejemplo, si falta "archivo_wav.npy", buscar "archivo.npy"
            alt_name = filename.replace("_wav.npy", ".npy")
            if (folder_path / alt_name).exists():
                status[key] = "NAME?" # Existe pero con otro nombre
    
    # Imprimir fila de la tabla
    # Usamos colores simples (si la terminal lo soporta) o texto
    row_str = f"{filename[:47]+'...':<50} | "
    
    for key in ["ONSETS", "FRAMES", "OFFSETS", "VEL"]:
        val = status.get(key, "N/A")
        if val == "OK":
            row_str += f"  OK     | "
        elif val == "NAME?":
            row_str += f" NAME?   | " # Nombre diferente
        else:
            row_str += f"  NO     | " # No existe

    print(row_str)

    if is_fully_complete:
        files_ok += 1
    else:
        files_error += 1

# 4. Resumen Final
print("-" * 60)
print("RESUMEN DEL DIAGNÓSTICO:")
print(f"Archivos Completos (Listos para entrenar): {files_ok}")
print(f"Archivos con Errores (Faltan targets):     {files_error}")
print("-" * 60)

if files_error > 0:
    print("\nCONSEJOS:")
    print("1. Si ves 'NO' en todas las columnas, revisa si generaste los targets.")
    print("2. Si ves 'NAME?', significa que el archivo existe pero se llama diferente (ej: sin '_wav').")
    print("   -> Solución: Usa el script de entrenamiento 'Flexible' que te pasé antes.")