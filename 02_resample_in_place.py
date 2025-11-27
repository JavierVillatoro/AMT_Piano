import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import os # Nueva importación clave

# --- CONFIGURACIÓN ---
TARGET_SR = 16000  # Frecuencia de muestreo objetivo: 16 kHz

def remuestrear_y_sobrescribir():
    # --- Configuración de Rutas (Asegura la compatibilidad) ---
    project_root = Path(__file__).resolve().parent.parent 
    root_path = project_root / "AMT_Piano_Sheet_Music" / "data" / "maestro-v3.0.0"

    # --- 1. Crear la lista de archivos usando os.walk (Método infalible) ---
    wav_files = []
    
    # Comprobación de existencia (Si esto falla, el directorio raíz no existe)
    if not root_path.exists():
        print(f"ERROR CRÍTICO: No se encontró la ruta principal: {root_path.resolve()}")
        return

    # os.walk recorre garantizadamente todos los subdirectorios
    for root, _, files in os.walk(root_path):
        for file in files:
            # Añadimos solo los archivos que terminan en .wav (ignorando mayúsculas/minúsculas)
            if file.lower().endswith(".wav"):
                wav_files.append(Path(root) / file) 
    # --------------------------------------------------------------------------

    print("-" * 50)
    print(f"🚀 Iniciando remuestreo y sobrescritura de {len(wav_files)} archivos a {TARGET_SR} Hz.")
    print("🚨 ADVERTENCIA: SE SOBREESCRIBIRÁN LOS ARCHIVOS ORIGINALES. 🚨")
    print("-" * 50)
    
    if not wav_files:
        print("ERROR: La lista de archivos sigue vacía. Verifica que los archivos .wav existan dentro de las carpetas de año.")
        return

    contador_resampleado = 0
    contador_saltado = 0

    for wav_path in tqdm(wav_files, desc="Procesando Audios"):
        
        try:
            # 2. Leer SOLO el encabezado para chequear el SR original (rápido)
            _, original_sr = librosa.load(wav_path, sr=None, duration=0.1)
            
            # 3. Verificar si necesita remuestreo
            if original_sr == TARGET_SR:
                contador_saltado += 1
                continue 

            # 4. Cargar el audio completo y remuestrear a TARGET_SR (y mono por defecto)
            y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            
            # 5. SOBREESCRIBIR el archivo original
            sf.write(wav_path, y, TARGET_SR, format='WAV')
            
            contador_resampleado += 1
            
        except Exception as e:
            print(f"\n❌ ERROR CRÍTICO procesando {wav_path.name}: {e}. Archivo no modificado.")

    print("-" * 50)
    print(f"🎉 Proceso terminado. Archivos remuestreados y sobrescritos: {contador_resampleado}")
    print(f"✅ Archivos saltados (ya estaban a {TARGET_SR} Hz): {contador_saltado}")
    print("✔️ La carpeta de origen ahora solo contiene audios a 16 kHz.")

if __name__ == "__main__":
    remuestrear_y_sobrescribir()