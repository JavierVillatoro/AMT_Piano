import librosa
from pathlib import Path
from tqdm import tqdm

def verificar_sr_dataset():
    # Rutas basadas en tu estructura
    root_path = Path("data/maestro-v3.0.0")
    
    # 1. Buscar todos los archivos .wav recursivamente en subcarpetas
    wav_files = list(root_path.rglob("*.wav"))
    
    # Usamos un 'set' para almacenar solo valores SR únicos
    sr_encontrados = set()
    
    print(f"🔍 Revisando {len(wav_files)} archivos WAV en: {root_path.resolve()}")
    
    for wav_path in tqdm(wav_files, desc="Procesando Audios"):
        try:
            # 2. Cargar solo el encabezado del audio (duration=0.1s)
            # Esto hace que el proceso sea muy rápido, ya que no carga todo el audio.
            _y, sr = librosa.load(wav_path, sr=None, duration=0.1) 
            sr_encontrados.add(sr)
        except Exception as e:
            # Capturar posibles errores de archivos corruptos
            print(f"\n⚠️ Error al leer el SR de {wav_path.name}: {e}")
            
    print("\n--- RESULTADO DE VERIFICACIÓN DE SAMPLE RATE ---")
    
    # 3. Mostrar el resultado final
    if len(sr_encontrados) == 1:
        # Si solo hay un valor, todos son iguales
        sr_comun = sr_encontrados.pop()
        print(f"✅ ¡CONSISTENTE! Todos los archivos tienen un SR de: {sr_comun} Hz")
    elif len(sr_encontrados) > 1:
        # Si hay más de un valor, hay inconsistencias
        print("❌ INCONSISTENTE: Se encontraron múltiples Sample Rates en el dataset.")
        print(f"SRs encontrados: {sorted(list(sr_encontrados))} Hz")
        print("DEBERÍAS REMUESTREAR a un SR común (ej. 16000 Hz o 44100 Hz) antes del entrenamiento.")
    else:
        print("No se encontraron archivos WAV o no se pudo leer el SR de ninguno.")

if __name__ == "__main__":
    verificar_sr_dataset()