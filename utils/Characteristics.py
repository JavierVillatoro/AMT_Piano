import platform
import psutil
import subprocess
import shutil
import os  # <--- Necesitamos 'os' para leer rutas

def print_header(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

# -------------------------------
# ALMACENAMIENTO (DISCO C:) - ¡NUEVO!
# -------------------------------
def disk_info():
    print_header("ALMACENAMIENTO (Disco Principal)")
    
    # Detectamos la ruta raíz dependiendo del sistema operativo
    # En Windows suele ser 'C:\', en Linux/Mac es '/'
    ruta = "C:\\" if platform.system() == "Windows" else "/"
    
    try:
        usage = shutil.disk_usage(ruta)
        
        gb = 1024**3 # Factor de conversión a Gigabytes
        
        print(f"Ruta analizada: {ruta}")
        print(f"Total:      {usage.total / gb:.2f} GB")
        print(f"Usado:      {usage.used / gb:.2f} GB")
        print(f"Disponible: {usage.free / gb:.2f} GB")
        
        # Porcentaje visual
        percent = (usage.used / usage.total) * 100
        print(f"Porcentaje de uso: {percent:.1f}%")
        
    except Exception as e:
        print(f"No se pudo leer el disco: {e}")

# -------------------------------
# CPU
# -------------------------------
def cpu_info():
    print_header("CPU INFO")
    print("Procesador:", platform.processor() or "No disponible")
    print("Arquitectura:", platform.machine())
    print("Núcleos físicos:", psutil.cpu_count(logical=False))
    print("Núcleos lógicos:", psutil.cpu_count(logical=True))
    try:
        freq = psutil.cpu_freq()
        if freq:
            print(f"Frecuencia CPU (MHz): {freq.current:.2f}")
    except:
        print("Frecuencia CPU: No disponible")

# -------------------------------
# RAM
# -------------------------------
def ram_info():
    print_header("MEMORIA RAM")
    ram = psutil.virtual_memory()
    print("Total (GB):", round(ram.total / (1024**3), 2))
    print("Disponible (GB):", round(ram.available / (1024**3), 2))

# -------------------------------
# GPU (NVIDIA)
# -------------------------------
def gpu_info():
    print_header("GPU INFO")
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("No se detectó ninguna GPU NVIDIA (vía GPUtil).")
            return
        for gpu in gpus:
            print(f"ID GPU: {gpu.id}")
            print(f"Modelo: {gpu.name}")
            print(f"VRAM Total (GB): {gpu.memoryTotal/1024:.2f}")
            print(f"VRAM Usada (GB): {gpu.memoryUsed/1024:.2f}")
            print(f"VRAM Libre (GB): {gpu.memoryFree/1024:.2f}")
            print(f"Temperatura: {gpu.temperature} °C")
            print("-" * 40)
    except ImportError:
        print("GPUtil no está instalado. (Opcional, pero recomendado)")

# -------------------------------
# CUDA
# -------------------------------
def cuda_info():
    print_header("CUDA INFO")
    if shutil.which("nvidia-smi"):
        try:
            # Ejecutamos nvidia-smi pero solo mostramos las primeras líneas para no saturar
            output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
            print(output) 
        except Exception:
            print("Error al ejecutar nvidia-smi.")
    else:
        print("nvidia-smi no encontrado.")

# -------------------------------
# Python / Deep Learning libs
# -------------------------------
def dl_libs_info():
    print_header("VERSIONES DE LIBRERÍAS")
    print("Python:", platform.python_version())

    # PyTorch
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA disponible:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Versión CUDA (PyTorch):", torch.version.cuda)
            print("Número de GPUs detectadas:", torch.cuda.device_count())
            print("GPU actual:", torch.cuda.get_device_name(0))
    except ImportError:
        print("PyTorch no está instalado.")

    # TensorFlow
    try:
        import tensorflow as tf
        print("TensorFlow:", tf.__version__)
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU disponible (TensorFlow): {'Sí' if gpus else 'No'} ({len(gpus)} detectadas)")
    except ImportError:
        print("TensorFlow no está instalado.")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    cpu_info()
    ram_info()
    disk_info()  # <--- AÑADIDO AQUÍ
    gpu_info()
    cuda_info()
    dl_libs_info()
