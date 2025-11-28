import platform
import psutil
import subprocess
import shutil

def print_header(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

# -------------------------------
# CPU
# -------------------------------
def cpu_info():
    print_header("CPU INFO")
    print("Procesador:", platform.processor() or "No disponible")
    print("Arquitectura:", platform.machine())
    print("Núcleos físicos:", psutil.cpu_count(logical=False))
    print("Núcleos lógicos:", psutil.cpu_count(logical=True))
    print("Frecuencia CPU (MHz):", psutil.cpu_freq().current)

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
            print("No se detectó ninguna GPU NVIDIA.")
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
        print("GPUtil no está instalado. Instálalo con: pip install gputil")

# -------------------------------
# CUDA
# -------------------------------
def cuda_info():
    print_header("CUDA INFO")
    # Chequear si existe comando nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
            print(output)
        except Exception:
            print("Error al ejecutar nvidia-smi.")
    else:
        print("nvidia-smi no encontrado. Probablemente no tienes CUDA o GPU NVIDIA.")

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
        print("GPU disponible (TensorFlow):", tf.config.list_physical_devices('GPU'))
    except ImportError:
        print("TensorFlow no está instalado.")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    cpu_info()
    ram_info()
    gpu_info()
    cuda_info()
    dl_libs_info()
