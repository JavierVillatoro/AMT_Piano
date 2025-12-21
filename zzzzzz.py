import torch
print(f"PyTorch versión: {torch.__version__}")
print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
print(f"Dispositivo actual: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Ninguno (CPU)'}")