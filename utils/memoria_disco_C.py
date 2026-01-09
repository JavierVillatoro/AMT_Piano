import shutil

# Ruta del disco que quieres verificar
ruta = "C:/"

# Obtener estadísticas del disco
total, usado, libre = shutil.disk_usage(ruta)

print(f"Espacio total: {total / (1024**3):.2f} GB")
print(f"Espacio usado: {usado / (1024**3):.2f} GB")
print(f"Espacio libre: {libre / (1024**3):.2f} GB")
