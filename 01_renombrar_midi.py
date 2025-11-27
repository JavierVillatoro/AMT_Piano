import os
from pathlib import Path

def renombrar_extensiones():
    # Definimos la ruta base relativa desde donde se ejecuta el script
    # Según tu estructura es data/maestro-v3.0.0
    base_path = Path("data/maestro-v3.0.0")

    # Verificamos que la carpeta exista antes de empezar
    if not base_path.exists():
        print(f"❌ Error: No se encuentra la ruta {base_path.resolve()}")
        print("Asegúrate de ejecutar este script desde la carpeta raíz 'AMT_Piano_Sheet_Music'")
        return

    print(f"📂 Buscando archivos .midi en: {base_path}\n")

    contador = 0
    
    # rglob('*') busca recursivamente en todas las subcarpetas (2004, 2006, etc.)
    # Buscamos específicamente archivos que terminen en .midi
    for archivo in base_path.rglob("*.midi"):
        # Construimos el nuevo nombre cambiando el sufijo
        nuevo_nombre = archivo.with_suffix(".mid")
        
        try:
            # Renombramos el archivo
            archivo.rename(nuevo_nombre)
            print(f"✅ Renombrado: {archivo.name} -> {nuevo_nombre.name}")
            contador += 1
        except Exception as e:
            print(f"❌ Error al renombrar {archivo.name}: {e}")

    print("-" * 30)
    print(f"🎉 Proceso terminado. Se han renombrado {contador} archivos.")

if __name__ == "__main__":
    renombrar_extensiones()