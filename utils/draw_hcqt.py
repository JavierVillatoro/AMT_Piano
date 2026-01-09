import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURACIÓN Y CARGA DE AUDIO ---
filename = 'Bill.mp3'  # Asegúrate que el archivo esté en la carpeta
sr = 22050
hop_length = 512 

# Definir el trozo a visualizar
start_time = 30  # Segundo donde quieres empezar (ej. segundo 30)
duration = 10    # Duración en segundos

# Cargar solo el fragmento específico (offset = inicio, duration = duración)
y, sr = librosa.load(filename, sr=sr, offset=start_time, duration=duration)

# --- 2. PARÁMETROS CQT / HCQT ---
fmin_fundamental = librosa.note_to_hz('C1') # ~32.7 Hz
n_bins = 60      # 5 octavas
bins_per_octave = 12
harmonics = [0.5, 1, 2] # Tus armónicos definidos

# --- 3. CÁLCULO DE LA CQT (Gráfica Superior) ---
# Esta es la CQT estándar (fundamental h=1)
cqt_fundamental = librosa.cqt(y, sr=sr, hop_length=hop_length, 
                              fmin=fmin_fundamental, 
                              n_bins=n_bins, 
                              bins_per_octave=bins_per_octave)
cqt_db_top = librosa.amplitude_to_db(np.abs(cqt_fundamental), ref=np.max)


# --- 4. CÁLCULO DE LA HCQT (Gráfica Inferior) ---
# Para visualizar la HCQT en 2D, calculamos la CQT para cada armónico 
# y los combinamos. En AMT, esto normalmente es un tensor 3D.
# Aquí haremos una "Proyección Máxima" para visualizar dónde coinciden los armónicos.

hcqt_layers = []

for h in harmonics:
    # El truco del HCQT es desplazar el fmin por el factor del armónico
    # h=1 -> fmin normal, h=2 -> fmin*2 (una octava arriba), h=0.5 -> fmin*0.5
    fmin_h = fmin_fundamental * h
    
    # Verificar límite de Nyquist para este armónico
    nyquist_limit = fmin_h * (2 ** (n_bins / bins_per_octave))
    if nyquist_limit < sr / 2:
        cqt_h = librosa.cqt(y, sr=sr, hop_length=hop_length, 
                            fmin=fmin_h, 
                            n_bins=n_bins, 
                            bins_per_octave=bins_per_octave)
        # Normalizamos cada capa individualmente
        hcqt_layers.append(np.abs(cqt_h))

# Convertimos la lista a un array 3D y tomamos el Máximo o Promedio para visualizar
# Esto resalta las notas que tienen fuerte presencia armónica
hcqt_stack = np.array(hcqt_layers)
hcqt_combined = np.max(hcqt_stack, axis=0) # "Max pooling" a través de los armónicos
cqt_db_bottom = librosa.amplitude_to_db(hcqt_combined, ref=np.max)


# --- 5. GRAFICAR Y GUARDAR ---
fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Gráfica 1: CQT Estándar
img1 = librosa.display.specshow(cqt_db_top, sr=sr, hop_length=hop_length,
                                x_axis='time', y_axis='cqt_hz',
                                fmin=fmin_fundamental, bins_per_octave=bins_per_octave,
                                ax=ax[0], vmin=-80, vmax=0, cmap='magma')
ax[0].set_title(f'CQT Estándar (Fundamental) - Segmento {start_time}s a {start_time+duration}s')
ax[0].set_xlabel('') # Ocultar etiqueta x en la gráfica de arriba
fig.colorbar(img1, ax=ax[0], format='%+2.0f dB')

# Gráfica 2: HCQT Visualización
# Usamos el eje Y de la fundamental para referencia visual de la nota "percibida"
img2 = librosa.display.specshow(cqt_db_bottom, sr=sr, hop_length=hop_length,
                                x_axis='time', y_axis='cqt_hz',
                                fmin=fmin_fundamental, bins_per_octave=bins_per_octave,
                                ax=ax[1], vmin=-80, vmax=0, cmap='viridis')
ax[1].set_title(f'HCQT Combinada (Armónicos: {harmonics})')
ax[1].set_ylabel('Frecuencia (Hz) - Alineada a Fundamental')
fig.colorbar(img2, ax=ax[1], format='%+2.0f dB')

plt.tight_layout()

# Guardar la imagen
output_file = 'Cqt_vs_Hcqt.png'
plt.savefig(output_file, dpi=300)
print(f"Imagen guardada como: {output_file}")

plt.show()