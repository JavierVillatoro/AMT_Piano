import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
import sys
from scipy.signal import find_peaks  # <--- Vital para Peak Picking

# ==========================================
# ⚙️ CONFIGURACIÓN (Coincide con Training Kaggle)
# ==========================================
SR = 16000              
HOP_LENGTH = 512        
MIN_NOTE = 'A0'
N_BINS = 88
BINS_PER_OCTAVE = 12   # Coincide con training
MIN_MIDI = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARMONICS = [0.5, 1, 2] # Para input HCQT (3 canales)

# Umbrales de Inferencia (Ajustables)
THRESHOLD_ONSET = 0.35  # Igual que en validación
THRESHOLD_FRAME = 0.5
THRESHOLD_OFFSET = 0.4  # Usado para cortar notas limpiamente

# ==========================================
# 1. ARQUITECTURA (EXACTA DEL ENTRENAMIENTO)
# ==========================================

# --- Bloque Residual ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_c, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_c, affine=True)
        
        self.downsample = None
        if in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_c, affine=True)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # Residual Connection
        out = self.relu(out)
        return out

# --- HDConv (Dilatación Armónica) ---
class HDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        # Armónicos superiores (el 0.5, 1 y 2 ya vienen en el input HCQT)
        harmonics = [1, 2, 3, 4] 
        
        for h in harmonics:
            if h == 1:
                d = 1 
            else:
                d = int(np.round(BINS_PER_OCTAVE * np.log2(h)))
            
            # Padding asimétrico para mantener tamaño
            self.convs.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=(3, 3), 
                padding=(d, 1), 
                dilation=(d, 1)
            ))
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_sum = sum([conv(x) for conv in self.convs])
        return self.fusion(x_sum)

# --- Acoustic Model (Con Residuals) ---
class AcousticModel(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        # Entrada
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU()
        )
        
        # Bloques Residuales Profundos
        self.res1 = ResidualBlock(base_channels, base_channels)
        self.res2 = ResidualBlock(base_channels, base_channels)
        
        # Visión Armónica
        self.hdc = HDConv(base_channels, base_channels)
        self.hdc_bn = nn.InstanceNorm2d(base_channels, affine=True)
        self.hdc_relu = nn.ReLU()
        
        # Contexto (Más residuales)
        self.context = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        
        # Sumamos la visión armónica (Residual Style)
        x_hdc = self.hdc(x)
        x_hdc = self.hdc_relu(self.hdc_bn(x_hdc))
        x = x + x_hdc 
        
        x = self.context(x)
        return x

# --- LSTM Head ---
class FG_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        b, c, f, t = x.shape
        # Permutar para LSTM (Batch*Freq, Time, Channels)
        x = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        output = self.proj(output)
        output = output.view(b, f, t)
        return output.permute(0, 2, 1) 

# --- HPPNet Final ---
class HPPNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=24, lstm_hidden=128): # <--- OJO: 128
        super().__init__()
        self.acoustic_onset = AcousticModel(in_channels, base_channels)
        self.acoustic_other = AcousticModel(in_channels, base_channels)
        
        self.head_onset = FG_LSTM(base_channels, lstm_hidden)
        
        concat_dim = base_channels * 2
        self.head_frame = FG_LSTM(concat_dim, lstm_hidden)
        self.head_offset = FG_LSTM(concat_dim, lstm_hidden)
        self.head_velocity = FG_LSTM(concat_dim, lstm_hidden)

    def forward(self, x):
        feat_onset = self.acoustic_onset(x)
        logits_onset = self.head_onset(feat_onset)
        
        feat_onset_detached = feat_onset.detach()
        feat_other = self.acoustic_other(x)
        feat_combined = torch.cat([feat_other, feat_onset_detached], dim=1)
        
        logits_frame = self.head_frame(feat_combined)
        logits_offset = self.head_offset(feat_combined)
        logits_velocity = self.head_velocity(feat_combined)
        
        return logits_onset, logits_frame, logits_offset, logits_velocity

# ==========================================
# 2. PREPROCESAMIENTO HCQT (SAFE)
# ==========================================
def compute_input_tensor(audio_path):
    print(f"🔄 Cargando audio y calculando HCQT Safe: {audio_path} ...")
    y, _ = librosa.load(str(audio_path), sr=SR)
    
    base_fmin_hz = librosa.note_to_hz(MIN_NOTE)
    nyquist = SR / 2 
    
    cqt_layers = []

    for h in HARMONICS:
        current_fmin = base_fmin_hz * h
        # Lógica de seguridad para evitar frecuencias por encima de Nyquist
        if current_fmin >= nyquist:
            n_bins_possible = 0
        else:
            max_freq_target = current_fmin * (2 ** ((N_BINS - 1) / BINS_PER_OCTAVE))
            if max_freq_target < nyquist:
                n_bins_possible = N_BINS
            else:
                limit_bins = int(np.floor(BINS_PER_OCTAVE * np.log2(nyquist / current_fmin)))
                n_bins_possible = max(0, limit_bins)

        if n_bins_possible > 0:
            cqt = librosa.cqt(y=y, sr=SR, hop_length=HOP_LENGTH, fmin=current_fmin, n_bins=n_bins_possible, bins_per_octave=BINS_PER_OCTAVE)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).T.astype(np.float32)
            cqt_db = (cqt_db + 80.0) / 80.0
            cqt_db = np.clip(cqt_db, 0, 1)
            # Rellenar bins faltantes si se cortaron por Nyquist
            if n_bins_possible < N_BINS:
                missing = N_BINS - n_bins_possible
                cqt_db = np.pad(cqt_db, ((0, 0), (0, missing)), 'constant')
        else:
            # Capa vacía si el armónico es demasiado alto
            n_frames = librosa.time_to_frames(librosa.get_duration(y=y, sr=SR), sr=SR, hop_length=HOP_LENGTH)
            cqt_db = np.zeros((n_frames, N_BINS), dtype=np.float32)
        
        cqt_layers.append(cqt_db)
    
    # Alinear longitud de capas
    min_len = min(layer.shape[0] for layer in cqt_layers)
    cqt_layers = [layer[:min_len, :] for layer in cqt_layers]
    hcqt_np = np.stack(cqt_layers, axis=-1)
    
    # Tensor: (1, 3, 88, Time) - Nota: Permute distinto al training dataset porque aquí añadimos batch manualmente
    tensor = torch.tensor(hcqt_np).permute(2, 1, 0).unsqueeze(0).float()
    
    return tensor

# ==========================================
# 3. MATRIX TO MIDI (CON PEAK PICKING)
# ==========================================
def matrix_to_midi(onset_probs, frame_probs, offset_probs, velocity_values, output_path, 
                   t_onset=THRESHOLD_ONSET, t_frame=THRESHOLD_FRAME, t_offset=THRESHOLD_OFFSET):
    """
    Decodificación usando Peak Picking (find_peaks) para máxima precisión.
    Coincide con la lógica de validación de Kaggle.
    """
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) 
    
    # Transponemos para iterar (88 keys, Time frames)
    # output del modelo viene (Time, 88), así que .T es necesario
    if onset_probs.shape[0] != 88:
        onset_probs = onset_probs.T
        frame_probs = frame_probs.T
        offset_probs = offset_probs.T
        velocity_values = velocity_values.T

    time_per_frame = HOP_LENGTH / SR
    print(f"🎹 Generando MIDI... (Onset: {t_onset}, Frame: {t_frame}, Offset: {t_offset})")

    for pitch in range(88):
        midi_num = pitch + MIN_MIDI
        
        # 1. Peak Picking: Encuentra los picos exactos de ataque
        # distance=2 evita notas dobles instantáneas
        peaks, _ = find_peaks(onset_probs[pitch, :], height=t_onset, distance=2)
        
        for onset_frame in peaks:
            # 2. Validación de Sustain
            # Miramos si hay 'cuerpo' de nota justo después del ataque
            check_frame = min(onset_frame + 1, frame_probs.shape[1] - 1)
            if frame_probs[pitch, check_frame] < t_frame:
                continue # Falso positivo (pico sin cuerpo)

            # 3. Buscar Offset (Final de la nota)
            end_frame = onset_frame + 1
            max_frames = frame_probs.shape[1]
            
            while end_frame < max_frames:
                f_val = frame_probs[pitch, end_frame]
                o_val = offset_probs[pitch, end_frame]
                
                # Criterio de corte:
                # A) La probabilidad de frame baja del umbral (la nota se apaga)
                # B) La probabilidad de offset es alta Y el frame ya no es muy fuerte (< 0.8)
                died_out = f_val < t_frame
                explicit_cut = (o_val > t_offset) and (f_val < 0.85)
                
                if died_out or explicit_cut:
                    break
                end_frame += 1
            
            # Duración mínima de ~30ms para evitar ruido
            if end_frame - onset_frame < 2:
                end_frame = onset_frame + 2
                
            start_time = onset_frame * time_per_frame
            end_time = end_frame * time_per_frame
            
            # 4. Velocity
            # Promedio de los primeros 5 frames para suavizar
            vel_seg = velocity_values[pitch, onset_frame:min(end_frame, onset_frame+5)]
            vel_val = np.mean(vel_seg) if len(vel_seg) > 0 else 0
            velocity_midi = int(np.clip(vel_val * 127, 20, 127))
            
            note = pretty_midi.Note(velocity=velocity_midi, pitch=midi_num, start=start_time, end=end_time)
            piano.notes.append(note)
            
    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"✅ MIDI guardado en: {output_path}")

# ==========================================
# 4. MAIN INTERFACE (CON CHUNKING PARA 4GB VRAM)
# ==========================================
def main():
    print("\n🎹 --- RESIDUAL HPPNET-SP FINAL INFERENCE (4GB VRAM SAFE) --- 🎹")
    
    output_dir = Path("midi_output_residual_kaggle")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Busca el modelo automáticamente si está en la misma carpeta
    default_model = "best_hppnet_kaggle.pth"
    if os.path.exists(default_model):
        model_path = default_model
    else:
        model_path = input("📂 Arrastra 'best_hppnet_kaggle.pth': ").strip().strip('"')

    if not os.path.exists(model_path):
        print("❌ Archivo de modelo no encontrado.")
        return
        
    print(f"\n🔧 Cargando modelo en {DEVICE}...")
    
    model = HPPNet(in_channels=3, lstm_hidden=128).to(DEVICE)
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print("✅ Pesos cargados correctamente.")
    except Exception as e:
        print(f"❌ Error cargando pesos: {e}")
        return

    model.eval()

    # --- CONFIGURACIÓN DEL CHUNKING ---
    # 1000 frames son aprox 32 segundos de audio.
    # Esto consume unos 1.5 GB de VRAM, perfecto para tu tarjeta de 4GB.
    CHUNK_SIZE = 1000 

    while True:
        print("\n" + "-"*30)
        audio_path = input("🎵 Arrastra AUDIO (o 'q'): ").strip().strip('"')
        
        if audio_path.lower() == 'q': break
        if not os.path.exists(audio_path): continue
            
        filename = Path(audio_path).stem 
        output_midi = output_dir / f"{filename}.mid"
        
        try:
            # 1. Calculamos el HCQT entero (Esto suele caber bien, el problema es la LSTM)
            input_tensor_full = compute_input_tensor(audio_path)
            total_frames = input_tensor_full.shape[3]
            print(f"📏 Longitud total: {total_frames} frames. Procesando por trozos...")

            # Listas para guardar los trocitos
            list_on, list_fr, list_off, list_vel = [], [], [], []

            # 2. Bucle de Inferencia por Trozos
            with torch.no_grad():
                for start in tqdm(range(0, total_frames, CHUNK_SIZE), desc="Procesando"):
                    end = min(start + CHUNK_SIZE, total_frames)
                    
                    # Cortamos el trocito: (1, 3, 88, Chunk_Time)
                    chunk = input_tensor_full[:, :, :, start:end].to(DEVICE)
                    
                    # Inferencia del trocito
                    p_on, p_fr, p_off, p_vel = model(chunk)
                    
                    # Guardamos resultados en CPU para liberar GPU
                    list_on.append(p_on.cpu())
                    list_fr.append(p_fr.cpu())
                    list_off.append(p_off.cpu())
                    list_vel.append(p_vel.cpu())
                    
                    # Limpieza explícita de VRAM
                    del chunk, p_on, p_fr, p_off, p_vel
                    torch.cuda.empty_cache()

            # 3. Pegamos todos los trozos (Concatenate)
            full_on = torch.cat(list_on, dim=1)
            full_fr = torch.cat(list_fr, dim=1)
            full_off = torch.cat(list_off, dim=1)
            full_vel = torch.cat(list_vel, dim=1)

            print("🧩 Trozos unidos. Decodificando MIDI...")

            # 4. Convertimos a Numpy y Generamos MIDI
            probs_onset = torch.sigmoid(full_on).numpy()[0]
            probs_frame = torch.sigmoid(full_fr).numpy()[0]
            probs_offset = torch.sigmoid(full_off).numpy()[0]
            probs_vel = torch.sigmoid(full_vel).numpy()[0]

            matrix_to_midi(probs_onset, probs_frame, probs_offset, probs_vel, output_midi)
            
        except Exception as e:
            print(f"❌ Error en inferencia: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()