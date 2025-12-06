import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- Necesario para el arreglo del tamaño
import numpy as np
import librosa
import pretty_midi
import soundfile as sf
from pathlib import Path
import sys

# --- CONFIGURACIÓN ---
SR = 16000
HOP_LENGTH = 512
MIN_MIDI = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ARQUITECTURA DEL MODELO
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 1)) if pool else None 

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.pool: x = self.pool(x)
        return x

class PianoCRNN(nn.Module):
    def __init__(self, input_channels=1, hidden_lstm=128):
        super().__init__()
        self.enc1 = ConvBlock(input_channels, 16, pool=False) 
        self.enc2 = ConvBlock(16, 32, pool=True)
        self.enc3 = ConvBlock(32, 64, pool=True)
        
        self.lstm_input_size = 64 * 88 
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_lstm, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_lstm * 2, self.lstm_input_size)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.dec3 = ConvBlock(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 1), stride=(2, 1))
        self.dec2 = ConvBlock(32, 16)
        
        self.head_onset = nn.Conv2d(16, 1, 1)
        self.head_frame = nn.Conv2d(16, 1, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b, c, t, f = e3.shape
        lstm_in = e3.permute(0, 2, 1, 3).reshape(b, t, c*f)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.dropout(self.relu(lstm_out))
        
        proj = self.lstm_proj(lstm_out).view(b, t, c, f).permute(0, 2, 1, 3)
        
        d3 = self.dec3(torch.cat([self.up3(proj), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        
        return self.head_onset(d2).squeeze(1), self.head_frame(d2).squeeze(1)

# ==========================================
# 2. PREPROCESAMIENTO EN TIEMPO REAL
# ==========================================
def compute_cqt_layer(y, harmonics_multiplier):
    base_fmin = librosa.note_to_hz('A0')
    target_fmin = base_fmin * harmonics_multiplier
    nyquist = SR / 2
    max_bins = int(np.floor(12 * np.log2(nyquist / target_fmin)))
    actual_bins = min(88, max_bins)
    
    if actual_bins <= 0: return np.zeros((88, int(len(y)/HOP_LENGTH) + 1), dtype=np.float32).T

    cqt = librosa.cqt(y=y, sr=SR, hop_length=HOP_LENGTH, fmin=target_fmin, n_bins=actual_bins, bins_per_octave=12)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).astype(np.float32)
    cqt_db = (cqt_db + 80.0) / 80.0
    cqt_db = np.clip(cqt_db, 0, 1)
    
    if actual_bins < 88:
        full = np.zeros((88, cqt_db.shape[1]), dtype=np.float32)
        full[:actual_bins, :] = cqt_db
        cqt_db = full
        
    return cqt_db.T

def compute_input(audio_path, mode):
    print(f"🔄 Cargando audio: {audio_path} ...")
    y, _ = librosa.load(audio_path, sr=SR)
    
    # Calcular input base
    if mode == "CQT":
        data = compute_cqt_layer(y, 1) # [Time, 88]
        tensor = torch.tensor(data).unsqueeze(0).unsqueeze(0) # [1, 1, Time, Freq]
    else: # HCQT
        layers = [compute_cqt_layer(y, h) for h in [1, 2, 3]]
        min_len = min([l.shape[0] for l in layers])
        layers = [l[:min_len, :] for l in layers]
        data = np.stack(layers, axis=-1) # [Time, 88, 3]
        tensor = torch.tensor(data).permute(2, 0, 1).unsqueeze(0) # [1, 3, Time, Freq]

    # --- CORRECCIÓN DE DIMENSIONES (PADDING) ---
    # El modelo hace 2 poolings (reducción /4). 
    # Necesitamos que la dimensión 'Time' sea múltiplo de 4.
    time_steps = tensor.shape[2] # Dimensión de tiempo
    pad_needed = (4 - (time_steps % 4)) % 4
    
    if pad_needed > 0:
        # F.pad funciona desde la última dimensión hacia atrás.
        # tensor es [Batch, Channel, Time, Freq]
        # (pad_left, pad_right, pad_top, pad_bottom) -> Freq_L, Freq_R, Time_Top, Time_Bot
        tensor = F.pad(tensor, (0, 0, 0, pad_needed))
        print(f"🔧 Ajustando tamaño: Añadidos {pad_needed} frames de padding.")

    return tensor

# ==========================================
# 3. POST-PROCESADO: MATRIX -> MIDI
# ==========================================
def matrix_to_midi(onset_probs, frame_probs, output_path, onset_thresh=0.5, frame_thresh=0.5):
    """Convierte probabilidades a MIDI"""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) 
    
    onsets = onset_probs > onset_thresh
    frames = frame_probs > frame_thresh
    time_per_frame = HOP_LENGTH / SR
    
    print("🎹 Convirtiendo a MIDI...")
    
    for pitch in range(88):
        midi_num = pitch + MIN_MIDI
        pitch_onsets = onsets[:, pitch]
        pitch_frames = frames[:, pitch]
        
        onset_diff = np.diff(pitch_onsets.astype(int), prepend=0)
        start_idxs = np.where(onset_diff == 1)[0]
        
        for start_frame in start_idxs:
            end_frame = start_frame + 1
            while end_frame < len(pitch_frames) and pitch_frames[end_frame]:
                end_frame += 1
            
            if end_frame == start_frame: end_frame += 1
                
            start_time = start_frame * time_per_frame
            end_time = end_frame * time_per_frame
            
            if end_time - start_time < 0.05: continue
                
            note = pretty_midi.Note(velocity=100, pitch=midi_num, start=start_time, end=end_time)
            piano.notes.append(note)
            
    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"✅ ¡MIDI guardado en: {output_path}!")

# ==========================================
# 4. INTERFAZ DE USUARIO
# ==========================================
def main():
    print("\n🎹 --- AMT PIANO INFERENCE --- 🎹")
    
    # 0. Crear carpeta de salida en la raíz
    output_dir = Path("mis_midis_generados")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Los resultados se guardarán en: {output_dir.resolve()}")
    
    # 1. Seleccionar Modelo
    model_path = input("📂 Arrastra aquí tu archivo .pth (modelo entrenado): ").strip().strip('"')
    if not os.path.exists(model_path):
        print("❌ Archivo no encontrado.")
        return
        
    # 2. Configurar Modo
    print("\n¿Qué tipo de modelo es?")
    print(" [1] CQT Estándar (1 canal)")
    print(" [2] HCQT (3 canales)")
    mode_sel = input("👉 Selección: ").strip()
    
    mode = "HCQT" if mode_sel == "2" else "CQT"
    in_channels = 3 if mode == "HCQT" else 1
    
    # 3. Cargar Modelo
    print(f"\n🔧 Cargando modelo {mode} en {DEVICE}...")
    try:
        model = PianoCRNN(input_channels=in_channels).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando modelo. ¿Seguro que seleccionaste el modo correcto? {e}")
        return

    # 4. Loop de Inferencia
    while True:
        print("\n" + "-"*30)
        audio_path = input("🎵 Arrastra aquí un archivo de AUDIO (o 'q' para salir): ").strip().strip('"')
        
        if audio_path.lower() == 'q': break
        if not os.path.exists(audio_path):
            print("❌ Archivo de audio no encontrado.")
            continue
            
        # --- CAMBIO AQUÍ: GUARDAR EN CARPETA EXTERNA ---
        filename = Path(audio_path).stem  # Obtenemos solo "cancion_ejemplo" (sin .wav)
        output_midi = output_dir / f"{filename}.mid"
        # -----------------------------------------------
        
        try:
            # A) Preprocesado (Con Padding Fix)
            input_tensor = compute_input(audio_path, mode).to(DEVICE)
            
            # B) Inferencia 
            with torch.no_grad():
                pred_onset, pred_frame = model(input_tensor)
                probs_onset = torch.sigmoid(pred_onset).cpu().numpy()[0]
                probs_frame = torch.sigmoid(pred_frame).cpu().numpy()[0]
                
                # C) Recorte del padding (Opcional pero elegante)
                # Si añadimos padding, el resultado será un poco más largo. 
                # matrix_to_midi lo gestionará bien, pero sabedlo.
            
            # D) Generar MIDI
            # Ajusta estos threshold si notas mucho ruido (súbelos a 0.6) o pocas notas (bájalos a 0.3)
            matrix_to_midi(probs_onset, probs_frame, output_midi, onset_thresh=0.4, frame_thresh=0.4)
            
        except Exception as e:
            print(f"❌ Error durante la inferencia: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()