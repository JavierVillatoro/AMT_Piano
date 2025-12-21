import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
import sys

# ==========================================
# ⚙️ CONFIGURACIÓN (IGUAL QUE EL TRAINING)
# ==========================================
SR = 16000              
HOP_LENGTH = 512        
MIN_NOTE = 'A0'
N_BINS = 88
BINS_PER_OCTAVE = 12   # 48 HACER CAMBIOS
MIN_MIDI = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARMONICS = [0.5, 1, 2] 

# ==========================================
# 1. ARQUITECTURA HPPNET 
# ==========================================
class HDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        harmonics = [1, 2, 3, 4] 
        dilations = [int(np.round(BINS_PER_OCTAVE * np.log2(h))) for h in harmonics]
        for d in dilations:
            d_safe = max(1, d)
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(d_safe, 1), dilation=(d_safe, 1)))
    def forward(self, x):
        return sum([conv(x) for conv in self.convs])

class FG_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 1)
    def forward(self, x):
        b, c, f, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        output, _ = self.lstm(x)
        output = self.proj(output).reshape(b, f, t).permute(0, 2, 1)
        # Output shape: (Batch, Time, Freq) -> (Batch, Time, 88)
        return output

class HPPNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=24, lstm_hidden=32):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.InstanceNorm2d(base_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.InstanceNorm2d(base_channels), nn.ReLU())
        self.hd_conv = HDConv(base_channels, base_channels)
        self.hd_bn = nn.InstanceNorm2d(base_channels)
        self.hd_relu = nn.ReLU()
        self.deep_conv = nn.Sequential(nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.InstanceNorm2d(base_channels*2), nn.ReLU())
        feat_dim = base_channels * 2
        self.head_onset = FG_LSTM(feat_dim, lstm_hidden)
        self.head_offset = FG_LSTM(feat_dim, lstm_hidden)
        self.head_frame = FG_LSTM(feat_dim, lstm_hidden)
        self.head_velocity = FG_LSTM(feat_dim, lstm_hidden)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.hd_relu(self.hd_bn(self.hd_conv(x)))
        features = self.deep_conv(x) 
        return (self.head_onset(features), self.head_frame(features), self.head_offset(features), self.head_velocity(features))

# ==========================================
# 2. PREPROCESAMIENTO SAFE (NYQUIST PROTECTED)
# ==========================================
def compute_input_tensor(audio_path):
    print(f"🔄 Cargando audio y calculando HCQT Safe: {audio_path} ...")
    y, _ = librosa.load(str(audio_path), sr=SR)
    
    base_fmin_hz = librosa.note_to_hz(MIN_NOTE)
    nyquist = SR / 2 
    
    cqt_layers = []

    for h in HARMONICS:
        current_fmin = base_fmin_hz * h
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
            if n_bins_possible < N_BINS:
                missing = N_BINS - n_bins_possible
                cqt_db = np.pad(cqt_db, ((0, 0), (0, missing)), 'constant')
        else:
            n_frames = librosa.time_to_frames(librosa.get_duration(y=y, sr=SR), sr=SR, hop_length=HOP_LENGTH)
            cqt_db = np.zeros((n_frames, N_BINS), dtype=np.float32)
        
        cqt_layers.append(cqt_db)
    
    min_len = min(layer.shape[0] for layer in cqt_layers)
    cqt_layers = [layer[:min_len, :] for layer in cqt_layers]
    hcqt_np = np.stack(cqt_layers, axis=-1)
    
    # Preparamos tensor (Batch, Channels, Height, Width) -> (1, 3, 88, Time)
    tensor = torch.tensor(hcqt_np).permute(2, 1, 0).unsqueeze(0).float()
    
    pad_needed = (16 - (tensor.shape[3] % 16)) % 16
    if pad_needed > 0:
        tensor = F.pad(tensor, (0, pad_needed))
        
    return tensor

# ==========================================
# 3. MATRIX TO MIDI (CORREGIDO)
# ==========================================
def matrix_to_midi(onset_probs, frame_probs, velocity_values, output_path, onset_thresh=0.4, frame_thresh=0.4):
    """
    IMPORTANTE: Aquí las matrices deben llegar como (88, Time).
    """
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) 
    
    onsets = onset_probs > onset_thresh
    frames = frame_probs > frame_thresh
    time_per_frame = HOP_LENGTH / SR
    
    print(f"🎹 Generando MIDI... Dimensiones Onset: {onsets.shape} (Debe ser 88, Time)")
    
    # Si las dimensiones siguen mal (Time, 88), lanzamos aviso
    if onsets.shape[0] != 88:
        print("⚠️ ALERTA: Las dimensiones parecen invertidas. Intentando auto-corregir dentro de la función...")
        onsets = onsets.T
        frames = frames.T
        velocity_values = velocity_values.T

    for pitch in range(88):
        midi_num = pitch + MIN_MIDI
        pitch_onsets = onsets[pitch, :] 
        pitch_frames = frames[pitch, :]
        pitch_velocity = velocity_values[pitch, :]
        
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
                
            vel_pred = pitch_velocity[start_frame]
            velocity_midi = int(np.clip(vel_pred * 127, 20, 127))
            
            note = pretty_midi.Note(velocity=velocity_midi, pitch=midi_num, start=start_time, end=end_time)
            piano.notes.append(note)
            
    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"✅ MIDI guardado en: {output_path}")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
def main():
    print("\n🎹 --- HPPNET INFERENCE (FIXED DIMENSIONS) --- 🎹")
    
    output_dir = Path("midi_output_hppnet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = input("📂 Arrastra 'best_model_HPPNET.pth': ").strip().strip('"')
    if not os.path.exists(model_path):
        print("❌ Archivo no encontrado.")
        return
        
    print(f"\n🔧 Cargando modelo en {DEVICE}...")
    
    model = None
    try:
        model = HPPNet(in_channels=3, lstm_hidden=32).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("   ✅ LSTM=32 Cargado.")
    except RuntimeError:
        try:
            model = HPPNet(in_channels=3, lstm_hidden=48).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print("   ✅ LSTM=48 Cargado.")
        except Exception as e:
            print(f"❌ Error fatal: {e}")
            return

    model.eval()

    while True:
        print("\n" + "-"*30)
        audio_path = input("🎵 Arrastra AUDIO (o 'q'): ").strip().strip('"')
        
        if audio_path.lower() == 'q': break
        if not os.path.exists(audio_path): continue
            
        filename = Path(audio_path).stem 
        output_midi = output_dir / f"{filename}.mid"
        
        try:
            input_tensor = compute_input_tensor(audio_path).to(DEVICE)
            
            with torch.no_grad():
                pred_on, pred_fr, pred_off, pred_vel = model(input_tensor)
                
                # ---------------------------------------------------------
                # 🔥 EL FIX ESTÁ AQUÍ: AÑADIMOS .T PARA GIRAR LA MATRIZ 🔥
                # ---------------------------------------------------------
                # El modelo escupe: (Batch, Time, 88)
                # Necesitamos:      (88, Time) para pretty_midi
                
                probs_onset = torch.sigmoid(pred_on).cpu().numpy()[0].T  # <--- .T
                probs_frame = torch.sigmoid(pred_fr).cpu().numpy()[0].T  # <--- .T
                probs_vel = torch.sigmoid(pred_vel).cpu().numpy()[0].T   # <--- .T

            matrix_to_midi(probs_onset, probs_frame, probs_vel, output_midi, onset_thresh=0.4, frame_thresh=0.4)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()