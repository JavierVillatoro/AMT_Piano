import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from tqdm import tqdm # Importante para ver el progreso de los bloques

# ==========================================
# ⚙️ CONFIGURACIÓN (IDÉNTICA AL TRAINING)
# ==========================================
SR = 16000
HOP_LENGTH = 512
MIN_NOTE = 'A0'
BINS_PER_OCTAVE = 48
N_BINS = 352
MIN_MIDI = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARMONICS = [1, 2, 4]

# 🔥 CONFIGURACIÓN PARA GPU LOCAL (EVITAR OOM)
# 320 frames son ~10 seg. Multiplicamos por 5 para procesar bloques manejables.
CHUNK_SIZE = 320 * 5 

# ==========================================
# 1. ARQUITECTURA HPPNET_SEPARATE (V2)
# ==========================================
# [Mantenemos tus clases HarmonicDilatedConv, AcousticBlock, FG_LSTM y HPPNet_Separate exactamente igual]
class HarmonicDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, bins_per_octave):
        super().__init__()
        self.convs = nn.ModuleList()
        harmonics = [1, 2, 3, 4] 
        dilations = [int(np.round(bins_per_octave * np.log2(h))) for h in harmonics]
        for d in dilations:
            d_safe = max(1, d)
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), 
                                        padding=(d_safe, 1), dilation=(d_safe, 1)))
    def forward(self, x):
        return sum([conv(x) for conv in self.convs])

class AcousticBlock(nn.Module):
    def __init__(self, in_channels, base_channels, bins_per_octave):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1),
                                   nn.InstanceNorm2d(base_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(base_channels, base_channels, 3, padding=1),
                                   nn.InstanceNorm2d(base_channels), nn.ReLU())
        self.hd_conv = HarmonicDilatedConv(base_channels, base_channels, bins_per_octave)
        self.hd_bn = nn.InstanceNorm2d(base_channels)
        self.hd_relu = nn.ReLU()
        self.out_conv = nn.Sequential(nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                                      nn.InstanceNorm2d(base_channels*2), nn.ReLU())
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.hd_relu(self.hd_bn(self.hd_conv(x)))
        return self.out_conv(x)

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
        return output

class HPPNet_Separate(nn.Module):
    def __init__(self, in_channels=3, base_channels=24, lstm_hidden=128):
        super().__init__()
        self.acoustic_onset = AcousticBlock(in_channels, base_channels, BINS_PER_OCTAVE)
        self.acoustic_frame = AcousticBlock(in_channels, base_channels, BINS_PER_OCTAVE)
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        feat_dim = base_channels * 2 
        self.head_onset = FG_LSTM(feat_dim, lstm_hidden)
        self.head_frame = FG_LSTM(feat_dim * 2, lstm_hidden)
        self.head_offset = FG_LSTM(feat_dim * 2, lstm_hidden)
        self.head_velocity = FG_LSTM(feat_dim * 2, lstm_hidden)

    def forward(self, x):
        ft_onset = self.acoustic_onset(x)
        ft_combined = torch.cat([self.acoustic_frame(x), ft_onset.detach()], dim=1)
        out_onset = self.head_onset(self.pool(ft_onset))
        ft_combined_pooled = self.pool(ft_combined)
        out_frame = self.head_frame(ft_combined_pooled)
        out_offset = self.head_offset(ft_combined_pooled)
        out_vel = self.head_velocity(ft_combined_pooled)
        return out_onset, out_frame, out_offset, out_vel

# ==========================================
# 2. PREPROCESAMIENTO HIGH-RES
# ==========================================
def compute_input_tensor(audio_path):
    print(f"🔄 Calculando HCQT Alta Resolución (48 bins/oct) ...")
    y, _ = librosa.load(str(audio_path), sr=SR)
    nyquist = SR / 2 
    base_fmin = librosa.note_to_hz(MIN_NOTE)
    
    cqt_layers = []
    for h in HARMONICS:
        fmin = base_fmin * h
        if fmin >= nyquist: n_bins = 0
        else:
            max_bins = int(np.floor(BINS_PER_OCTAVE * np.log2(nyquist / fmin)))
            n_bins = min(N_BINS, max_bins)

        if n_bins > 0:
            cqt = librosa.cqt(y=y, sr=SR, hop_length=HOP_LENGTH, fmin=fmin, 
                              n_bins=n_bins, bins_per_octave=BINS_PER_OCTAVE)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).T.astype(np.float32)
            cqt_db = np.clip((cqt_db + 80.0) / 80.0, 0, 1)
            if n_bins < N_BINS:
                cqt_db = np.pad(cqt_db, ((0, 0), (0, N_BINS - n_bins)), 'constant')
        else:
            n_frames = librosa.time_to_frames(librosa.get_duration(y=y, sr=SR), sr=SR, hop_length=HOP_LENGTH)
            cqt_db = np.zeros((n_frames, N_BINS), dtype=np.float32)
        cqt_layers.append(cqt_db)
    
    hcqt = np.stack(cqt_layers, axis=-1)
    return torch.tensor(hcqt).permute(2, 1, 0).unsqueeze(0).float()

# ==========================================
# 3. MIDI GENERATION
# ==========================================
def matrix_to_midi(on_p, fr_p, vel_p, output_path, on_thresh=0.4, fr_thresh=0.4):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) 
    time_per_frame = HOP_LENGTH / SR
    onsets, frames = on_p > on_thresh, fr_p > fr_thresh

    for pitch in range(88):
        midi_num = pitch + MIN_MIDI
        p_onsets, p_frames, p_vel = onsets[pitch, :], frames[pitch, :], vel_p[pitch, :]
        diff = np.diff(p_onsets.astype(int), prepend=0)
        start_frames = np.where(diff == 1)[0]
        for start_f in start_frames:
            end_f = start_f + 1
            while end_f < len(p_frames) and p_frames[end_f]: end_f += 1
            start_t, end_t = start_f * time_per_frame, end_f * time_per_frame
            if end_t - start_t < 0.03: continue 
            v = int(np.clip(p_vel[start_f] * 127, 30, 120))
            piano.notes.append(pretty_midi.Note(velocity=v, pitch=midi_num, start=start_t, end=end_t))
    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"✅ MIDI generado: {output_path}")

# ==========================================
# 4. MAIN CON SLIDING WINDOW (EL CAMBIO)
# ==========================================
def main():
    print("\n🎹 --- HPPNET INFERENCE V2 (SLIDING WINDOW OPTIMIZED) --- 🎹")
    model_path = input("📂 Arrastra el modelo (.pth): ").strip().strip('"')
    model = HPPNet_Separate(in_channels=3, lstm_hidden=128).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    while True:
        audio_path = input("\n🎵 Arrastra AUDIO (o 'q' para salir): ").strip().strip('"')
        if audio_path.lower() == 'q': break
        if not os.path.exists(audio_path): continue
        
        try:
            # Calculamos el tensor HCQT completo en CPU
            full_tensor = compute_input_tensor(audio_path)
            total_frames = full_tensor.shape[3]
            
            all_on, all_fr, all_vel = [], [], []
            
            print(f"🧩 Procesando bloques de {CHUNK_SIZE} frames...")
            for start in tqdm(range(0, total_frames, CHUNK_SIZE)):
                end = min(start + CHUNK_SIZE, total_frames)
                # Extraemos un trozo y lo subimos a GPU
                chunk = full_tensor[:, :, :, start:end].to(DEVICE)
                
                with torch.no_grad():
                    # Padding temporal interno para que el pooling sea exacto
                    pad_val = (16 - (chunk.shape[3] % 16)) % 16
                    if pad_val > 0: chunk = F.pad(chunk, (0, pad_val))
                    
                    p_on, p_fr, _, p_vel = model(chunk)
                    
                    # Recortamos el padding y pasamos a CPU
                    p_on = p_on[:, :chunk.shape[3]-pad_val, :]
                    p_fr = p_fr[:, :chunk.shape[3]-pad_val, :]
                    p_vel = p_vel[:, :chunk.shape[3]-pad_val, :]
                    
                    all_on.append(torch.sigmoid(p_on).cpu().numpy()[0])
                    all_fr.append(torch.sigmoid(p_fr).cpu().numpy()[0])
                    all_vel.append(torch.sigmoid(p_vel).cpu().numpy()[0])
                
                # Liberar memoria de GPU
                torch.cuda.empty_cache()

            # Unimos los resultados y trasponemos para matrix_to_midi
            on_m = np.concatenate(all_on, axis=0).T
            fr_m = np.concatenate(all_fr, axis=0).T
            ve_m = np.concatenate(all_vel, axis=0).T

            out_midi = Path(audio_path).with_suffix(".mid")
            matrix_to_midi(on_m, fr_m, ve_m, out_midi, on_thresh=0.5, fr_thresh=0.4)
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()