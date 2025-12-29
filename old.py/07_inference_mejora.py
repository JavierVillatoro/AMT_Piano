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
# ⚙️ CONFIGURACIÓN (Debe coincidir con Training)
# ==========================================
SR = 16000              
HOP_LENGTH = 512        
MIN_NOTE = 'A0'
N_BINS = 88
BINS_PER_OCTAVE = 12   # Importante: Training usó 12
MIN_MIDI = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARMONICS = [0.5, 1, 2] # Armónicos para el HCQT

# ==========================================
# 1. ARQUITECTURA HPPNET-SP (ACTUALIZADA)
# ==========================================
# Copiada exactamente del script de entrenamiento 'train_hppnet_final.py'

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3,3), padding=(1,1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class HDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        harmonics = [1, 2, 3, 4] 
        dilations = [int(np.round(BINS_PER_OCTAVE * np.log2(h))) for h in harmonics]
        
        for d in dilations:
            d_safe = max(1, d)
            self.convs.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=(3, 3), 
                padding=(d_safe, 1), 
                dilation=(d_safe, 1)
            ))
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_sum = sum([conv(x) for conv in self.convs])
        return self.fusion(x_sum)

class AcousticModel(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.block1 = ConvBlock(in_channels, base_channels)
        self.block2 = ConvBlock(base_channels, base_channels)
        
        self.hdc = HDConv(base_channels, base_channels)
        self.hdc_bn = nn.BatchNorm2d(base_channels)
        self.hdc_relu = nn.ReLU()
        
        dilated_blocks = []
        for _ in range(3):
            dilated_blocks.append(nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=(3,3), padding=(1,1)),
                nn.BatchNorm2d(base_channels),
                nn.ReLU()
            ))
        self.context_stack = nn.Sequential(*dilated_blocks)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x_hdc = self.hdc(x)
        x = self.hdc_relu(self.hdc_bn(x_hdc))
        x = self.context_stack(x)
        return x

class FG_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        # x: (B, C, F, T) -> (B, F, T)
        b, c, f, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        output = self.proj(output)
        output = output.view(b, f, t)
        
        # Return (B, T, F)
        return output.permute(0, 2, 1) 

class HPPNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=24, lstm_hidden=32):
        super().__init__()
        # HPPNet-sp (Separated Processing)
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
        
        # Detach & Concat logic
        feat_onset_detached = feat_onset.detach()
        feat_other = self.acoustic_other(x)
        feat_combined = torch.cat([feat_other, feat_onset_detached], dim=1)
        
        logits_frame = self.head_frame(feat_combined)
        logits_offset = self.head_offset(feat_combined)
        logits_velocity = self.head_velocity(feat_combined)
        
        # Output: (Batch, Time, Freq)
        return logits_onset, logits_frame, logits_offset, logits_velocity


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
    # NOTA: En training era (3, 88, Time) dentro del Dataset. Aquí unsqueeze añade Batch.
    tensor = torch.tensor(hcqt_np).permute(2, 1, 0).unsqueeze(0).float()
    
    # Padding de seguridad para convoluciones profundas
    pad_needed = (16 - (tensor.shape[3] % 16)) % 16
    if pad_needed > 0:
        tensor = F.pad(tensor, (0, pad_needed))
        
    return tensor

# ==========================================
# 3. MATRIX TO MIDI
# ==========================================
def matrix_to_midi(onset_probs, frame_probs, offset_probs, velocity_values, output_path, 
                   onset_thresh=0.4, frame_thresh=0.5, offset_thresh=0.7): # <--- SUBIMOS offset a 0.7
    """
    Decodificación corregida para evitar notas cortadas ("staccato").
    """
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) 
    
    # Transponemos para iterar (88, Time)
    if onset_probs.shape[0] != 88:
        onset_probs = onset_probs.T
        frame_probs = frame_probs.T
        offset_probs = offset_probs.T
        velocity_values = velocity_values.T

    time_per_frame = HOP_LENGTH / SR
    
    print(f"🎹 Generando MIDI con lógica SUSTAIN... (Offset Thresh: {offset_thresh})")

    for pitch in range(88):
        midi_num = pitch + MIN_MIDI
        p_onset = onset_probs[pitch, :] 
        p_frame = frame_probs[pitch, :]
        p_offset = offset_probs[pitch, :]
        p_vel = velocity_values[pitch, :]
        
        # 1. Detectar picos de inicio (Peak Picking simple)
        # Esto es mejor que solo > umbral, evita onsets repetidos pegados
        onset_peaks = np.where((p_onset[:-1] < onset_thresh) & (p_onset[1:] >= onset_thresh))[0]
        
        for start_frame in onset_peaks:
            # Corrección de índice por el shift del peak picking
            start_frame += 1 
            
            # Chequeo de seguridad: ¿Hay "cuerpo" (Frame) justo después?
            # Si el modelo dice "Onset" pero al instante siguiente Frame es 0, es ruido.
            if start_frame + 1 < len(p_frame) and p_frame[start_frame+1] < 0.3:
                continue

            end_frame = start_frame + 1
            
            # --- LÓGICA DE CORTE CORREGIDA ---
            while end_frame < len(p_frame):
                frame_val = p_frame[end_frame]
                offset_val = p_offset[end_frame]
                
                # CONDICIÓN 1: El sonido desaparece visualmente (Frame baja mucho)
                died_out = frame_val < frame_thresh
                
                # CONDICIÓN 2: El modelo grita "OFFSET" PERO solo le hacemos caso
                # si el frame también está empezando a bajar (< 0.8).
                # Esto evita cortar notas fuertes que tienen un falso positivo de offset.
                explicit_cut = (offset_val > offset_thresh) and (frame_val < 0.85)
                
                if died_out or explicit_cut:
                    break
                
                end_frame += 1
            # ---------------------------------
            
            # Forzar duración mínima (evita notas de 0.01s)
            if end_frame - start_frame < 3: 
                end_frame = start_frame + 3
            
            start_time = start_frame * time_per_frame
            end_time = end_frame * time_per_frame
            
            vel_pred = p_vel[start_frame]
            velocity_midi = int(np.clip(vel_pred * 127, 25, 127))
            
            note = pretty_midi.Note(velocity=velocity_midi, pitch=midi_num, start=start_time, end=end_time)
            piano.notes.append(note)
            
    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"✅ MIDI guardado en: {output_path}")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
def main():
    print("\n🎹 --- HPPNET-SP INFERENCE (UPDATED) --- 🎹")
    
    output_dir = Path("midi_output_hppnet_10")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = input("📂 Arrastra 'best_hppnet.pth': ").strip().strip('"')
    if not os.path.exists(model_path):
        print("❌ Archivo no encontrado.")
        return
        
    print(f"\n🔧 Cargando modelo en {DEVICE}...")
    
    model = None
    try:
        # Intentamos cargar con hidden=32 (como en el script de training)
        model = HPPNet(in_channels=3, lstm_hidden=32).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("   ✅ Modelo HPPNet-sp (LSTM=32) Cargado correctamente.")
    except RuntimeError:
        try:
            # Fallback por si acaso entrenaste con 48
            print("   ⚠️ Error de dimensiones con 32, probando con 48...")
            model = HPPNet(in_channels=3, lstm_hidden=48).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print("   ✅ Modelo HPPNet-sp (LSTM=48) Cargado correctamente.")
        except Exception as e:
            print(f"❌ Error fatal cargando modelo: {e}")
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
                # El modelo nuevo devuelve 4 salidas
                pred_on, pred_fr, pred_off, pred_vel = model(input_tensor)
                
                # OUTPUT SHAPE DEL MODELO: (Batch, Time, Freq) -> (1, T, 88)
                
                # Transponemos (.T) para que sea (88, Time) que es lo que quiere pretty_midi
                probs_onset = torch.sigmoid(pred_on).cpu().numpy()[0].T 
                probs_frame = torch.sigmoid(pred_fr).cpu().numpy()[0].T
                probs_offset = torch.sigmoid(pred_off).cpu().numpy()[0].T 
                probs_vel = torch.sigmoid(pred_vel).cpu().numpy()[0].T 
                # Offset lo tenemos pero no lo usamos para generar MIDI básico (solo para training)

            matrix_to_midi(probs_onset, probs_frame,probs_offset, probs_vel, output_midi, onset_thresh=0.5, frame_thresh=0.5, offset_thresh=0.3)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()