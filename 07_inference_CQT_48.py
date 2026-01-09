import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks

# ==========================================
# ⚙️ CONFIGURACIÓN (Debe coincidir con Training)
# ==========================================
SR = 16000
HOP_LENGTH = 320         
MIN_NOTE = 'A0'
MIN_MIDI = 21
NUM_CLASSES = 88

# Configuración High-Res
BINS_PER_OCTAVE = 48     
INPUT_BINS = 352         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Umbrales
THRESHOLD_ONSET = 0.35
THRESHOLD_FRAME = 0.35
THRESHOLD_OFFSET = 0.40

# ==========================================
# 1. ARQUITECTURA (CORREGIDA)
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation=(1, 1)):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_c, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=padding, dilation=dilation, bias=False)
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
        out += identity
        out = self.relu(out)
        return out

class HDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        harmonics = [1, 2, 3, 4, 5, 6, 7, 8] 
        
        for h in harmonics:
            d_math = int(np.round(BINS_PER_OCTAVE * np.log2(h)))
            if d_math == 0:
                dil_pt = (1, 1); pad_pt = (1, 1)
            else:
                dil_pt = (d_math, 1); pad_pt = (d_math, 1)
            
            self.convs.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=(3, 3), 
                padding=pad_pt, 
                dilation=dil_pt
            ))
            
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        x_sum = sum([conv(x) for conv in self.convs])
        return self.fusion(x_sum)

class AcousticModel(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(7, 7), padding=(3,3), bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(base_channels, base_channels)
        self.hdc = HDConv(base_channels, base_channels)
        self.hdc_bn = nn.InstanceNorm2d(base_channels, affine=True)
        self.hdc_relu = nn.ReLU()
        
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        
        self.context = nn.Sequential(
            ResidualBlock(base_channels, base_channels, dilation=(1, 1)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 2)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 4))
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res1(x)
        x_hdc = self.hdc(x) 
        x_hdc = self.hdc_relu(self.hdc_bn(x_hdc))
        x = x + x_hdc 
        x = self.pool(x) 
        x = self.context(x)
        return x

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
        output = output.view(b, f, t) # [Batch, Freq, Time]
        # Salida final: [Batch, Time, Freq]
        return output.permute(0, 2, 1) 

class HPPNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=24, lstm_hidden=128):
        super().__init__()
        self.acoustic_onset = AcousticModel(in_channels, base_channels)
        self.acoustic_other = AcousticModel(in_channels, base_channels)
        
        self.head_onset = FG_LSTM(base_channels, lstm_hidden)
        
        concat_dim = base_channels * 2
        self.head_frame = FG_LSTM(concat_dim, lstm_hidden)
        self.head_offset = FG_LSTM(concat_dim + 1, lstm_hidden) 
        self.head_velocity = FG_LSTM(concat_dim, lstm_hidden)

    def forward(self, x):
        feat_onset = self.acoustic_onset(x)
        logits_onset = self.head_onset(feat_onset) # [Batch, Time, Freq]
        
        feat_onset_detached = feat_onset.detach()
        feat_other = self.acoustic_other(x)
        # feat_combined shape: [Batch, Channels, Freq, Time]
        feat_combined = torch.cat([feat_other, feat_onset_detached], dim=1)
        
        logits_frame = self.head_frame(feat_combined) # [Batch, Time, Freq]
        
        # --- CORRECCIÓN DEL ERROR ---
        # 1. Obtener probs: [Batch, Time, Freq]
        prob_frame = torch.sigmoid(logits_frame).detach() 
        # 2. Girar a [Batch, Freq, Time] para coincidir con feat_combined
        prob_frame = prob_frame.permute(0, 2, 1)          
        # 3. Añadir canal: [Batch, 1, Freq, Time]
        prob_frame = prob_frame.unsqueeze(1)              
        
        feat_offset_in = torch.cat([feat_combined, prob_frame], dim=1)
        
        logits_offset = self.head_offset(feat_offset_in)
        logits_velocity = self.head_velocity(feat_combined)
        
        return logits_onset, logits_frame, logits_offset, logits_velocity

# ==========================================
# 2. PREPROCESAMIENTO
# ==========================================
def compute_input_tensor(audio_path):
    print(f"🔄 Cargando audio... {audio_path}")
    y, _ = librosa.load(str(audio_path), sr=SR)
    
    base_fmin = librosa.note_to_hz(MIN_NOTE)
    
    cqt = librosa.cqt(
        y=y, 
        sr=SR, 
        hop_length=HOP_LENGTH, 
        fmin=base_fmin, 
        n_bins=INPUT_BINS,        
        bins_per_octave=BINS_PER_OCTAVE 
    )
    
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).astype(np.float32)
    cqt_db = (cqt_db + 80.0) / 80.0
    cqt_db = np.clip(cqt_db, 0, 1)

    tensor = torch.tensor(cqt_db).unsqueeze(0).unsqueeze(0).float()
    return tensor

# ==========================================
# 3. MIDI GENERATION
# ==========================================
def matrix_to_midi(onset_probs, frame_probs, offset_probs, velocity_values, output_path, 
                   t_onset=THRESHOLD_ONSET, t_frame=THRESHOLD_FRAME, t_offset=THRESHOLD_OFFSET):
    
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) 
    
    # Transponer si viene como [Time, 88] -> Queremos [88, Time]
    if onset_probs.shape[0] != 88:
        onset_probs = onset_probs.T
        frame_probs = frame_probs.T
        offset_probs = offset_probs.T
        velocity_values = velocity_values.T

    time_per_frame = HOP_LENGTH / SR
    n_frames = frame_probs.shape[1]
    
    print(f"🎹 Generando MIDI... (Frames: {n_frames})")

    for pitch in range(88):
        midi_num = pitch + MIN_MIDI
        
        peaks, _ = find_peaks(onset_probs[pitch, :], height=t_onset, distance=2)
        
        for onset_frame in peaks:
            check_frame = min(onset_frame + 1, n_frames - 1)
            if frame_probs[pitch, check_frame] < t_frame:
                continue 

            end_frame = onset_frame + 1
            while end_frame < n_frames:
                frame_active = frame_probs[pitch, end_frame] > t_frame
                is_offset_hit = offset_probs[pitch, end_frame] > t_offset
                
                if (not frame_active) or is_offset_hit:
                    if is_offset_hit: end_frame += 1 
                    break
                
                end_frame += 1
            
            if end_frame - onset_frame < 2:
                continue
                
            start_time = onset_frame * time_per_frame
            end_time = end_frame * time_per_frame
            
            vel_seg = velocity_values[pitch, onset_frame:min(end_frame, onset_frame+5)]
            vel_val = np.mean(vel_seg) if len(vel_seg) > 0 else 0
            velocity_midi = int(np.clip(vel_val * 127, 20, 127))
            
            note = pretty_midi.Note(velocity=velocity_midi, pitch=midi_num, start=start_time, end=end_time)
            piano.notes.append(note)
            
    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"✅ MIDI guardado en: {output_path}")

# ==========================================
# 4. MAIN (CORREGIDO CONCATENACIÓN)
# ==========================================
def main():
    print("\n🎹 --- INFERENCIA HPPNET HIGH-RES (352 Bins) --- 🎹")
    
    output_dir = Path("midi_output_CQT_48")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar modelos
    possible_models = ["best_hppnet_kaggle.pth", "latest_checkpoint.pth"]
    model_path = None
    for pm in possible_models:
        if os.path.exists(pm):
            model_path = pm
            break
            
    if model_path is None:
        model_path = input("📂 Arrastra el archivo .pth del modelo: ").strip().strip('"')

    if not os.path.exists(model_path):
        print("❌ Modelo no encontrado.")
        return
        
    print(f"🔧 Inicializando HPPNet(in=1, base=24) en {DEVICE}...")
    model = HPPNet(in_channels=1, base_channels=24, lstm_hidden=128).to(DEVICE)
    
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
        print(f"❌ Error crítico cargando pesos: {e}")
        return

    model.eval()
    CHUNK_SIZE = 1000 

    while True:
        print("\n" + "-"*30)
        audio_path = input("🎵 Introduce ruta de AUDIO (o 'q' para salir): ").strip().strip('"')
        
        if audio_path.lower() == 'q': break
        if not os.path.exists(audio_path): 
            print("❌ Archivo no existe")
            continue
            
        filename = Path(audio_path).stem 
        output_midi = output_dir / f"{filename}.mid"
        
        try:
            input_tensor_full = compute_input_tensor(audio_path)
            total_frames = input_tensor_full.shape[3]
            print(f"📏 Frames totales: {total_frames} (resolución 20ms)")

            list_on, list_fr, list_off, list_vel = [], [], [], []

            with torch.no_grad():
                for start in tqdm(range(0, total_frames, CHUNK_SIZE), desc="Procesando"):
                    end = min(start + CHUNK_SIZE, total_frames)
                    chunk = input_tensor_full[:, :, :, start:end].to(DEVICE)
                    
                    p_on, p_fr, p_off, p_vel = model(chunk)
                    
                    # Salida del modelo: [Batch, Time, Freq]
                    list_on.append(p_on.cpu())
                    list_fr.append(p_fr.cpu())
                    list_off.append(p_off.cpu())
                    list_vel.append(p_vel.cpu())
                    
                    del chunk
                    torch.cuda.empty_cache()

            # --- CORRECCIÓN EN MAIN ---
            # Concatenamos en dimensión 1 (Tiempo) porque la salida es [Batch, Time, Freq]
            full_on = torch.cat(list_on, dim=1) 
            full_fr = torch.cat(list_fr, dim=1)
            full_off = torch.cat(list_off, dim=1)
            full_vel = torch.cat(list_vel, dim=1)

            # Convertir a numpy [Total_Frames, 88]
            probs_onset = torch.sigmoid(full_on).numpy()[0]
            probs_frame = torch.sigmoid(full_fr).numpy()[0]
            probs_offset = torch.sigmoid(full_off).numpy()[0]
            probs_vel = torch.sigmoid(full_vel).numpy()[0]

            # matrix_to_midi detectará que está invertido y lo transpondrá internamente
            matrix_to_midi(probs_onset, probs_frame, probs_offset, probs_vel, output_midi)
            
        except Exception as e:
            print(f"❌ Error durante inferencia: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()