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

# Umbrales Notas
THRESHOLD_ONSET = 0.35
THRESHOLD_FRAME = 0.35
THRESHOLD_OFFSET = 0.40

# Umbrales Pedal (Algoritmo 2)
TH_PED_ON = 0.5
TH_PED_FR = 0.5
TH_PED_OFF = 0.5

# ==========================================
# 1. ARQUITECTURA COMPLETA
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
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=pad_pt, dilation=dil_pt))
            
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

class PedalAcousticModel(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(7, 7), padding=(3,3), bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(base_channels, base_channels)
        self.pool = nn.AdaptiveMaxPool2d((1, None)) 
        self.context = nn.Sequential(
            ResidualBlock(base_channels, base_channels, dilation=(1, 1)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 2)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 4))
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res1(x)
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
        x = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        output = self.proj(output)
        output = output.view(b, f, t) 
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

        self.pedal_acoustic = PedalAcousticModel(in_channels, base_channels)
        self.pedal_head_onset = FG_LSTM(base_channels, lstm_hidden // 2)
        self.pedal_head_frame = FG_LSTM(base_channels, lstm_hidden // 2)
        self.pedal_head_offset = FG_LSTM(base_channels, lstm_hidden // 2)

    def forward(self, x):
        feat_onset = self.acoustic_onset(x)
        logits_onset = self.head_onset(feat_onset) 
        
        feat_onset_detached = feat_onset.detach()
        feat_other = self.acoustic_other(x)
        feat_combined = torch.cat([feat_other, feat_onset_detached], dim=1)
        
        logits_frame = self.head_frame(feat_combined) 
        prob_frame = torch.sigmoid(logits_frame).detach().permute(0, 2, 1).unsqueeze(1)          
        feat_offset_in = torch.cat([feat_combined, prob_frame], dim=1)
        
        logits_offset = self.head_offset(feat_offset_in)
        logits_velocity = self.head_velocity(feat_combined)
        
        feat_pedal = self.pedal_acoustic(x)
        logits_pedal_on = self.pedal_head_onset(feat_pedal)
        logits_pedal_fr = self.pedal_head_frame(feat_pedal)
        logits_pedal_off = self.pedal_head_offset(feat_pedal)
        
        return logits_onset, logits_frame, logits_offset, logits_velocity, logits_pedal_on, logits_pedal_fr, logits_pedal_off

# ==========================================
# 2. PREPROCESAMIENTO AUDIO
# ==========================================
def compute_input_tensor(audio_path):
    print(f"🔄 Cargando audio... {audio_path}")
    y, _ = librosa.load(str(audio_path), sr=SR)
    base_fmin = librosa.note_to_hz(MIN_NOTE)
    cqt = librosa.cqt(y=y, sr=SR, hop_length=HOP_LENGTH, fmin=base_fmin, n_bins=INPUT_BINS, bins_per_octave=BINS_PER_OCTAVE)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).astype(np.float32)
    cqt_db = (cqt_db + 80.0) / 80.0
    cqt_db = np.clip(cqt_db, 0, 1)
    return torch.tensor(cqt_db).unsqueeze(0).unsqueeze(0).float()

# ==========================================
# 3. ALGORITMO DE DECODIFICACIÓN (NOTAS Y PEDAL)
# ==========================================
def decode_pedal(ped_fr, ped_off, t_on=TH_PED_ON, t_fr=TH_PED_FR, t_off=TH_PED_OFF):
    """
    Implementación exacta del 'Algoritmo 2: Transcripción de eventos del pedal'
    """
    events = []
    is_on = False
    start_time = 0.0
    time_per_frame = HOP_LENGTH / SR
    T = len(ped_fr)
    
    for t in range(1, T - 1):
        if not is_on:
            # Línea 5: Detectar inicio (Onset)
            if ped_fr[t] > t_on and ped_fr[t] > ped_fr[t-1]:
                start_time = t * time_per_frame
                is_on = True
        else:
            # Línea 10: Detectar fin (Offset)
            if ped_off[t] > t_off or ped_fr[t] < t_fr:
                
                # Refinamiento a nivel sub-frame analizando A, B y C
                yA = ped_off[t-1]
                yB = ped_off[t]   # Frame central
                yC = ped_off[t+1]
                
                ajuste = 0.0
                
                if yC > yA:
                    ajuste = 0.5 * (yC - yA) / max((yB - yA), 1e-8)
                    exact_frame = t + ajuste
                else:
                    ajuste = 0.5 * (yA - yC) / max((yB - yC), 1e-8)
                    exact_frame = t - ajuste
                    
                end_time = exact_frame * time_per_frame
                
                if end_time > start_time:
                    events.append((start_time, end_time))
                    
                is_on = False
                
    # Caso borde final de la pista
    if is_on:
        events.append((start_time, (T-1) * time_per_frame))
        
    return events

def matrix_to_midi(onset_probs, frame_probs, offset_probs, velocity_values, 
                   ped_fr_probs, ped_off_probs, output_path):
    
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) 
    
    if onset_probs.shape[0] != 88:
        onset_probs = onset_probs.T
        frame_probs = frame_probs.T
        offset_probs = offset_probs.T
        velocity_values = velocity_values.T

    time_per_frame = HOP_LENGTH / SR
    n_frames = frame_probs.shape[1]
    
    print(f"🎹 Generando MIDI... (Frames: {n_frames})")

    # 1. Procesar Notas (Igual que antes)
    for pitch in range(88):
        midi_num = pitch + MIN_MIDI
        peaks, _ = find_peaks(onset_probs[pitch, :], height=THRESHOLD_ONSET, distance=2)
        
        for onset_frame in peaks:
            check_frame = min(onset_frame + 1, n_frames - 1)
            if frame_probs[pitch, check_frame] < THRESHOLD_FRAME: continue 

            end_frame = onset_frame + 1
            while end_frame < n_frames:
                frame_active = frame_probs[pitch, end_frame] > THRESHOLD_FRAME
                is_offset_hit = offset_probs[pitch, end_frame] > THRESHOLD_OFFSET
                
                if (not frame_active) or is_offset_hit:
                    if is_offset_hit: end_frame += 1 
                    break
                end_frame += 1
            
            if end_frame - onset_frame < 2: continue
                
            start_time = onset_frame * time_per_frame
            end_time = end_frame * time_per_frame
            
            vel_seg = velocity_values[pitch, onset_frame:min(end_frame, onset_frame+5)]
            vel_val = np.mean(vel_seg) if len(vel_seg) > 0 else 0
            velocity_midi = int(np.clip(vel_val * 127, 20, 127))
            
            note = pretty_midi.Note(velocity=velocity_midi, pitch=midi_num, start=start_time, end=end_time)
            piano.notes.append(note)

    # 2. Procesar Pedal usando el Algoritmo 2
    pedal_intervals = decode_pedal(ped_fr_probs, ped_off_probs)
    print(f"🦶 Eventos de pedal detectados: {len(pedal_intervals)}")
    
    for start_t, end_t in pedal_intervals:
        # Añadir evento de pisar el pedal (Value=127)
        piano.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=start_t))
        # Añadir evento de soltar el pedal (Value=0)
        piano.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=end_t))
            
    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"✅ MIDI guardado en: {output_path}")

# ==========================================
# 4. BUCLE PRINCIPAL
# ==========================================
def main():
    print("\n🎹 --- INFERENCIA HPPNET HIGH-RES + PEDAL --- 🎹")
    
    output_dir = Path("midi_output_split_100")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar modelos (Añadido tu nuevo nombre como primera opción)
    possible_models = [""]
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
        
    print(f"🔧 Inicializando modelo desde {model_path} en {DEVICE}...")
    model = HPPNet(in_channels=1, base_channels=24, lstm_hidden=128).to(DEVICE)
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
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

            l_on, l_fr, l_off, l_vel = [], [], [], []
            l_ped_on, l_ped_fr, l_ped_off = [], [], []

            with torch.no_grad():
                for start in tqdm(range(0, total_frames, CHUNK_SIZE), desc="Procesando"):
                    end = min(start + CHUNK_SIZE, total_frames)
                    chunk = input_tensor_full[:, :, :, start:end].to(DEVICE)
                    
                    p_on, p_fr, p_off, p_vel, ped_on, ped_fr, ped_off = model(chunk)
                    
                    l_on.append(p_on.cpu()); l_fr.append(p_fr.cpu())
                    l_off.append(p_off.cpu()); l_vel.append(p_vel.cpu())
                    l_ped_on.append(ped_on.cpu()); l_ped_fr.append(ped_fr.cpu())
                    l_ped_off.append(ped_off.cpu())
                    
                    del chunk
                    torch.cuda.empty_cache()

            # Concatenar en la dimensión de tiempo
            full_on, full_fr = torch.cat(l_on, dim=1), torch.cat(l_fr, dim=1)
            full_off, full_vel = torch.cat(l_off, dim=1), torch.cat(l_vel, dim=1)
            full_ped_fr, full_ped_off = torch.cat(l_ped_fr, dim=1), torch.cat(l_ped_off, dim=1)

            # Convertir a numpy 
            probs_onset = torch.sigmoid(full_on).numpy()[0]
            probs_frame = torch.sigmoid(full_fr).numpy()[0]
            probs_offset = torch.sigmoid(full_off).numpy()[0]
            probs_vel = torch.sigmoid(full_vel).numpy()[0]
            
            # El pedal tiene dimensión (Time, 1), lo aplanamos a 1D con flatten()
            probs_pedal_fr = torch.sigmoid(full_ped_fr).numpy()[0].flatten()
            probs_pedal_off = torch.sigmoid(full_ped_off).numpy()[0].flatten()

            matrix_to_midi(probs_onset, probs_frame, probs_offset, probs_vel, probs_pedal_fr, probs_pedal_off, output_midi)
            
        except Exception as e:
            print(f"❌ Error durante inferencia: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()