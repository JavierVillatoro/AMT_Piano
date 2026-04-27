#!pip install mir_eval 
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mir_eval 
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
from scipy.signal import find_peaks  # <--- NUEVO: Para Peak Picking

# ==========================================
# 0. CONFIGURACIÓN Y ESTÁNDARES
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Usando dispositivo: {DEVICE}")

SEED = 42
SR = 16000           
HOP_LENGTH = 320    #ANTES 512     
SEGMENT_FRAMES = 512   #Para tener los 10 segundos como antes  
BINS_PER_OCTAVE = 48  
INPUT_BINS = 352
# Nota: En Kaggle working directory es donde descomprimimos
DATA_PATH = Path("/kaggle/input/maestro2/processed_data_HPPNET_4_h")
CHECKPOINT_PATH = Path("/kaggle/input/ruta-a-tu-modelo/latest_checkpoint.pth")
 

# Hyperparámetros Optimizados para Kaggle
BATCH_SIZE = 16           # Subido de 4 a 32 para aprovechar GPU P100/T4 , CAMBIAR SI DA OOM 
FINAL_EPOCHS = 50         
LEARNING_RATE = 0.0006    
PATIENCE_LR = 1           # Un poco más de paciencia
FACTOR_LR = 0.6           
NUM_WORKERS = 4           # Kaggle tiene buena CPU I/O

# Umbrales
THRESHOLD_ONSET = 0.35    # Ajustado para Peak Picking
THRESHOLD_FRAME = 0.35     #ANTES 0.5
THRESHOLD_OFFSET = 0.4

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ==========================================
# 1. DATASET
# ==========================================
class PianoDataset(Dataset):
    def __init__(self, processed_dir, split='train', val_split=0.15):
        self.processed_dir = Path(processed_dir)
        p = self.processed_dir / "inputs_hcqt"
        if not p.exists(): raise RuntimeError(f"❌ Ruta no existe: {p}")
        
        all_files = sorted(list(p.glob("*.npy")))
        if len(all_files) == 0: raise RuntimeError(f"❌ No se encontraron archivos .npy en {p}")
        
        random.Random(SEED).shuffle(all_files)
        split_idx = int(len(all_files) * (1 - val_split))
        self.files = all_files[:split_idx] if split == 'train' else all_files[split_idx:]
        
        self.segments = []
        # Pre-calculamos segmentos
        print(f"   Calculando segmentos para {split}...")
        for idx, f in enumerate(self.files):
            try:
                # mmap_mode='r' lee solo la cabecera para ser rápido
                shape = np.load(f, mmap_mode='r').shape
                n_frames = shape[0]
                num_clips = math.ceil(n_frames / SEGMENT_FRAMES)
                for i in range(num_clips):
                    start = i * SEGMENT_FRAMES
                    end = min(start + SEGMENT_FRAMES, n_frames)
                    if (end - start) > 30: 
                        self.segments.append((idx, start, end))
            except: continue
        print(f"   ✅ {split.upper()}: {len(self.segments)} segmentos cargados.")

    def __len__(self): return len(self.segments)

    def __getitem__(self, idx):
        file_idx, start, end = self.segments[idx]
        fid = self.files[file_idx].name
        try:
            base = self.processed_dir
            # Carga con mmap para velocidad
            hcqt = np.load(base / "inputs_hcqt" / fid, mmap_mode='r')[start:end] 
            
            # Verifica que el archivo tenga 352 bins. Si no, fuerza el error para ir al 'except'.
            if hcqt.shape[1] != INPUT_BINS:
                raise ValueError(f"Bad shape in {fid}: {hcqt.shape}")
            
            # Target Notas
            onset = np.load(base / "targets_onset" / fid, mmap_mode='r')[start:end]
            frame = np.load(base / "targets_frame" / fid, mmap_mode='r')[start:end]
            offset = np.load(base / "targets_offset" / fid, mmap_mode='r')[start:end]
            vel = np.load(base / "targets_velocity" / fid, mmap_mode='r')[start:end]
            
            # Target Pedal
            ped_onset = np.load(base / "targets_pedal_onset" / fid, mmap_mode='r')[start:end]
            ped_offset = np.load(base / "targets_pedal_offset" / fid, mmap_mode='r')[start:end]
            ped_frame = np.load(base / "targets_pedal_frame" / fid, mmap_mode='r')[start:end]
            
            curr_len = hcqt.shape[0]
            if curr_len < SEGMENT_FRAMES:
                pad = SEGMENT_FRAMES - curr_len
                hcqt = np.pad(hcqt, ((0, pad), (0,0), (0,0)))
                onset = np.pad(onset, ((0, pad), (0,0)))
                frame = np.pad(frame, ((0, pad), (0,0)))
                offset = np.pad(offset, ((0, pad), (0,0)))
                vel = np.pad(vel, ((0, pad), (0,0)))
                ped_onset = np.pad(ped_onset, ((0, pad), (0,0)))
                ped_offset = np.pad(ped_offset, ((0, pad), (0,0)))
                ped_frame = np.pad(ped_frame, ((0, pad), (0,0)))
            
            hcqt_t = torch.tensor(hcqt).permute(2, 1, 0).float()
            
            return {
                "hcqt": hcqt_t,
                "onset": torch.tensor(onset).float(),
                "frame": torch.tensor(frame).float(),
                "offset": torch.tensor(offset).float(),
                "velocity": torch.tensor(vel).float(),
                "pedal_onset": torch.tensor(ped_onset).float(),  
                "pedal_offset": torch.tensor(ped_offset).float(), 
                "pedal_frame": torch.tensor(ped_frame).float()
            }
        except Exception as e:
            print(f"Error loading {fid}: {e}")
            z = torch.zeros(SEGMENT_FRAMES, 88)
            zp = torch.zeros(SEGMENT_FRAMES, 1) # Tensor de ceros para el pedal (1 canal)
            return {
                "hcqt": torch.zeros(1, INPUT_BINS, SEGMENT_FRAMES), 
                "onset": z, "frame": z, "offset": z, "velocity": z,
                "pedal_onset": zp, "pedal_offset": zp, "pedal_frame": zp
            }

# ==========================================
# 2. ARQUITECTURA FINAL (CON PEDAL INTEGRADO)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class ComboLoss(nn.Module):
    def __init__(self, bce_weight=5.0):
        super(ComboLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_weight]))
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return self.bce(inputs, targets) + self.dice(inputs, targets)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation=(1, 1)):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=padding,dilation=dilation, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_c, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=padding,dilation=dilation, bias=False)
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
                dil_pt = (1, 1)
                pad_pt = (1, 1)
            else:
                dil_pt = (d_math, 1)
                pad_pt = (d_math, 1)
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
        
        # MaxPool para bajar de 352 -> 88 (Una frecuencia por tecla)
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
        x = self.pool(x) # -> (B, C, 88, T)
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

class PedalAcousticModel(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(7, 7), padding=(3,3), bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(base_channels, base_channels)
        
        # EL TRUCO: Aplastamos el eje de frecuencias (352) a 1 solo canal global.
        self.pool = nn.AdaptiveMaxPool2d((1, None)) 
        
        self.context = nn.Sequential(
            ResidualBlock(base_channels, base_channels, dilation=(1, 1)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 2)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 4))
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res1(x)
        x = self.pool(x) # -> (B, C, 1, T)
        x = self.context(x)
        return x

class HPPNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=24, lstm_hidden=128):
        super().__init__()
        # ====================
        # RAMA DE LAS NOTAS
        # ====================
        self.acoustic_onset = AcousticModel(in_channels, base_channels)
        self.acoustic_other = AcousticModel(in_channels, base_channels)
        
        self.head_onset = FG_LSTM(base_channels, lstm_hidden)
        
        concat_dim = base_channels * 2
        self.head_frame = FG_LSTM(concat_dim, lstm_hidden)
        self.head_offset = FG_LSTM(concat_dim + 1, lstm_hidden) 
        self.head_velocity = FG_LSTM(concat_dim, lstm_hidden)

        # ====================
        # RAMA DEL PEDAL 
        # ====================
        self.pedal_acoustic = PedalAcousticModel(in_channels, base_channels)
        
        # LSTM más ligeras para el pedal (lstm_hidden // 2)
        self.pedal_head_onset = FG_LSTM(base_channels, lstm_hidden // 2)
        self.pedal_head_frame = FG_LSTM(base_channels, lstm_hidden // 2)
        self.pedal_head_offset = FG_LSTM(base_channels, lstm_hidden // 2)

    def forward(self, x):
        # --- 1. PROCESAMIENTO DE NOTAS ---
        feat_onset = self.acoustic_onset(x)
        logits_onset = self.head_onset(feat_onset)
        
        feat_onset_detached = feat_onset.detach()
        feat_other = self.acoustic_other(x)
        feat_combined = torch.cat([feat_other, feat_onset_detached], dim=1)
        
        logits_frame = self.head_frame(feat_combined)
        
        prob_frame = torch.sigmoid(logits_frame).detach() 
        prob_frame = prob_frame.permute(0, 2, 1).unsqueeze(1)
        feat_offset_in = torch.cat([feat_combined, prob_frame], dim=1)
        
        logits_offset = self.head_offset(feat_offset_in)
        logits_velocity = self.head_velocity(feat_combined)
        
        # --- 2. PROCESAMIENTO DEL PEDAL ---
        feat_pedal = self.pedal_acoustic(x)
        logits_pedal_on = self.pedal_head_onset(feat_pedal)
        logits_pedal_fr = self.pedal_head_frame(feat_pedal)
        logits_pedal_off = self.pedal_head_offset(feat_pedal)
        
        # El modelo ahora devuelve las 7 predicciones
        return logits_onset, logits_frame, logits_offset, logits_velocity, logits_pedal_on, logits_pedal_fr, logits_pedal_off
    

# ==========================================
# 3. UTILS (MÉTRICAS Y PLOTTING)
# ==========================================
def tensor_to_notes(onset_pred, frame_pred, offset_pred, velocity_pred=None, t_onset=0.35, t_frame=0.6, t_offset=0.4):
    notes = []
    for pitch in range(88):
        peaks, _ = find_peaks(onset_pred[:, pitch], height=t_onset, distance=2)
        for onset_frame in peaks:
            check_frame = min(onset_frame + 1, frame_pred.shape[0] - 1)
            if frame_pred[check_frame, pitch] < t_frame:
                continue 
                
            end_frame = onset_frame + 1
            while end_frame < frame_pred.shape[0]:
                frame_active = frame_pred[end_frame, pitch] > t_frame
                is_offset_hit = offset_pred[end_frame, pitch] > t_offset
                if frame_active and not is_offset_hit:
                    end_frame += 1
                else:
                    break
            
            if end_frame < frame_pred.shape[0] and offset_pred[end_frame, pitch] > t_offset:
                 end_frame += 1

            if end_frame - onset_frame > 2:
                onset_time = onset_frame * HOP_LENGTH / SR
                offset_time = end_frame * HOP_LENGTH / SR
                vel = 0
                if velocity_pred is not None:
                    vel_seg = velocity_pred[onset_frame:min(end_frame, onset_frame+5), pitch]
                    vel = np.mean(vel_seg) if len(vel_seg) > 0 else 0
                notes.append([onset_time, offset_time, pitch + 21, vel])
    return notes

def compute_metrics_standard(ref_notes_batch, est_notes_batch):
    total_tp, total_fp, total_fn = 0, 0, 0
    for ref_notes, est_notes in zip(ref_notes_batch, est_notes_batch):
        ref_arr = np.array(ref_notes)
        est_arr = np.array(est_notes)
        if len(ref_arr) == 0 and len(est_arr) == 0: continue
        if len(ref_arr) == 0:
            total_fp += len(est_arr); continue
        if len(est_arr) == 0:
            total_fn += len(ref_arr); continue

        ref_int, ref_p = ref_arr[:, :2], ref_arr[:, 2]
        est_int, est_p = est_arr[:, :2], est_arr[:, 2]
        
        matched = mir_eval.transcription.match_notes(
            ref_int, ref_p, est_int, est_p, onset_tolerance=0.05, offset_ratio=None
        )
        tp = len(matched)
        total_tp += tp
        total_fp += (len(est_p) - tp)
        total_fn += (len(ref_p) - tp)
        
    p = total_tp / (total_tp + total_fp + 1e-8)
    r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r

def compute_note_offset_metrics(ref_notes_batch, est_notes_batch):
    total_tp, total_fp, total_fn = 0, 0, 0
    for ref_notes, est_notes in zip(ref_notes_batch, est_notes_batch):
        ref_arr = np.array(ref_notes)
        est_arr = np.array(est_notes)
        if len(ref_arr) == 0 and len(est_arr) == 0: continue
        if len(ref_arr) == 0:
            total_fp += len(est_arr); continue
        if len(est_arr) == 0:
            total_fn += len(ref_arr); continue

        ref_int, ref_p = ref_arr[:, :2], ref_arr[:, 2]
        est_int, est_p = est_arr[:, :2], est_arr[:, 2]
        
        matched = mir_eval.transcription.match_notes(
            ref_int, ref_p, est_int, est_p, 
            onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05
        )
        tp = len(matched)
        total_tp += tp
        total_fp += (len(est_p) - tp)
        total_fn += (len(ref_p) - tp)
        
    p = total_tp / (total_tp + total_fp + 1e-8)
    r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r

def compute_note_offset_velocity_metrics(ref_notes_batch, est_notes_batch, velocity_tolerance=0.1):
    total_tp, total_fp, total_fn = 0, 0, 0
    for ref_notes, est_notes in zip(ref_notes_batch, est_notes_batch):
        ref_arr = np.asarray(ref_notes)
        est_arr = np.asarray(est_notes)
        if len(ref_arr) == 0 and len(est_arr) == 0: continue
        if len(ref_arr) == 0:
            total_fp += len(est_arr); continue
        if len(est_arr) == 0:
            total_fn += len(ref_arr); continue

        ref_intervals, ref_pitches, ref_velocities = ref_arr[:, :2], ref_arr[:, 2], ref_arr[:, 3]
        est_intervals, est_pitches, est_velocities = est_arr[:, :2], est_arr[:, 2], est_arr[:, 3]

        matched = mir_eval.transcription.match_notes(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05,
        )
        n_matched = len(matched)
        tp_velocity = sum([1 for ref_idx, est_idx in matched if abs(ref_velocities[ref_idx] - est_velocities[est_idx]) <= velocity_tolerance])

        total_tp += tp_velocity
        total_fp += (len(est_pitches) - n_matched) + (n_matched - tp_velocity)
        total_fn += (len(ref_pitches) - n_matched) + (n_matched - tp_velocity)

    p = total_tp / (total_tp + total_fp + 1e-8)
    r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r

def get_pixel_metrics(preds_list, targs_list):
    """Función auxiliar para calcular métricas de arrays concatenados"""
    p = np.concatenate(preds_list)
    t = np.concatenate(targs_list)
    f1 = f1_score(t, p, zero_division=0)
    prec = precision_score(t, p, zero_division=0)
    rec = recall_score(t, p, zero_division=0)
    return f1, prec, rec

def plot_training_history(csv_path="training_log_kaggle.csv"):
    if not os.path.exists(csv_path): return
        
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 1: return
        df = df.dropna(subset=['train_loss']) 
        
        sns.set_theme(style="whitegrid", context="paper")
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('HPPNet Optimized Metrics (Notes & Pedal)', fontsize=16)

        # Plot Loss
        if 'train_loss' in df.columns: sns.lineplot(data=df, x='epoch', y='train_loss', label='Train', ax=axes[0,0])
        if 'val_loss' in df.columns: sns.lineplot(data=df, x='epoch', y='val_loss', label='Val', ax=axes[0,0], linestyle='--')
        axes[0,0].set_title('1. Loss')

        # Plot Notes Onset
        if 'onset_f1' in df.columns:
            sns.lineplot(data=df, x='epoch', y='onset_f1', label='F1', ax=axes[0,1], color='g')
            sns.lineplot(data=df, x='epoch', y='onset_p', label='Precision', ax=axes[0,1], linestyle=':', alpha=0.6)
            sns.lineplot(data=df, x='epoch', y='onset_r', label='Recall', ax=axes[0,1], linestyle=':', alpha=0.6)
        axes[0,1].set_title('2. Notes: Onset Metrics (mir_eval)')

        # Plot Notes Frame/Offset
        if 'frame_f1' in df.columns: sns.lineplot(data=df, x='epoch', y='frame_f1', label='Frame F1', ax=axes[1,0])
        if 'offset_f1' in df.columns: sns.lineplot(data=df, x='epoch', y='offset_f1', label='Offset Pix F1', ax=axes[1,0], color='orange', linestyle='--')
        if 'note_off_f1' in df.columns: sns.lineplot(data=df, x='epoch', y='note_off_f1', label='Note Off F1', ax=axes[1,0], color='red')
        axes[1,0].set_title('3. Notes: Frame & Offset F1')

        # Plot Velocity
        if 'velocity_mse' in df.columns: sns.lineplot(data=df, x='epoch', y='velocity_mse', color='purple', ax=axes[1,1])
        axes[1,1].set_title('4. Velocity MSE')

        # Plot Pedal F1 (Frame)
        if 'pedal_fr_f1' in df.columns:
            sns.lineplot(data=df, x='epoch', y='pedal_fr_f1', label='Pedal Frame F1', ax=axes[2,0], color='magenta')
            sns.lineplot(data=df, x='epoch', y='pedal_fr_p', label='Pedal Frame P', ax=axes[2,0], linestyle=':', alpha=0.6, color='magenta')
            sns.lineplot(data=df, x='epoch', y='pedal_fr_r', label='Pedal Frame R', ax=axes[2,0], linestyle='-.', alpha=0.6, color='magenta')
        axes[2,0].set_title('5. Pedal: Frame Metrics')

        # Plot Pedal Onset/Offset F1
        if 'pedal_on_f1' in df.columns:
            sns.lineplot(data=df, x='epoch', y='pedal_on_f1', label='Pedal Onset F1', ax=axes[2,1], color='green')
        if 'pedal_off_f1' in df.columns:
            sns.lineplot(data=df, x='epoch', y='pedal_off_f1', label='Pedal Offset F1', ax=axes[2,1], color='red')
        axes[2,1].set_title('6. Pedal: Onset & Offset F1')

        plt.tight_layout()
        plt.savefig("training_results_kaggle.png", dpi=200)
        plt.close(fig) 
    except Exception as e:
        print(f"⚠️ Error generando gráficas: {e}")

# ==========================================
# 4. MAIN (ENTRENAMIENTO COMPLETO)
# ==========================================
if __name__ == "__main__":
    print(f"\n🚀 HPPNET-SP KAGGLE Training + PEDAL ({DEVICE})")
    
    train_ds = PianoDataset(DATA_PATH, split='train')
    val_ds = PianoDataset(DATA_PATH, split='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = HPPNet(in_channels=1, lstm_hidden=128).to(DEVICE)  
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Cargando pesos desde {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
        # strict=False es crítico aquí: cargará los pesos de las notas, y creará desde cero los del pedal
        model.load_state_dict(state_dict, strict=False)
        print("✅ Pesos cargados.")

    if torch.cuda.device_count() > 1:
        print(f"🔥 Usando {torch.cuda.device_count()} GPUs en DataParallel")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=FACTOR_LR, patience=PATIENCE_LR)
    scaler = torch.amp.GradScaler('cuda') 
    
    crit_onset = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)
    crit_frame = ComboLoss(bce_weight=2.0).to(DEVICE)
    crit_offset = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)
    crit_vel = nn.MSELoss(reduction='none')

    # CABECERA DE 29 COLUMNAS
    log_file = open("training_log_kaggle.csv", "w")
    header = "epoch,train_loss,val_loss,onset_f1,onset_p,onset_r,frame_f1,frame_p,frame_r,offset_f1,offset_p,offset_r,note_off_f1,note_off_p,note_off_r,note_off_vel_f1,note_off_vel_p,note_off_vel_r,velocity_mse,pedal_on_f1,pedal_on_p,pedal_on_r,pedal_fr_f1,pedal_fr_p,pedal_fr_r,pedal_off_f1,pedal_off_p,pedal_off_r,lr\n"
    log_file.write(header)
    log_file.flush()
    
    best_frame_f1 = 0.0

    try:
        for epoch in range(FINAL_EPOCHS):
            model.train()
            t_loss = 0
            
            with tqdm(train_loader, desc=f"Ep {epoch+1}/{FINAL_EPOCHS}", leave=False) as bar:
                for batch in bar:
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        # Extraemos las 7 salidas
                        p_on, p_fr, p_off, p_vel, ped_on, ped_fr, ped_off = model(hcqt)
                        
                        # Loss Notas
                        l_on = crit_onset(p_on, targets['onset'])
                        l_fr = crit_frame(p_fr, targets['frame'])
                        l_off = crit_offset(p_off, targets['offset'])
                        mask = targets['frame']
                        l_vel = (crit_vel(torch.sigmoid(p_vel), targets['velocity']) * mask).sum() / (mask.sum() + 1e-6)
                        
                        # Loss Pedal
                        l_ped_on = crit_onset(ped_on, targets['pedal_onset'])
                        l_ped_fr = crit_frame(ped_fr, targets['pedal_frame'])
                        l_ped_off = crit_offset(ped_off, targets['pedal_offset'])
                        
                        # Suma de Pérdidas
                        loss = (1.0 * l_on) + (5.0 * l_fr) + (5.0 * l_off) + l_vel + (1.0 * l_ped_on) + (5.0 * l_ped_fr) + (5.0 * l_ped_off)
                        
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    t_loss += loss.item()
                    bar.set_postfix(loss=loss.item())
            
            avg_t_loss = t_loss / len(train_loader)
            should_validate = ((epoch + 1) % 3 == 0) or ((epoch + 1) == FINAL_EPOCHS)

            if should_validate:
                model.eval()
                v_loss = 0
                
                # Colecciones Notas
                ref_all, est_all = [], []
                fr_preds, fr_targs, off_preds, off_targs = [], [], [], []
                vel_accum, vel_count = 0, 0
                
                # Colecciones Pedal
                ped_on_preds, ped_on_targs = [], [] 
                ped_fr_preds, ped_fr_targs = [], [] 
                ped_off_preds, ped_off_targs = [], [] 
                
                with torch.no_grad():
                    for batch in val_loader:
                        hcqt = batch['hcqt'].to(DEVICE)
                        targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                        p_on, p_fr, p_off, p_vel, ped_on, ped_fr, ped_off = model(hcqt)
                        
                        v_loss += loss.item() 
                        
                        pr_on, pr_fr, pr_off, pr_vel = torch.sigmoid(p_on), torch.sigmoid(p_fr), torch.sigmoid(p_off), torch.sigmoid(p_vel)
                        pr_ped_on, pr_ped_fr, pr_ped_off = torch.sigmoid(ped_on), torch.sigmoid(ped_fr), torch.sigmoid(ped_off)
                        
                        # Recolectar píxeles de Notas
                        fr_preds.append((pr_fr > THRESHOLD_FRAME).cpu().numpy().flatten())
                        fr_targs.append((targets['frame'] > 0.5).cpu().numpy().flatten())
                        off_preds.append((pr_off > THRESHOLD_OFFSET).cpu().numpy().flatten())
                        off_targs.append((targets['offset'] > 0.5).cpu().numpy().flatten())
                        
                        # Velocity MSE
                        v_p = pr_vel.cpu().numpy().flatten()
                        v_t = targets['velocity'].cpu().numpy().flatten()
                        m = targets['frame'].cpu().numpy().flatten().astype(bool)
                        if m.sum() > 0:
                            vel_accum += mean_squared_error(v_t[m], v_p[m]) * m.sum()
                            vel_count += m.sum()

                        # Recolectar píxeles del Pedal
                        ped_on_preds.append((pr_ped_on > THRESHOLD_ONSET).cpu().numpy().flatten())
                        ped_on_targs.append((targets['pedal_onset'] > 0.5).cpu().numpy().flatten())
                        
                        ped_fr_preds.append((pr_ped_fr > THRESHOLD_FRAME).cpu().numpy().flatten())
                        ped_fr_targs.append((targets['pedal_frame'] > 0.5).cpu().numpy().flatten())
                        
                        ped_off_preds.append((pr_ped_off > THRESHOLD_OFFSET).cpu().numpy().flatten())
                        ped_off_targs.append((targets['pedal_offset'] > 0.5).cpu().numpy().flatten())

                        # Construir eventos MIDI (Notas) para mir_eval
                        for i in range(len(hcqt)):
                            v_map = pr_vel[i].cpu().numpy()
                            est = tensor_to_notes(pr_on[i].cpu().numpy(), pr_fr[i].cpu().numpy(), pr_off[i].cpu().numpy(), v_map, t_onset=THRESHOLD_ONSET, t_frame=THRESHOLD_FRAME, t_offset=THRESHOLD_OFFSET)
                            
                            ref = []
                            ref_on = targets['onset'][i].cpu().numpy()
                            ref_fr = targets['frame'][i].cpu().numpy()
                            ref_vel = targets['velocity'][i].cpu().numpy()
                            
                            for pitch in range(88):
                                ons = np.where(ref_on[:, pitch] > 0.5)[0]
                                for o in ons:
                                    e = o + 1
                                    while e < ref_fr.shape[0] and ref_fr[e, pitch] > 0.5: e += 1
                                    if e - o > 1: 
                                        vel_val = ref_vel[o, pitch]   
                                        ref.append([o*HOP_LENGTH/SR, e*HOP_LENGTH/SR, pitch+21, vel_val])
                            est_all.append(est)
                            ref_all.append(ref)

                avg_v_loss = v_loss / len(val_loader)
                
                # --- MÉTRICAS NOTAS ---
                onset_f1, onset_p, onset_r = compute_metrics_standard(ref_all, est_all)
                note_off_f1, note_off_p, note_off_r = compute_note_offset_metrics(ref_all, est_all)
                note_off_vel_f1, note_off_vel_p, note_off_vel_r = compute_note_offset_velocity_metrics(ref_all, est_all, velocity_tolerance=0.1)
                
                frame_f1, frame_p, frame_r = get_pixel_metrics(fr_preds, fr_targs)
                offset_f1, offset_p, offset_r = get_pixel_metrics(off_preds, off_targs)
                vel_mse = vel_accum / (vel_count + 1e-8)

                # --- MÉTRICAS PEDAL ---
                pedal_on_f1, pedal_on_p, pedal_on_r = get_pixel_metrics(ped_on_preds, ped_on_targs)
                pedal_fr_f1, pedal_fr_p, pedal_fr_r = get_pixel_metrics(ped_fr_preds, ped_fr_targs)
                pedal_off_f1, pedal_off_p, pedal_off_r = get_pixel_metrics(ped_off_preds, ped_off_targs)

                curr_lr = optimizer.param_groups[0]['lr']

                # --- IMPRESIÓN LIMPIA EN CONSOLA ---
                print("-" * 80)
                print(f"🏁 Epoch {epoch+1} Results: Train Loss={avg_t_loss:.4f} | Val Loss={avg_v_loss:.4f}")
                print(f"🎹 NOTAS:")
                print(f"   ➤ Onset      : F1={onset_f1:.4f} | P={onset_p:.4f} | R={onset_r:.4f}")
                print(f"   ➤ Frame      : F1={frame_f1:.4f} | P={frame_p:.4f} | R={frame_r:.4f}")
                print(f"   ➤ Offset     : F1={offset_f1:.4f} | P={offset_p:.4f} | R={offset_r:.4f}")
                print(f"   ➤ Note-Off   : F1={note_off_f1:.4f} | P={note_off_p:.4f} | R={note_off_r:.4f}")
                print(f"   ➤ N-Off+Vel  : F1={note_off_vel_f1:.4f} | P={note_off_vel_p:.4f} | R={note_off_vel_r:.4f}")
                print(f"   ➤ Vel MSE    : {vel_mse:.4f}")
                print(f"🦶 PEDAL:")
                print(f"   ➤ Onset      : F1={pedal_on_f1:.4f} | P={pedal_on_p:.4f} | R={pedal_on_r:.4f}")
                print(f"   ➤ Frame      : F1={pedal_fr_f1:.4f} | P={pedal_fr_p:.4f} | R={pedal_fr_r:.4f}")
                print(f"   ➤ Offset     : F1={pedal_off_f1:.4f} | P={pedal_off_p:.4f} | R={pedal_off_r:.4f}")
                print(f"🧠 LR: {curr_lr:.2e}")
                print("-" * 80)
                
                # --- GUARDADO CSV ---
                log_line = f"{epoch+1},{avg_t_loss},{avg_v_loss},{onset_f1},{onset_p},{onset_r},{frame_f1},{frame_p},{frame_r},{offset_f1},{offset_p},{offset_r},{note_off_f1},{note_off_p},{note_off_r},{note_off_vel_f1},{note_off_vel_p},{note_off_vel_r},{vel_mse},{pedal_on_f1},{pedal_on_p},{pedal_on_r},{pedal_fr_f1},{pedal_fr_p},{pedal_fr_r},{pedal_off_f1},{pedal_off_p},{pedal_off_r},{curr_lr}\n"
                
                scheduler.step(frame_f1) # Reducir el LR si el Frame de las notas se estanca
                
                if frame_f1 > best_frame_f1:
                    best_frame_f1 = frame_f1
                    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save(state_dict, "best_hppnet_phase3_pedal.pth")
                    print(f"   💾 Nuevo Récord! Modelo guardado (Frame F1: {best_frame_f1:.4f})")

                log_file.write(log_line)
                log_file.flush()
                plot_training_history("training_log_kaggle.csv")

            else:
                print(f"⏩ Epoch {epoch+1}: Train Loss={avg_t_loss:.4f} (Validación saltada)")
                # Añadir las 26 comas vacías para rellenar las columnas y no romper pandas
                log_line = f"{epoch+1},{avg_t_loss},,,,,,,,,,,,,,,,,,,,,,,,,,,{optimizer.param_groups[0]['lr']}\n"
                log_file.write(log_line)
                log_file.flush()
                
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, "latest_checkpoint.pth")
    
    except KeyboardInterrupt:
            print("\n🛑 Entrenamiento detenido.")
            
    finally:
        log_file.close()
        plot_training_history("training_log_kaggle.csv")