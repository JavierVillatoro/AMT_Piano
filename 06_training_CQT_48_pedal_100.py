# Antifugas 2 
#!pip install mir_eval
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
from scipy.signal import find_peaks
import gc

# ==========================================
# 0. CONFIGURACIÓN Y ESTÁNDARES
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Usando dispositivo: {DEVICE}")

SEED = 42
SR = 16000           
HOP_LENGTH = 320    
SEGMENT_FRAMES = 512   
BINS_PER_OCTAVE = 48  
INPUT_BINS = 352

DATA_PATH = Path("/kaggle/input/datasets/javivillatoro/amt-maestro-split-full/processed_data_cqt_pedal_100")
CHECKPOINT_PATH = Path("/kaggle/working/latest_checkpoint.pth") 

BATCH_SIZE = 16           
FINAL_EPOCHS = 50         
LEARNING_RATE = 0.0003    
PATIENCE_LR = 1           
FACTOR_LR = 0.6           
NUM_WORKERS = 2           

# Umbrales Notas
THRESHOLD_ONSET = 0.35    
THRESHOLD_FRAME = 0.35     
THRESHOLD_OFFSET = 0.40

# Umbrales Pedal
TH_PED_ON = 0.30
TH_PED_FR = 0.40
TH_PED_OFF = 0.45

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ==========================================
# 1. DATASET (MODIFICADO ANTI-FUGAS DE MEMORIA)
# ==========================================
class PianoDataset(Dataset):
    def __init__(self, processed_dir, split='train'):
        self.processed_dir = Path(processed_dir) / split 
        p = self.processed_dir / "inputs_hcqt"
        if not p.exists(): raise RuntimeError(f"❌ Ruta no existe: {p}")
        
        self.files = sorted(list(p.glob("*.npy")))
        if len(self.files) == 0: raise RuntimeError(f"❌ No se encontraron archivos .npy en {p}")
        
        random.Random(SEED).shuffle(self.files)
        
        self.segments = []
        print(f"   Calculando segmentos para {split}...")
        for idx, f in enumerate(self.files):
            try:
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
            
            # 🛡️ FUNCIÓN INTERNA PARA CARGAR Y DESTRUIR MMAP
            def load_slice(folder):
                path = base / folder / fid
                m = np.load(path, mmap_mode='r')
                chunk = np.array(m[start:end]) # Copia a RAM
                if hasattr(m, '_mmap'):
                    m._mmap.close() # Cierra el archivo del OS
                del m # Destruye el objeto
                return chunk

            hcqt = load_slice("inputs_hcqt")
            if hcqt.shape[1] != INPUT_BINS:
                raise ValueError(f"Bad shape in {fid}: {hcqt.shape}")
            
            onset = load_slice("targets_onset")
            frame = load_slice("targets_frame")
            offset = load_slice("targets_offset")
            vel = load_slice("targets_velocity")
            
            ped_onset = load_slice("targets_pedal_onset")
            ped_offset = load_slice("targets_pedal_offset")
            ped_frame = load_slice("targets_pedal_frame")
            
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
            zp = torch.zeros(SEGMENT_FRAMES, 1) 
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
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).float().view(-1)
        targets = targets.float().view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class ComboLoss(nn.Module):
    def __init__(self, bce_weight=5.0):
        super(ComboLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_weight]))
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
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
                dil_pt, pad_pt = (1, 1), (1, 1)
            else:
                dil_pt, pad_pt = (d_math, 1), (d_math, 1)
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
        x = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        self.lstm.flatten_parameters()
        # 🛡️ FIX DE CLAUDE: Forzar float32 en LSTM para evitar explosión matemática (NaN)
        with torch.amp.autocast('cuda', enabled=False):
            x_f32 = x.float()
            output, _ = self.lstm(x_f32)
        #output, _ = self.lstm(x)
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
        
        prob_frame = torch.sigmoid(logits_frame).detach() 
        prob_frame = prob_frame.permute(0, 2, 1).unsqueeze(1)
        feat_offset_in = torch.cat([feat_combined, prob_frame], dim=1)
        
        logits_offset = self.head_offset(feat_offset_in)
        logits_velocity = self.head_velocity(feat_combined)
        
        feat_pedal = self.pedal_acoustic(x)
        logits_pedal_on = self.pedal_head_onset(feat_pedal)
        logits_pedal_fr = self.pedal_head_frame(feat_pedal)
        logits_pedal_off = self.pedal_head_offset(feat_pedal)
        
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

def calc_tp_fp_fn_onset(ref_notes, est_notes):
    ref_arr, est_arr = np.array(ref_notes), np.array(est_notes)
    if len(ref_arr) == 0 and len(est_arr) == 0: return 0, 0, 0
    if len(ref_arr) == 0: return 0, len(est_arr), 0
    if len(est_arr) == 0: return 0, 0, len(ref_arr)
    
    matched = mir_eval.transcription.match_notes(
        ref_arr[:, :2], ref_arr[:, 2], est_arr[:, :2], est_arr[:, 2], 
        onset_tolerance=0.05, offset_ratio=None
    )
    tp = len(matched)
    return tp, len(est_arr) - tp, len(ref_arr) - tp

def calc_tp_fp_fn_offset(ref_notes, est_notes):
    ref_arr, est_arr = np.array(ref_notes), np.array(est_notes)
    if len(ref_arr) == 0 and len(est_arr) == 0: return 0, 0, 0
    if len(ref_arr) == 0: return 0, len(est_arr), 0
    if len(est_arr) == 0: return 0, 0, len(ref_arr)

    matched = mir_eval.transcription.match_notes(
        ref_arr[:, :2], ref_arr[:, 2], est_arr[:, :2], est_arr[:, 2], 
        onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05
    )
    tp = len(matched)
    return tp, len(est_arr) - tp, len(ref_arr) - tp

def calc_tp_fp_fn_velocity(ref_notes, est_notes, velocity_tolerance=0.1):
    ref_arr, est_arr = np.array(ref_notes), np.array(est_notes)
    if len(ref_arr) == 0 and len(est_arr) == 0: return 0, 0, 0
    if len(ref_arr) == 0: return 0, len(est_arr), 0
    if len(est_arr) == 0: return 0, 0, len(ref_arr)

    matched = mir_eval.transcription.match_notes(
        ref_arr[:, :2], ref_arr[:, 2], est_arr[:, :2], est_arr[:, 2],
        onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05,
    )
    n_matched = len(matched)
    tp_vel = sum([1 for r_idx, e_idx in matched if abs(ref_arr[r_idx, 3] - est_arr[e_idx, 3]) <= velocity_tolerance])
    
    fp = (len(est_arr) - n_matched) + (n_matched - tp_vel)
    fn = (len(ref_arr) - n_matched) + (n_matched - tp_vel)
    return tp_vel, fp, fn

def plot_training_history(csv_path="training_log_kaggle.csv"):
    if not os.path.exists(csv_path): return
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 1: return
        df = df.dropna(subset=['train_loss']) 
        
        sns.set_theme(style="whitegrid", context="paper")
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('HPPNet Optimized Metrics (Notes & Pedal)', fontsize=16)

        if 'train_loss' in df.columns: sns.lineplot(data=df, x='epoch', y='train_loss', label='Train', ax=axes[0,0])
        if 'val_loss' in df.columns: sns.lineplot(data=df, x='epoch', y='val_loss', label='Val', ax=axes[0,0], linestyle='--')
        axes[0,0].set_title('1. Loss')

        if 'onset_f1' in df.columns:
            sns.lineplot(data=df, x='epoch', y='onset_f1', label='F1', ax=axes[0,1], color='g')
            sns.lineplot(data=df, x='epoch', y='onset_p', label='Precision', ax=axes[0,1], linestyle=':', alpha=0.6)
            sns.lineplot(data=df, x='epoch', y='onset_r', label='Recall', ax=axes[0,1], linestyle=':', alpha=0.6)
        axes[0,1].set_title('2. Notes: Onset Metrics (mir_eval)')

        if 'frame_f1' in df.columns: sns.lineplot(data=df, x='epoch', y='frame_f1', label='Frame F1', ax=axes[1,0])
        if 'offset_f1' in df.columns: sns.lineplot(data=df, x='epoch', y='offset_f1', label='Offset Pix F1', ax=axes[1,0], color='orange', linestyle='--')
        if 'note_off_f1' in df.columns: sns.lineplot(data=df, x='epoch', y='note_off_f1', label='Note Off F1', ax=axes[1,0], color='red')
        axes[1,0].set_title('3. Notes: Frame & Offset F1')

        if 'velocity_mse' in df.columns: sns.lineplot(data=df, x='epoch', y='velocity_mse', color='purple', ax=axes[1,1])
        axes[1,1].set_title('4. Velocity MSE')

        if 'pedal_fr_f1' in df.columns:
            sns.lineplot(data=df, x='epoch', y='pedal_fr_f1', label='Pedal Frame F1', ax=axes[2,0], color='magenta')
            sns.lineplot(data=df, x='epoch', y='pedal_fr_p', label='Pedal Frame P', ax=axes[2,0], linestyle=':', alpha=0.6, color='magenta')
            sns.lineplot(data=df, x='epoch', y='pedal_fr_r', label='Pedal Frame R', ax=axes[2,0], linestyle='-.', alpha=0.6, color='magenta')
        axes[2,0].set_title('5. Pedal: Frame Metrics')

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
# 4. MAIN (ENTRENAMIENTO COMPLETO STREAMING)
# ==========================================
if __name__ == "__main__":
    print(f"\n🚀 HPPNET-SP KAGGLE Training + PEDAL ({DEVICE})")
    
    train_ds = PianoDataset(DATA_PATH, split='train')
    val_ds = PianoDataset(DATA_PATH, split='validation') 
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False) #Pin True
    
    model = HPPNet(in_channels=1, lstm_hidden=128).to(DEVICE)  
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Cargando pesos desde {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
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
                for batch_idx, batch in enumerate(bar):
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    
                    # 🕵️‍♂️ FORENSE 1: Chequeo de Entrada
                    if torch.isnan(hcqt).any() or torch.isinf(hcqt).any():
                        print(f"\n🚨 [FORENSE] El Audio de entrada en el lote {batch_idx} contiene NaN o Inf.")
                        continue
                    
                    optimizer.zero_grad()
                    
                    with torch.amp.autocast('cuda'):
                        p_on, p_fr, p_off, p_vel, ped_on, ped_fr, ped_off = model(hcqt)

                    # 🕵️‍♂️ FORENSE 2: Chequeo post-red (Explosión en LSTM / FP16)
                    if torch.isnan(p_fr).any():
                        print(f"\n🚨 [FORENSE] La salida del modelo (LSTMs) se volvió NaN. Posible desbordamiento de gradientes FP16.")
                        optimizer.zero_grad()
                        # 👇 NUEVO: Destruimos las variables y limpiamos la gráfica
                        del p_on, p_fr, p_off, p_vel, ped_on, ped_fr, ped_off
                        torch.cuda.empty_cache()
                        continue

                    p_on = p_on.float(); p_fr = p_fr.float(); p_off = p_off.float(); p_vel = p_vel.float()
                    ped_on = ped_on.float(); ped_fr = ped_fr.float(); ped_off = ped_off.float()
                    
                    l_on = crit_onset(p_on, targets['onset'])
                    l_fr = crit_frame(p_fr, targets['frame'])
                    l_off = crit_offset(p_off, targets['offset'])
                    mask = targets['frame'].float()
                    l_vel = (crit_vel(torch.sigmoid(p_vel), targets['velocity'].float()) * mask).sum() / (mask.sum() + 1e-6)
                    
                    l_ped_on = crit_onset(ped_on, targets['pedal_onset'])
                    l_ped_fr = crit_frame(ped_fr, targets['pedal_frame'])
                    l_ped_off = crit_offset(ped_off, targets['pedal_offset'])
                    
                    loss = (1.0 * l_on) + (5.0 * l_fr) + (5.0 * l_off) + l_vel + (1.0 * l_ped_on) + (5.0 * l_ped_fr) + (5.0 * l_ped_off)
                    
                    # 🕵️‍♂️ FORENSE 3: Disección del Loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("\n" + "="*50)
                        print(f"💀 MANZANA ENVENENADA DETECTADA (Lote {batch_idx})")
                        print("DIAGNÓSTICO DEL LOSS:")
                        print(f" - Onset Notas Loss : {l_on.item()}")
                        print(f" - Frame Notas Loss : {l_fr.item()}")
                        print(f" - Offset Notas Loss: {l_off.item()}")
                        print(f" - Velocity Loss    : {l_vel.item()}")
                        print(f" - Pedal Onset Loss : {l_ped_on.item()}")
                        print(f" - Pedal Frame Loss : {l_ped_fr.item()}")
                        print(f" - Pedal Offset Loss: {l_ped_off.item()}")
                        print("="*50)
                        optimizer.zero_grad()
                        # 👇 NUEVO: Destruimos el grafo y el loss
                        del p_on, p_fr, p_off, p_vel, ped_on, ped_fr, ped_off, loss
                        torch.cuda.empty_cache()
                        continue 

                    # Actualización de pesos
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    
                    # El clipping (recorte de gradientes) evita explosiones futuras
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
                
                vel_accum, vel_count = 0, 0
                
                # 🛡️ CONTADORES INCREMENTALES GLOBALES (Ocupan 0 memoria)
                metric_fr = np.zeros(3); metric_off = np.zeros(3)
                metric_ped_on = np.zeros(3); metric_ped_fr = np.zeros(3); metric_ped_off = np.zeros(3)
                
                metric_note_on = np.zeros(3)  # [TP, FP, FN] para Onset
                metric_note_off = np.zeros(3) # [TP, FP, FN] para Offset
                metric_note_vel = np.zeros(3) # [TP, FP, FN] para Velocity

                def update_metrics(preds, targs, metric_array):
                    p = preds.bool(); t = targs.bool()
                    metric_array[0] += (p & t).sum().item()
                    metric_array[1] += (p & ~t).sum().item()
                    metric_array[2] += (~p & t).sum().item()

                with torch.no_grad():
                    for batch in val_loader:
                        hcqt = batch['hcqt'].to(DEVICE)
                        targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                        p_on, p_fr, p_off, p_vel, ped_on, ped_fr, ped_off = model(hcqt)
                        
                        l_on = crit_onset(p_on, targets['onset'])
                        l_fr = crit_frame(p_fr, targets['frame'])
                        l_off = crit_offset(p_off, targets['offset'])
                        mask = targets['frame'].float()
                        l_vel = (crit_vel(torch.sigmoid(p_vel), targets['velocity'].float()) * mask).sum() / (mask.sum() + 1e-4)
                        l_ped_on = crit_onset(ped_on, targets['pedal_onset'])
                        l_ped_fr = crit_frame(ped_fr, targets['pedal_frame'])
                        l_ped_off = crit_offset(ped_off, targets['pedal_offset'])
                        
                        val_batch_loss = (1.0 * l_on) + (5.0 * l_fr) + (5.0 * l_off) + l_vel + (1.0 * l_ped_on) + (5.0 * l_ped_fr) + (5.0 * l_ped_off)
                        v_loss += val_batch_loss.item() 
                        
                        pr_on, pr_fr, pr_off, pr_vel = torch.sigmoid(p_on), torch.sigmoid(p_fr), torch.sigmoid(p_off), torch.sigmoid(p_vel)
                        pr_ped_on, pr_ped_fr, pr_ped_off = torch.sigmoid(ped_on), torch.sigmoid(ped_fr), torch.sigmoid(ped_off)
                        
                        update_metrics(pr_fr > THRESHOLD_FRAME, targets['frame'] > 0.5, metric_fr)
                        update_metrics(pr_off > THRESHOLD_OFFSET, targets['offset'] > 0.5, metric_off)
                        update_metrics(pr_ped_on > TH_PED_ON, targets['pedal_onset'] > 0.5, metric_ped_on)
                        update_metrics(pr_ped_fr > TH_PED_FR, targets['pedal_frame'] > 0.5, metric_ped_fr)
                        update_metrics(pr_ped_off > TH_PED_OFF, targets['pedal_offset'] > 0.5, metric_ped_off)
                        
                        v_p = pr_vel.cpu().numpy().flatten()
                        v_t = targets['velocity'].cpu().numpy().flatten()
                        m = targets['frame'].cpu().numpy().flatten().astype(bool)
                        if m.sum() > 0:
                            vel_accum += mean_squared_error(v_t[m], v_p[m]) * m.sum()
                            vel_count += m.sum()

                        # 🛡️ EVALUACIÓN AL VUELO DE NOTAS (mir_eval)
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
                                        ref.append([o*HOP_LENGTH/SR, e*HOP_LENGTH/SR, pitch+21, ref_vel[o, pitch]])
                            
                            # Calcular métricas de este segmento y sumar a globales
                            tp, fp, fn = calc_tp_fp_fn_onset(ref, est)
                            metric_note_on += [tp, fp, fn]
                            
                            tp, fp, fn = calc_tp_fp_fn_offset(ref, est)
                            metric_note_off += [tp, fp, fn]
                            
                            tp, fp, fn = calc_tp_fp_fn_velocity(ref, est, 0.1)
                            metric_note_vel += [tp, fp, fn]
                            
                            # 🛡️ LIMPIEZA EXPLÍCITA DE VARIABLES PESADAS
                            del est, ref, ref_on, ref_fr, ref_vel, v_map
                        
                        # Limpieza del lote general
                        del hcqt, targets, p_on, p_fr, p_off, p_vel, ped_on, ped_fr, ped_off
                        del pr_on, pr_fr, pr_off, pr_vel, pr_ped_on, pr_ped_fr, pr_ped_off, v_p, v_t, m

                avg_v_loss = v_loss / len(val_loader)
                vel_mse = vel_accum / (vel_count + 1e-8)

                def calc_f1(metric_array):
                    tp, fp, fn = metric_array
                    p = tp / (tp + fp + 1e-8)
                    r = tp / (tp + fn + 1e-8)
                    f1 = 2 * p * r / (p + r + 1e-8)
                    return f1, p, r

                # Calcular F1 finales a partir de los contadores
                onset_f1, onset_p, onset_r = calc_f1(metric_note_on)
                note_off_f1, note_off_p, note_off_r = calc_f1(metric_note_off)
                note_off_vel_f1, note_off_vel_p, note_off_vel_r = calc_f1(metric_note_vel)

                frame_f1, frame_p, frame_r = calc_f1(metric_fr)
                offset_f1, offset_p, offset_r = calc_f1(metric_off)
                pedal_on_f1, pedal_on_p, pedal_on_r = calc_f1(metric_ped_on)
                pedal_fr_f1, pedal_fr_p, pedal_fr_r = calc_f1(metric_ped_fr)
                pedal_off_f1, pedal_off_p, pedal_off_r = calc_f1(metric_ped_off)

                curr_lr = optimizer.param_groups[0]['lr']

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
                
                log_line = f"{epoch+1},{avg_t_loss},{avg_v_loss},{onset_f1},{onset_p},{onset_r},{frame_f1},{frame_p},{frame_r},{offset_f1},{offset_p},{offset_r},{note_off_f1},{note_off_p},{note_off_r},{note_off_vel_f1},{note_off_vel_p},{note_off_vel_r},{vel_mse},{pedal_on_f1},{pedal_on_p},{pedal_on_r},{pedal_fr_f1},{pedal_fr_p},{pedal_fr_r},{pedal_off_f1},{pedal_off_p},{pedal_off_r},{curr_lr}\n"
                
                scheduler.step(frame_f1)
                
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
                log_line = f"{epoch+1},{avg_t_loss},,,,,,,,,,,,,,,,,,,,,,,,,,,{optimizer.param_groups[0]['lr']}\n"
                log_file.write(log_line)
                log_file.flush()
                
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, "latest_checkpoint.pth")

            # =======================================================
            # 🛡️ LIMPIEZA TOTAL DE MEMORIA AL FINAL DE CADA ÉPOCA
            # =======================================================
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
            print("\n🛑 Entrenamiento detenido.")
            
    finally:
        log_file.close()
        plot_training_history("training_log_kaggle.csv")