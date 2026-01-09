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
from scipy.signal import find_peaks

# ==========================================
# 0. CONFIGURACIÓN Y ESTÁNDARES
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Usando dispositivo: {DEVICE}")

SEED = 42
SR = 16000           
HOP_LENGTH = 512     
SEGMENT_FRAMES = 320 
BINS_PER_OCTAVE = 12  
# Asegúrate de que esta ruta es correcta
DATA_PATH = Path("processed_data_HPPNET_4_h_pedal_5") 

# Hyperparámetros
BATCH_SIZE = 32           
FINAL_EPOCHS = 50        
LEARNING_RATE = 0.0006    
PATIENCE_LR = 2           # <--- BAJADO A 2
FACTOR_LR = 0.5           
NUM_WORKERS = 4           

# Umbrales
THRESHOLD_ONSET = 0.35    
THRESHOLD_FRAME = 0.5     
THRESHOLD_OFFSET = 0.3    
THRESHOLD_PEDAL = 0.5     

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
        base = self.processed_dir
        
        try:
            # Carga Notas
            hcqt = np.load(base / "inputs_hcqt" / fid, mmap_mode='r')[start:end] 
            onset = np.load(base / "targets_onset" / fid, mmap_mode='r')[start:end]
            frame = np.load(base / "targets_frame" / fid, mmap_mode='r')[start:end]
            offset = np.load(base / "targets_offset" / fid, mmap_mode='r')[start:end]
            vel = np.load(base / "targets_velocity" / fid, mmap_mode='r')[start:end]
            
            # Carga Pedal
            p_frame = np.load(base / "targets_pedal_frame" / fid, mmap_mode='r')[start:end]
            p_onset = np.load(base / "targets_pedal_onset" / fid, mmap_mode='r')[start:end]
            p_offset = np.load(base / "targets_pedal_offset" / fid, mmap_mode='r')[start:end]
            
            curr_len = hcqt.shape[0]
            if curr_len < SEGMENT_FRAMES:
                pad = SEGMENT_FRAMES - curr_len
                hcqt = np.pad(hcqt, ((0, pad), (0,0), (0,0)))
                onset = np.pad(onset, ((0, pad), (0,0)))
                frame = np.pad(frame, ((0, pad), (0,0)))
                offset = np.pad(offset, ((0, pad), (0,0)))
                vel = np.pad(vel, ((0, pad), (0,0)))
                p_frame = np.pad(p_frame, ((0, pad), (0,0)))
                p_onset = np.pad(p_onset, ((0, pad), (0,0)))
                p_offset = np.pad(p_offset, ((0, pad), (0,0)))
            
            hcqt_t = torch.tensor(hcqt).permute(2, 1, 0).float()
            
            return {
                "hcqt": hcqt_t,
                "onset": torch.tensor(onset).float(), "frame": torch.tensor(frame).float(),
                "offset": torch.tensor(offset).float(), "velocity": torch.tensor(vel).float(),
                "pedal_frame": torch.tensor(p_frame).float(), "pedal_onset": torch.tensor(p_onset).float(),
                "pedal_offset": torch.tensor(p_offset).float()
            }
        except Exception as e:
            print(f"Error loading {fid}: {e}")
            z = torch.zeros(SEGMENT_FRAMES, 88); zp = torch.zeros(SEGMENT_FRAMES, 1)
            return {"hcqt": torch.zeros(4, 88, SEGMENT_FRAMES), 
                    "onset": z, "frame": z, "offset": z, "velocity": z,
                    "pedal_frame": zp, "pedal_onset": zp, "pedal_offset": zp}

# ==========================================
# 2. ARQUITECTURA
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0: loss = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * loss
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
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
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_c, affine=True); self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_c, affine=True)
        self.downsample = nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.InstanceNorm2d(out_c, affine=True)) if in_c != out_c else None
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + (self.downsample(x) if self.downsample else x))

class HDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        for h in [1, 2, 3, 4, 5, 6]:
            d = 1 if h == 1 else int(np.round(BINS_PER_OCTAVE * np.log2(h)))
            self.convs.append(nn.Conv2d(in_channels, out_channels, (3, 3), padding=(d, 1), dilation=(d, 1)))
        self.fusion = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.InstanceNorm2d(out_channels, affine=True), nn.ReLU())
    def forward(self, x): return self.fusion(sum([conv(x) for conv in self.convs]))

class AcousticModel(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.input_conv = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False), nn.InstanceNorm2d(base_channels, affine=True), nn.ReLU())
        self.res1 = ResidualBlock(base_channels, base_channels); self.res2 = ResidualBlock(base_channels, base_channels)
        self.hdc = HDConv(base_channels, base_channels)
        self.hdc_bn = nn.InstanceNorm2d(base_channels, affine=True); self.hdc_relu = nn.ReLU()
        self.context = nn.Sequential(ResidualBlock(base_channels, base_channels), ResidualBlock(base_channels, base_channels), ResidualBlock(base_channels, base_channels))
    def forward(self, x):
        x = self.input_conv(x); x = self.res1(x); x = self.res2(x)
        x = x + self.hdc_relu(self.hdc_bn(self.hdc(x)))
        return self.context(x)

class FG_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x):
        b, c, f, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        self.lstm.flatten_parameters()
        output = self.proj(self.lstm(x)[0]).view(b, f, t, -1)
        return output.squeeze(-1).permute(0, 2, 1) if output.shape[-1] == 1 else output.permute(0, 2, 1)

class HPPNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=24, lstm_hidden=128):
        super().__init__()
        self.acoustic_onset = AcousticModel(in_channels, base_channels)
        self.acoustic_other = AcousticModel(in_channels, base_channels)
        
        # Heads Notas
        self.head_onset = FG_LSTM(base_channels, lstm_hidden)
        self.head_frame = FG_LSTM(base_channels * 2, lstm_hidden)
        self.head_offset = FG_LSTM(base_channels * 2 + 1, lstm_hidden) 
        self.head_velocity = FG_LSTM(base_channels * 2, lstm_hidden)

        # Heads Pedal (Feature Reduction + LSTM)
        self.pedal_reduce_on = nn.Sequential(nn.Conv2d(base_channels, base_channels, (88, 1)), nn.ReLU())
        self.pedal_reduce_comb = nn.Sequential(nn.Conv2d(base_channels*2, base_channels*2, (88, 1)), nn.ReLU())
        
        self.head_pedal_onset = nn.LSTM(base_channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.head_pedal_frame = nn.LSTM(base_channels*2, lstm_hidden, batch_first=True, bidirectional=True)
        self.head_pedal_offset = nn.LSTM(base_channels*2, lstm_hidden, batch_first=True, bidirectional=True)
        
        self.fc_p_on = nn.Linear(lstm_hidden*2, 1); self.fc_p_fr = nn.Linear(lstm_hidden*2, 1); self.fc_p_off = nn.Linear(lstm_hidden*2, 1)

    def forward(self, x):
        feat_on = self.acoustic_onset(x)
        feat_oth = self.acoustic_other(x)
        feat_comb = torch.cat([feat_oth, feat_on.detach()], dim=1)
        
        # Branch Notas
        logits_on = self.head_onset(feat_on)
        logits_fr = self.head_frame(feat_comb)
        prob_fr = torch.sigmoid(logits_fr).detach().permute(0, 2, 1).unsqueeze(1)
        logits_off = self.head_offset(torch.cat([feat_comb, prob_fr], dim=1))
        logits_vel = self.head_velocity(feat_comb)
        
        # Branch Pedal
        p_on_feat = self.pedal_reduce_on(feat_on).squeeze(2).permute(0, 2, 1)
        p_comb_feat = self.pedal_reduce_comb(feat_comb).squeeze(2).permute(0, 2, 1)
        
        logits_p_on = self.fc_p_on(self.head_pedal_onset(p_on_feat)[0])
        logits_p_fr = self.fc_p_fr(self.head_pedal_frame(p_comb_feat)[0])
        logits_p_off = self.fc_p_off(self.head_pedal_offset(p_comb_feat)[0])
        
        return logits_on, logits_fr, logits_off, logits_vel, logits_p_on, logits_p_fr, logits_p_off

# ==========================================
# 3. UTILS: HIGH-RES DECODING & METRICS
# ==========================================
def refine_time(frame_idx, regression_array, hop_length=HOP_LENGTH, sr=SR):
    """ High-Resolution Refinement (Eq 10/11 del Paper) """
    if frame_idx <= 0 or frame_idx >= len(regression_array) - 1: return frame_idx * hop_length / sr
    y_A, y_B, y_C = regression_array[frame_idx-1], regression_array[frame_idx], regression_array[frame_idx+1]
    if y_B == y_A or y_B == y_C: shift = 0
    elif y_C > y_A: shift = (y_C - y_A) / (2 * (y_B - y_A))
    else: shift = (y_C - y_A) / (2 * (y_B - y_C))
    return (frame_idx + np.clip(shift, -0.5, 0.5)) * hop_length / sr

def tensor_to_notes_high_res(onset_pred, frame_pred, offset_pred, velocity_pred, t_on=0.35, t_fr=0.5, t_off=0.3):
    """ Decodificación High-Res para Notas """
    notes = []
    for pitch in range(88):
        peaks, _ = find_peaks(onset_pred[:, pitch], height=t_on, distance=2)
        for onset_frame in peaks:
            if frame_pred[min(onset_frame+1, len(frame_pred)-1), pitch] < t_fr: continue
            
            onset_time = refine_time(onset_frame, onset_pred[:, pitch]) # High-Res Start
            
            end_frame = onset_frame + 1
            while end_frame < len(frame_pred):
                if frame_pred[end_frame, pitch] < t_fr or offset_pred[end_frame, pitch] > t_off: break
                end_frame += 1
            
            if end_frame < len(frame_pred) and offset_pred[end_frame, pitch] > t_off:
                # High-Res End
                search_win = offset_pred[max(0, end_frame-2):min(len(offset_pred), end_frame+3), pitch]
                if len(search_win) > 0:
                    offset_time = refine_time(max(0, end_frame-2) + np.argmax(search_win), offset_pred[:, pitch])
                else: offset_time = end_frame * HOP_LENGTH / SR
            else: offset_time = end_frame * HOP_LENGTH / SR
            
            vel = 0
            if velocity_pred is not None:
                vel_seg = velocity_pred[onset_frame:min(end_frame, onset_frame+5), pitch]
                vel = np.mean(vel_seg) if len(vel_seg) > 0 else 0
            
            if offset_time > onset_time: notes.append([onset_time, offset_time, pitch + 21, vel])
    return notes

def tensor_to_pedal_high_res(p_onset, p_frame, p_offset, t_on=0.5, t_fr=0.5, t_off=0.5):
    """ Decodificación High-Res para Pedal (Algoritmo 2 del Paper) """
    pedals = []
    on, fr, off = p_onset[:, 0], p_frame[:, 0], p_offset[:, 0]
    peaks, _ = find_peaks(on, height=t_on, distance=10)
    for onset_idx in peaks:
        if np.mean(fr[onset_idx:min(len(fr), onset_idx+5)]) < t_fr: continue
        start_time = refine_time(onset_idx, on)
        
        end_idx = onset_idx + 1
        while end_idx < len(fr):
            if (off[end_idx] > t_off and off[end_idx] > off[end_idx-1]) or fr[end_idx] < t_fr: break
            end_idx += 1
            
        if end_idx < len(fr) and off[end_idx] > t_off: end_time = refine_time(end_idx, off)
        else: end_time = end_idx * HOP_LENGTH / SR
        
        if end_time > start_time: pedals.append([start_time, end_time, 0])
    return pedals

def compute_full_metrics(ref_notes, est_notes, ref_pedal, est_pedal, fr_p, fr_t, off_p, off_t):
    """ Calcula TODAS las métricas solicitadas """
    # 1. MIR_EVAL (Event Based)
    # Notas: Onset (50ms)
    tp, fp, fn = 0, 0, 0
    for r, e in zip(ref_notes, est_notes):
        ra, ea = np.array(r), np.array(e)
        if len(ra)==0 and len(ea)==0: continue
        if len(ra)==0: fp+=len(ea); continue
        if len(ea)==0: fn+=len(ra); continue
        m = mir_eval.transcription.match_notes(ra[:,:2], ra[:,2], ea[:,:2], ea[:,2], onset_tolerance=0.05)
        tp+=len(m); fp+=len(ea)-len(m); fn+=len(ra)-len(m)
    n_on_p, n_on_r, n_on_f1 = (tp/(tp+fp+1e-8), tp/(tp+fn+1e-8), 2*tp/(2*tp+fp+fn+1e-8))

    # Notas: Note w/ Offset (50ms)
    tp, fp, fn = 0, 0, 0
    for r, e in zip(ref_notes, est_notes):
        ra, ea = np.array(r), np.array(e)
        if len(ra)==0 and len(ea)==0: continue
        if len(ra)==0: fp+=len(ea); continue
        if len(ea)==0: fn+=len(ra); continue
        # offset_ratio=0.2, offset_min_tolerance=0.05
        m = mir_eval.transcription.match_notes(ra[:,:2], ra[:,2], ea[:,:2], ea[:,2], onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05)
        tp+=len(m); fp+=len(ea)-len(m); fn+=len(ra)-len(m)
    n_off_p, n_off_r, n_off_f1 = (tp/(tp+fp+1e-8), tp/(tp+fn+1e-8), 2*tp/(2*tp+fp+fn+1e-8))

    # Pedal (mir_eval)
    ptp, pfp, pfn = 0, 0, 0
    for r, e in zip(ref_pedal, est_pedal):
        ra, ea = np.array(r), np.array(e)
        if len(ra)==0 and len(ea)==0: continue
        if len(ra)==0: pfp+=len(ea); continue
        if len(ea)==0: pfn+=len(ra); continue
        m = mir_eval.transcription.match_notes(ra[:,:2], np.zeros(len(ra)), ea[:,:2], np.zeros(len(ea)), onset_tolerance=0.1, offset_ratio=0.2)
        ptp+=len(m); pfp+=len(ea)-len(m); pfn+=len(ra)-len(m)
    p_p, p_r, p_f1 = (ptp/(ptp+pfp+1e-8), ptp/(ptp+pfn+1e-8), 2*ptp/(2*ptp+pfp+pfn+1e-8))

    # 2. PIXEL-WISE (Frame & Offset) - SKLEARN (Igual que script antiguo)
    f_f1 = f1_score(fr_t, fr_p, zero_division=0)
    f_p = precision_score(fr_t, fr_p, zero_division=0)
    f_r = recall_score(fr_t, fr_p, zero_division=0)
    
    o_f1 = f1_score(off_t, off_p, zero_division=0)
    o_p = precision_score(off_t, off_p, zero_division=0)
    o_r = recall_score(off_t, off_p, zero_division=0)

    return n_on_f1, n_on_p, n_on_r, n_off_f1, n_off_p, n_off_r, p_f1, p_p, p_r, f_f1, f_p, f_r, o_f1, o_p, o_r

def plot_training_history(csv_path="training_log_kaggle.csv"):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    sns.lineplot(data=df, x='epoch', y='train_loss', label='Train', ax=axes[0,0])
    sns.lineplot(data=df, x='epoch', y='val_loss', label='Val', ax=axes[0,0], linestyle='--')
    axes[0,0].set_title("Loss")
    
    # Note Onset
    sns.lineplot(data=df, x='epoch', y='note_on_f1', label='F1', ax=axes[0,1], color='g')
    sns.lineplot(data=df, x='epoch', y='note_on_p', label='P', ax=axes[0,1], linestyle=':')
    sns.lineplot(data=df, x='epoch', y='note_on_r', label='R', ax=axes[0,1], linestyle=':')
    axes[0,1].set_title("Note Onset (50ms)")
    
    # Note Offset
    sns.lineplot(data=df, x='epoch', y='note_off_f1', label='F1', ax=axes[1,0], color='orange')
    sns.lineplot(data=df, x='epoch', y='note_off_p', label='P', ax=axes[1,0], linestyle=':')
    sns.lineplot(data=df, x='epoch', y='note_off_r', label='R', ax=axes[1,0], linestyle=':')
    axes[1,0].set_title("Note w/ Off (50ms)")
    
    # Pedal
    sns.lineplot(data=df, x='epoch', y='pedal_f1', label='F1', ax=axes[0,2], color='r')
    sns.lineplot(data=df, x='epoch', y='pedal_p', label='P', ax=axes[0,2], linestyle=':')
    sns.lineplot(data=df, x='epoch', y='pedal_r', label='R', ax=axes[0,2], linestyle=':')
    axes[0,2].set_title("Pedal Metrics")
    
    # Frames (Pixel)
    sns.lineplot(data=df, x='epoch', y='frame_f1', label='Frame F1', ax=axes[1,1])
    sns.lineplot(data=df, x='epoch', y='offset_f1', label='Offset F1', ax=axes[1,1], linestyle='--')
    axes[1,1].set_title("Pixel-wise (Frame/Off)")
    
    # Velocity
    sns.lineplot(data=df, x='epoch', y='velocity_mse', color='purple', ax=axes[1,2])
    axes[1,2].set_title("Velocity MSE")
    
    plt.tight_layout()
    plt.savefig("training_results_kaggle.png", dpi=150)

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print(f"\n🚀 HPPNET HIGH-RES + PEDAL (Full Metrics) ({DEVICE})")
    
    train_ds = PianoDataset(DATA_PATH, split='train')
    val_ds = PianoDataset(DATA_PATH, split='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = HPPNet(in_channels=4, lstm_hidden=128).to(DEVICE)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=FACTOR_LR, patience=PATIENCE_LR)
    scaler = torch.cuda.amp.GradScaler() 
    
    # Losses
    crit_onset = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)
    crit_frame = ComboLoss(bce_weight=8.0).to(DEVICE)
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(DEVICE))
    crit_vel = nn.MSELoss(reduction='none')
    
    crit_pedal_on = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)
    crit_pedal_fr = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))
    crit_pedal_off = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(DEVICE))

    # Logs (TODAS las columnas)
    log_file = open("training_log_kaggle.csv", "w")
    log_file.write("epoch,train_loss,val_loss,note_on_f1,note_on_p,note_on_r,note_off_f1,note_off_p,note_off_r,pedal_f1,pedal_p,pedal_r,frame_f1,frame_p,frame_r,offset_f1,offset_p,offset_r,velocity_mse,lr\n")
    log_file.flush()
    best_f1 = 0.0

    try:
        for epoch in range(FINAL_EPOCHS):
            model.train()
            t_loss = 0
            with tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False) as bar:
                for batch in bar:
                    hcqt = batch['hcqt'].to(DEVICE)
                    t_on, t_fr = batch['onset'].to(DEVICE), batch['frame'].to(DEVICE)
                    t_off, t_vel = batch['offset'].to(DEVICE), batch['velocity'].to(DEVICE)
                    tp_on, tp_fr = batch['pedal_onset'].to(DEVICE), batch['pedal_frame'].to(DEVICE)
                    tp_off = batch['pedal_offset'].to(DEVICE)
                    
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        p_on, p_fr, p_off, p_vel, pp_on, pp_fr, pp_off = model(hcqt)
                        
                        l_notes = 10*crit_onset(p_on, t_on) + crit_frame(p_fr, t_fr) + crit_offset(p_off, t_off)
                        l_vel = (crit_vel(torch.sigmoid(p_vel), t_vel)*t_fr).sum()/(t_fr.sum()+1e-6)
                        l_pedal = 10*crit_pedal_on(pp_on, tp_on) + crit_pedal_fr(pp_fr, tp_fr) + crit_pedal_off(pp_off, tp_off)
                        loss = l_notes + l_vel + l_pedal
                    
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    t_loss += loss.item(); bar.set_postfix(loss=loss.item())
            
            avg_t_loss = t_loss/len(train_loader)

            if (epoch+1)%3==0 or (epoch+1)==FINAL_EPOCHS:
                model.eval()
                v_loss, vel_mse = 0, 0
                est_n, ref_n, est_p, ref_p = [], [], [], []
                fr_p_list, fr_t_list, off_p_list, off_t_list = [], [], [], []
                vel_accum, vel_count = 0, 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        hcqt = batch['hcqt'].to(DEVICE)
                        t_on, t_fr = batch['onset'].to(DEVICE), batch['frame'].to(DEVICE)
                        t_off, t_vel = batch['offset'].to(DEVICE), batch['velocity'].to(DEVICE)
                        tp_on, tp_fr = batch['pedal_onset'].to(DEVICE), batch['pedal_frame'].to(DEVICE)
                        tp_off = batch['pedal_offset'].to(DEVICE)
                        
                        p_on, p_fr, p_off, p_vel, pp_on, pp_fr, pp_off = model(hcqt)
                        
                        # Loss validation (Approx)
                        l_notes = 10*crit_onset(p_on, t_on) + crit_frame(p_fr, t_fr) + crit_offset(p_off, t_off)
                        l_vel = (crit_vel(torch.sigmoid(p_vel), t_vel)*t_fr).sum()/(t_fr.sum()+1e-6)
                        l_pedal = 10*crit_pedal_on(pp_on, tp_on) + crit_pedal_fr(pp_fr, tp_fr) + crit_pedal_off(pp_off, tp_off)
                        v_loss += (l_notes + l_vel + l_pedal).item()
                        
                        pr_on, pr_fr, pr_off = torch.sigmoid(p_on), torch.sigmoid(p_fr), torch.sigmoid(p_off)
                        ppr_on, ppr_fr, ppr_off = torch.sigmoid(pp_on), torch.sigmoid(pp_fr), torch.sigmoid(pp_off)
                        
                        for i in range(len(hcqt)):
                            # 1. Event Decoding (High-Res)
                            est_n.append(tensor_to_notes_high_res(pr_on[i].cpu().numpy(), pr_fr[i].cpu().numpy(), pr_off[i].cpu().numpy(), torch.sigmoid(p_vel[i]).cpu().numpy()))
                            est_p.append(tensor_to_pedal_high_res(ppr_on[i].cpu().numpy(), ppr_fr[i].cpu().numpy(), ppr_off[i].cpu().numpy()))
                            
                            ref_n.append(tensor_to_notes_high_res(t_on[i].cpu().numpy(), t_fr[i].cpu().numpy(), t_off[i].cpu().numpy(), None, t_on=0.5))
                            ref_p.append(tensor_to_pedal_high_res(tp_on[i].cpu().numpy(), tp_fr[i].cpu().numpy(), tp_off[i].cpu().numpy(), t_on=0.5))
                            
                            # 2. Pixel-wise Data Collection (Restaurado para Frames y Offsets)
                            fr_p_list.append((pr_fr[i] > THRESHOLD_FRAME).cpu().numpy().flatten())
                            fr_t_list.append((t_fr[i] > 0.5).cpu().numpy().flatten())
                            off_p_list.append((pr_off[i] > THRESHOLD_OFFSET).cpu().numpy().flatten())
                            off_t_list.append((t_off[i] > 0.5).cpu().numpy().flatten())
                            
                            # Velocity MSE
                            m = (t_fr[i] > 0.5).cpu().numpy().flatten()
                            if m.sum() > 0:
                                v_p, v_t = torch.sigmoid(p_vel[i]).cpu().numpy().flatten(), t_vel[i].cpu().numpy().flatten()
                                vel_accum += mean_squared_error(v_t[m], v_p[m]) * m.sum()
                                vel_count += m.sum()

                # --- METRICS ---
                avg_v_loss = v_loss / len(val_loader)
                # Compute all (Events + Pixel-wise)
                n_on_f1, n_on_p, n_on_r, n_off_f1, n_off_p, n_off_r, pf1, pp, pr, f_f1, f_p, f_r, o_f1, o_p, o_r = compute_full_metrics(ref_n, est_n, ref_p, est_p, np.concatenate(fr_p_list), np.concatenate(fr_t_list), np.concatenate(off_p_list), np.concatenate(off_t_list))
                
                v_mse = vel_accum / (vel_count + 1e-8)
                lr = optimizer.param_groups[0]['lr']

                print("-" * 80)
                print(f"🏁 Ep {epoch+1} | Loss T:{avg_t_loss:.3f} V:{avg_v_loss:.3f} | LR: {lr:.2e}")
                print(f"   🎹 Note Onset (50ms): F1={n_on_f1:.4f} P={n_on_p:.3f} R={n_on_r:.3f}")
                print(f"   🏁 Note w/ Off(50ms): F1={n_off_f1:.4f} P={n_off_p:.3f} R={n_off_r:.3f}") # 
                print(f"   🦶 Pedal (Evt)      : F1={pf1:.4f} P={pp:.3f} R={pr:.3f}")
                print(f"   🖼️ Frame (Pix)      : F1={f_f1:.4f} P={f_p:.3f} R={f_r:.3f}")
                print(f"   📉 Offset (Pix)     : F1={o_f1:.4f} P={o_p:.3f} R={o_r:.3f}")
                print("-" * 80)
                
                log_line = f"{epoch+1},{avg_t_loss},{avg_v_loss},{n_on_f1},{n_on_p},{n_on_r},{n_off_f1},{n_off_p},{n_off_r},{pf1},{pp},{pr},{f_f1},{f_p},{f_r},{o_f1},{o_p},{o_r},{v_mse},{lr}\n"
                log_file.write(log_line); log_file.flush()
                scheduler.step(n_on_f1)
                plot_training_history()
                
                if n_on_f1 > best_f1:
                    best_f1 = n_on_f1
                    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "best_hppnet_kaggle.pth")
                    print("   💾 Model Saved!")
            else:
                print(f"⏩ Ep {epoch+1}: Loss={avg_t_loss:.4f}"); log_file.write(f"{epoch+1},{avg_t_loss},,,,,,,,,,,,,,,,,,,{optimizer.param_groups[0]['lr']}\n")
                log_file.flush()
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "latest_checkpoint.pth")
    except KeyboardInterrupt: print("STOP")
    finally: log_file.close(); plot_training_history()