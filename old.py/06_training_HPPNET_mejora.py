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

# ==========================================
# 0. CONFIGURACIÓN Y ESTÁNDARES
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SR = 16000           
HOP_LENGTH = 512     
SEGMENT_FRAMES = 320 
BINS_PER_OCTAVE = 12  
DATA_PATH = Path("processed_data_HPPNET_10") 

# Hyperparámetros de Entrenamiento
BATCH_SIZE = 4            
FINAL_EPOCHS = 50         
LEARNING_RATE = 0.0006    
PATIENCE_LR = 3           
FACTOR_LR = 0.5           

# --- CAMBIO 1: UMBRALES MÁS PERMISIVOS ---
# Bajamos Offset a 0.3 para capturar predicciones débiles al inicio
THRESHOLD_ONSET = 0.5     
THRESHOLD_FRAME = 0.5
THRESHOLD_OFFSET = 0.3  # Antes 0.5. Bajado para mejorar Recall.

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
        random.Random(SEED).shuffle(all_files)
        split_idx = int(len(all_files) * (1 - val_split))
        self.files = all_files[:split_idx] if split == 'train' else all_files[split_idx:]
        
        self.segments = []
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

    def __len__(self): return len(self.segments)

    def __getitem__(self, idx):
        file_idx, start, end = self.segments[idx]
        fid = self.files[file_idx].name
        try:
            base = self.processed_dir
            hcqt = np.load(base / "inputs_hcqt" / fid, mmap_mode='r')[start:end] 
            onset = np.load(base / "targets_onset" / fid, mmap_mode='r')[start:end]
            frame = np.load(base / "targets_frame" / fid, mmap_mode='r')[start:end]
            offset = np.load(base / "targets_offset" / fid, mmap_mode='r')[start:end]
            vel = np.load(base / "targets_velocity" / fid, mmap_mode='r')[start:end]
            
            curr_len = hcqt.shape[0]
            if curr_len < SEGMENT_FRAMES:
                pad = SEGMENT_FRAMES - curr_len
                hcqt = np.pad(hcqt, ((0, pad), (0,0), (0,0)))
                onset = np.pad(onset, ((0, pad), (0,0)))
                frame = np.pad(frame, ((0, pad), (0,0)))
                offset = np.pad(offset, ((0, pad), (0,0)))
                vel = np.pad(vel, ((0, pad), (0,0)))
            
            # (Time, Freq, Harmonics) -> (Harmonics, Freq, Time)
            hcqt_t = torch.tensor(hcqt).permute(2, 1, 0).float()
            
            return {
                "hcqt": hcqt_t,
                "onset": torch.tensor(onset).float(),
                "frame": torch.tensor(frame).float(),
                "offset": torch.tensor(offset).float(),
                "velocity": torch.tensor(vel).float()
            }
        except:
            z = torch.zeros(SEGMENT_FRAMES, 88)
            return {"hcqt": torch.zeros(3, 88, SEGMENT_FRAMES), "onset": z, "frame": z, "offset": z, "velocity": z}

# ==========================================
# 2. ARQUITECTURA HPPNET (HPPNet-sp)
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
        
        # Return (B, T, F) to match targets
        return output.permute(0, 2, 1) 

class HPPNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=24, lstm_hidden=32):
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
# 3. UTILS
# ==========================================
def tensor_to_notes(onset_pred, frame_pred, t_onset=0.5, t_frame=0.5):
    """
    Decodificación estándar basada en Onset y Frame.
    Nota: Aunque no usamos 'offset_pred' explícitamente aquí para cortar notas,
    el entrenamiento del offset es crucial para futuras implementaciones avanzadas.
    """
    notes = []
    for pitch_idx in range(88):
        onsets_locs = np.where(onset_pred[:, pitch_idx] > t_onset)[0]
        for onset_frame in onsets_locs:
            end_frame = onset_frame + 1
            while end_frame < frame_pred.shape[0] and frame_pred[end_frame, pitch_idx] > t_frame:
                end_frame += 1
            if end_frame - onset_frame > 1:
                onset_time = onset_frame * HOP_LENGTH / SR
                offset_time = end_frame * HOP_LENGTH / SR
                notes.append([onset_time, offset_time, pitch_idx + 21]) 
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

def plot_training_history(csv_path="training_log.csv"):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('HPPNet Training Metrics (Maestro)', fontsize=16)

    sns.lineplot(data=df, x='epoch', y='train_loss', label='Train', ax=axes[0,0])
    sns.lineplot(data=df, x='epoch', y='val_loss', label='Val', ax=axes[0,0], linestyle='--')
    axes[0,0].set_title('Loss')

    sns.lineplot(data=df, x='epoch', y='onset_f1', label='F1', ax=axes[0,1], color='g')
    sns.lineplot(data=df, x='epoch', y='onset_p', label='Precision', ax=axes[0,1], linestyle=':', alpha=0.6)
    sns.lineplot(data=df, x='epoch', y='onset_r', label='Recall', ax=axes[0,1], linestyle=':', alpha=0.6)
    axes[0,1].set_title('Onset Metrics')
    axes[0,1].set_ylim(0, 1)

    sns.lineplot(data=df, x='epoch', y='frame_f1', label='Frame F1', ax=axes[1,0])
    sns.lineplot(data=df, x='epoch', y='offset_f1', label='Offset F1', ax=axes[1,0], color='orange')
    axes[1,0].set_title('Frame & Offset F1')
    axes[1,0].set_ylim(0, 1)

    sns.lineplot(data=df, x='epoch', y='velocity_mse', color='purple', ax=axes[1,1])
    axes[1,1].set_title('Velocity MSE')

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=200)
    print("📊 Gráficas guardadas en training_results.png")

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    print(f"\n🚀 HPPNET-SP Training ({DEVICE})")
    
    train_ds = PianoDataset(DATA_PATH, split='train')
    val_ds = PianoDataset(DATA_PATH, split='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model = HPPNet(in_channels=3, lstm_hidden=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=FACTOR_LR, patience=PATIENCE_LR)
    scaler = torch.amp.GradScaler('cuda') 
    
    # --- CAMBIO 2: FOCAL LOSS AGRESIVA PARA OFFSET ---
    # Alpha 0.75 significa que le importa más la clase positiva (el offset)
    crit_onset = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)
    crit_offset = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE) # Antes era 0.5
    crit_frame = nn.BCEWithLogitsLoss().to(DEVICE)
    crit_vel = nn.MSELoss(reduction='none')

    log_file = open("training_log.csv", "w")
    header = "epoch,train_loss,val_loss,onset_f1,onset_p,onset_r,frame_f1,frame_p,frame_r,offset_f1,offset_p,offset_r,velocity_mse,lr\n"
    log_file.write(header)
    
    best_f1 = 0.0

    try:
        for epoch in range(FINAL_EPOCHS):
            model.train()
            t_loss = 0
            
            # --- TRAIN ---
            with tqdm(train_loader, desc=f"Ep {epoch+1}/{FINAL_EPOCHS}", leave=False) as bar:
                for batch in bar:
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        p_on, p_fr, p_off, p_vel = model(hcqt)
                        
                        l_on = crit_onset(p_on, targets['onset'])
                        l_fr = crit_frame(p_fr, targets['frame'])
                        l_off = crit_offset(p_off, targets['offset'])
                        
                        mask = targets['frame']
                        l_vel = (crit_vel(torch.sigmoid(p_vel), targets['velocity']) * mask).sum() / (mask.sum() + 1e-6)
                        
                        # --- CAMBIO 3: PESOS EN LA LOSS ---
                        # Multiplicamos offset por 10.0 (igual que onset) para que el modelo no lo ignore.
                        # Estructura: (10 * Onset) + Frame + (10 * Offset) + Vel
                        loss = (10.0 * l_on) + l_fr + (10.0 * l_off) + l_vel
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    t_loss += loss.item()
                    bar.set_postfix(loss=loss.item())
            
            # --- VAL ---
            model.eval()
            v_loss = 0
            ref_all, est_all = [], []
            fr_preds, fr_targs = [], []
            off_preds, off_targs = [], []
            vel_accum = 0; vel_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    p_on, p_fr, p_off, p_vel = model(hcqt)
                    
                    l_on = crit_onset(p_on, targets['onset'])
                    l_fr = crit_frame(p_fr, targets['frame'])
                    l_off = crit_offset(p_off, targets['offset'])
                    mask = targets['frame']
                    l_vel = (crit_vel(torch.sigmoid(p_vel), targets['velocity']) * mask).sum() / (mask.sum() + 1e-6)
                    
                    # Aplicamos los mismos pesos en validación para ser consistentes
                    v_loss += ((10.0 * l_on) + l_fr + (10.0 * l_off) + l_vel).item()
                    
                    pr_on = torch.sigmoid(p_on)
                    pr_fr = torch.sigmoid(p_fr)
                    pr_off = torch.sigmoid(p_off)
                    
                    # Notes decoding
                    for i in range(len(hcqt)):
                        est = tensor_to_notes(pr_on[i].cpu().numpy(), pr_fr[i].cpu().numpy(), t_onset=THRESHOLD_ONSET, t_frame=THRESHOLD_FRAME)
                        ref = tensor_to_notes(targets['onset'][i].cpu().numpy(), targets['frame'][i].cpu().numpy(), t_onset=0.5, t_frame=0.5)
                        est_all.append(est)
                        ref_all.append(ref)
                    
                    # Pixel-wise decoding
                    fr_preds.append((pr_fr > THRESHOLD_FRAME).cpu().numpy().flatten())
                    fr_targs.append((targets['frame'] > 0.5).cpu().numpy().flatten())
                    
                    # Aquí usamos el THRESHOLD_OFFSET más bajo (0.3)
                    off_preds.append((pr_off > THRESHOLD_OFFSET).cpu().numpy().flatten())
                    off_targs.append((targets['offset'] > 0.5).cpu().numpy().flatten())
                    
                    v_p = torch.sigmoid(p_vel).cpu().numpy().flatten()
                    v_t = targets['velocity'].cpu().numpy().flatten()
                    m = mask.cpu().numpy().flatten().astype(bool)
                    if m.sum() > 0:
                        vel_accum += mean_squared_error(v_t[m], v_p[m]) * m.sum()
                        vel_count += m.sum()

            # --- METRICS CALCULATION ---
            avg_t_loss = t_loss / len(train_loader)
            avg_v_loss = v_loss / len(val_loader)
            
            # 1. Onset
            onset_f1, onset_p, onset_r = compute_metrics_standard(ref_all, est_all)
            
            # 2. Frame
            f_p = np.concatenate(fr_preds); f_t = np.concatenate(fr_targs)
            frame_f1 = f1_score(f_t, f_p, zero_division=0)
            frame_p = precision_score(f_t, f_p, zero_division=0)
            frame_r = recall_score(f_t, f_p, zero_division=0)
            
            # 3. Offset (Ahora debería ser > 0.0)
            o_p = np.concatenate(off_preds); o_t = np.concatenate(off_targs)
            offset_f1 = f1_score(o_t, o_p, zero_division=0)
            offset_p = precision_score(o_t, o_p, zero_division=0)
            offset_r = recall_score(o_t, o_p, zero_division=0)
            
            # 4. Velocity
            vel_mse = vel_accum / (vel_count + 1e-8)
            curr_lr = optimizer.param_groups[0]['lr']

            print("-" * 80)
            print(f"🏁 Epoch {epoch+1} Results:")
            print(f"   📉 Loss   : Train={avg_t_loss:.4f} | Val={avg_v_loss:.4f}")
            print(f"   🎹 Onset  : F1={onset_f1:.4f} | P={onset_p:.4f} | R={onset_r:.4f}")
            print(f"   🖼️ Frame  : F1={frame_f1:.4f} | P={frame_p:.4f} | R={frame_r:.4f}")
            print(f"   🏁 Offset : F1={offset_f1:.4f} | P={offset_p:.4f} | R={offset_r:.4f}")
            print(f"   ⚡ Vel    : MSE={vel_mse:.4f}")
            print(f"   🧠 LR     : {curr_lr:.2e}")
            print("-" * 80)
            
            log_line = f"{epoch+1},{avg_t_loss},{avg_v_loss},{onset_f1},{onset_p},{onset_r},{frame_f1},{frame_p},{frame_r},{offset_f1},{offset_p},{offset_r},{vel_mse},{curr_lr}\n"
            log_file.write(log_line)
            log_file.flush()
            
            scheduler.step(onset_f1)
            
            if onset_f1 > best_f1:
                best_f1 = onset_f1
                torch.save(model.state_dict(), "best_hppnet.pth")
                print(f"   💾 Nuevo Récord! Modelo guardado (F1: {best_f1:.4f})")

    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento detenido.")
        
    finally:
        log_file.close()
        plot_training_history("training_log.csv")