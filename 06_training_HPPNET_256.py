import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import optuna
import sys

# ==========================================
# 0. CONFIGURACIÓN (1060 Ti Optimizado)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# --- AJUSTES DEL DATASET ---
SEGMENT_FRAMES = 640        # ~10 segundos (para no saturar VRAM)
DATA_PATH = Path("processed_data_HPPNET_256") # <--- RUTA ACTUALIZADA

# --- AJUSTES DE ENTRENAMIENTO ---
PHYSICAL_BATCH_SIZE = 4     # Batch real en GPU
ACCUM_STEPS = 4             # Gradiente acumulado (Simula Batch = 16)
N_TRIALS = 3                # Intentos de Optuna
MAX_EPOCHS_OPTUNA = 3       # Épocas cortas para Optuna
FINAL_EPOCHS = 40           # Entrenamiento real

# Nombres de salida con sufijo _256
LOG_FILE_NAME = "training_log_256.csv"
MODEL_SAVE_NAME = "best_model_HPPNET_256.pth"
INTERRUPTED_NAME = "interrupted_model_256.pth"
PLOT_NAME = "loss_curve_256.png"

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
        if not all_files: raise RuntimeError(f"❌ No hay archivos .npy en {p}")

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
                    if (end - start) > 100: 
                        self.segments.append((idx, start, end))
            except: continue

    def __len__(self): return len(self.segments)

    def __getitem__(self, idx):
        file_idx, start, end = self.segments[idx]
        fid = self.files[file_idx].name
        
        try:
            # Carga perezosa
            hcqt = np.load(self.processed_dir / "inputs_hcqt" / fid, mmap_mode='r')[start:end] 
            onset = np.load(self.processed_dir / "targets_onset" / fid, mmap_mode='r')[start:end]
            frame = np.load(self.processed_dir / "targets_frame" / fid, mmap_mode='r')[start:end]
            offset = np.load(self.processed_dir / "targets_offset" / fid, mmap_mode='r')[start:end]
            vel = np.load(self.processed_dir / "targets_velocity" / fid, mmap_mode='r')[start:end]
            
            # Padding
            curr_len = hcqt.shape[0]
            if curr_len < SEGMENT_FRAMES:
                pad = SEGMENT_FRAMES - curr_len
                hcqt = np.pad(hcqt, ((0, pad), (0,0), (0,0)))
                onset = np.pad(onset, ((0, pad), (0,0)))
                frame = np.pad(frame, ((0, pad), (0,0)))
                offset = np.pad(offset, ((0, pad), (0,0)))
                vel = np.pad(vel, ((0, pad), (0,0)))
            
            # Transpose (Time, Freq, Ch) -> (Ch, Freq, Time)
            hcqt_t = torch.tensor(hcqt).permute(2, 1, 0).float()
            
            return {
                "hcqt": hcqt_t,
                "onset": torch.tensor(onset).float(),
                "frame": torch.tensor(frame).float(),
                "offset": torch.tensor(offset).float(),
                "velocity": torch.tensor(vel).float()
            }
        except Exception:
            z = torch.zeros(SEGMENT_FRAMES, 88)
            return {"hcqt": torch.zeros(3, 88, SEGMENT_FRAMES), "onset": z, "frame": z, "offset": z, "velocity": z}

# ==========================================
# 2. ARQUITECTURA HPPNET
# ==========================================
class DilatedFreqBlock(nn.Module):
    def __init__(self, channels, dilation_freq=3):
        super().__init__()
        pad_freq = int((dilation_freq * (3 - 1)) / 2)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, 
                              padding=(pad_freq, 1), dilation=(dilation_freq, 1))
        self.norm = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.norm(self.conv(x)))

class HDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        # 12 bins por octava
        dilations = [int(np.round(12 * np.log2(h))) for h in [1, 2, 3, 4]]
        for d in dilations:
            d_safe = max(1, d)
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), 
                                        padding=(d_safe, 1), dilation=(d_safe, 1)))
    def forward(self, x): return sum([conv(x) for conv in self.convs])

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

class HPPNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=24, lstm_hidden=64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, padding=2), nn.InstanceNorm2d(base_channels), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.InstanceNorm2d(base_channels), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.InstanceNorm2d(base_channels), nn.ReLU(),
        )
        self.hd_conv = HDConv(base_channels, base_channels)
        self.hd_bn = nn.InstanceNorm2d(base_channels)
        self.hd_relu = nn.ReLU()
        
        deep_channels = base_channels * 2
        self.adapter = nn.Conv2d(base_channels, deep_channels, 1)
        self.deep_blocks = nn.Sequential(
            DilatedFreqBlock(deep_channels, dilation_freq=3), DilatedFreqBlock(deep_channels, dilation_freq=3),
            DilatedFreqBlock(deep_channels, dilation_freq=3), DilatedFreqBlock(deep_channels, dilation_freq=3)
        )
        self.head_onset = FG_LSTM(deep_channels, lstm_hidden)
        self.head_frame = FG_LSTM(deep_channels, lstm_hidden)
        self.head_offset = FG_LSTM(deep_channels, lstm_hidden)
        self.head_velocity = FG_LSTM(deep_channels, lstm_hidden)

    def forward(self, x):
        x = self.conv_block(x)
        x_hd = self.hd_conv(x)
        x = self.hd_relu(self.hd_bn(x_hd))
        x = self.adapter(x)
        features = self.deep_blocks(x)
        return (self.head_onset(features), self.head_frame(features), 
                self.head_offset(features), self.head_velocity(features))

# ==========================================
# 3. UTILS & OPTUNA
# ==========================================
def calculate_simple_f1(model, loader):
    """F1 rápido para Optuna"""
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 20: break 
            hcqt = batch['hcqt'].to(DEVICE)
            p_on, _, _, _ = model(hcqt)
            preds.append(torch.sigmoid(p_on).cpu().numpy().flatten())
            targs.append(batch['onset'].numpy().flatten())
    p_bin = np.concatenate(preds) > 0.4
    t_bin = np.concatenate(targs) > 0.5
    return f1_score(t_bin, p_bin, zero_division=0)

def objective(trial, train_ds, val_ds):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [48, 64]) 
    onset_w = trial.suggest_float("onset_weight", 2.0, 5.0) 
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    model = HPPNet(in_channels=3, lstm_hidden=lstm_hidden).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda') 
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([onset_w]).to(DEVICE))
    
    best_f1 = 0.0
    for epoch in range(MAX_EPOCHS_OPTUNA):
        model.train()
        for i, batch in enumerate(train_loader):
            if i > 80: break 
            optimizer.zero_grad()
            hcqt = batch['hcqt'].to(DEVICE)
            targets = batch['onset'].to(DEVICE)
            with torch.amp.autocast('cuda'):
                p_on, _, _, _ = model(hcqt)
                loss = crit_onset(p_on, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        f1 = calculate_simple_f1(model, val_loader)
        trial.report(f1, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        if f1 > best_f1: best_f1 = f1
    return best_f1

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\n🚀 HPPNET-SP V2 Start | ⚡ MODO OPTIMIZADO PARA GTX 1060 Ti ⚡")
    print(f"⚙️  Frames por Segmento: {SEGMENT_FRAMES} | Path: {DATA_PATH}")
    print(f"⚙️  Physical Batch: {PHYSICAL_BATCH_SIZE} | Accum Steps: {ACCUM_STEPS}")
    
    print("\n📂 Inicializando Datasets...")
    try:
        full_train_ds = PianoDataset(DATA_PATH, split='train')
        full_val_ds = PianoDataset(DATA_PATH, split='val')
        print(f"✅ Clips Train: {len(full_train_ds)} | Clips Val: {len(full_val_ds)}")
    except Exception as e:
        print(f"Error cargando datos: {e}")
        sys.exit(1)

    print(f"\n🔍 Optuna LR Search ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, full_train_ds, full_val_ds), n_trials=N_TRIALS)
    bp = study.best_params
    print(f"\n🏆 Mejores Params: {bp}")

    print(f"\n🏋️ Entrenamiento Final ({FINAL_EPOCHS} Épocas)...")
    final_model = HPPNet(in_channels=3, lstm_hidden=bp['lstm_hidden']).to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=bp['lr'])
    scaler = torch.amp.GradScaler('cuda')
    
    # Pesos de Loss
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bp['onset_weight']]).to(DEVICE))
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(DEVICE)) 
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(DEVICE))
    crit_vel = nn.MSELoss(reduction='none')

    # Data Loaders
    train_loader = DataLoader(full_train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(full_val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    log_file = open(LOG_FILE_NAME, "w")
    header = "epoch,train_loss,val_loss,onset_f1,onset_prec,onset_rec,frame_f1,frame_prec,frame_rec,offset_f1,offset_prec,offset_rec,vel_mse\n"
    log_file.write(header)
    
    hist_train, hist_val = [], []
    best_val_f1 = 0.0
    
    try:
        for epoch in range(FINAL_EPOCHS):
            final_model.train()
            t_loss = 0
            
            with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{FINAL_EPOCHS}")
                optimizer.zero_grad() 
                
                for i, batch in enumerate(tepoch):
                    if batch is None: continue
                    
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    
                    with torch.amp.autocast('cuda'):
                        p_on, p_fr, p_off, p_vel = final_model(hcqt)
                        
                        loss = crit_onset(p_on, targets['onset']) + \
                               crit_frame(p_fr, targets['frame']) + \
                               crit_offset(p_off, targets['offset'])
                        
                        v_raw = crit_vel(torch.sigmoid(p_vel), targets['velocity'])
                        # Velocity Loss solo donde hay frames activos
                        l_vel = (v_raw * targets['frame']).sum() / (targets['frame'].sum() + 1e-6)
                        loss += l_vel
                        
                        loss = loss / ACCUM_STEPS
                    
                    scaler.scale(loss).backward()
                    
                    if (i + 1) % ACCUM_STEPS == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    t_loss += loss.item() * ACCUM_STEPS 
                    tepoch.set_postfix(loss=loss.item() * ACCUM_STEPS)

            # --- VALIDACIÓN COMPLETA (Todas las métricas) ---
            final_model.eval()
            v_loss = 0
            val_preds = {'onset': [], 'frame': [], 'offset': [], 'velocity': []}
            val_targs = {'onset': [], 'frame': [], 'offset': [], 'velocity': []}
            
            with torch.no_grad():
                for batch in val_loader:
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    
                    p_on, p_fr, p_off, p_vel = final_model(hcqt)
                    
                    l_total = crit_onset(p_on, targets['onset']) + \
                              crit_frame(p_fr, targets['frame']) + \
                              crit_offset(p_off, targets['offset'])
                    v_raw = crit_vel(torch.sigmoid(p_vel), targets['velocity'])
                    l_total += (v_raw * targets['frame']).sum() / (targets['frame'].sum() + 1e-6)
                    v_loss += l_total.item()
                    
                    # Guardar para métricas
                    val_preds['onset'].append(torch.sigmoid(p_on).cpu().numpy().flatten())
                    val_targs['onset'].append(targets['onset'].cpu().numpy().flatten())
                    val_preds['frame'].append(torch.sigmoid(p_fr).cpu().numpy().flatten())
                    val_targs['frame'].append(targets['frame'].cpu().numpy().flatten())
                    val_preds['offset'].append(torch.sigmoid(p_off).cpu().numpy().flatten())
                    val_targs['offset'].append(targets['offset'].cpu().numpy().flatten())
                    
                    # Velocity (masked)
                    mask_np = targets['frame'].cpu().numpy().flatten() > 0.5
                    pred_vel = torch.sigmoid(p_vel).cpu().numpy().flatten()
                    targ_vel = targets['velocity'].cpu().numpy().flatten()
                    if mask_np.sum() > 0:
                        val_preds['velocity'].extend(pred_vel[mask_np])
                        val_targs['velocity'].extend(targ_vel[mask_np])

            avg_t = t_loss/len(train_loader)
            avg_v = v_loss/len(val_loader)
            hist_train.append(avg_t)
            hist_val.append(avg_v)
            
            print(f"\n✅ Ep {epoch+1} | T.Loss: {avg_t:.4f} | V.Loss: {avg_v:.4f}")
            
            # Cálculo de métricas
            results = {} 
            for task in ['onset', 'frame', 'offset']:
                vp = np.concatenate(val_preds[task]) > 0.4 # Threshold
                vt = np.concatenate(val_targs[task]) > 0.5
                f1 = f1_score(vt, vp, zero_division=0)
                prec = precision_score(vt, vp, zero_division=0)
                rec = recall_score(vt, vp, zero_division=0)
                results[task] = (f1, prec, rec)
                print(f"   🔹 {task.upper():<6} -> F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

            # Metric velocity MSE
            if len(val_preds['velocity']) > 0:
                vel_mse = mean_squared_error(val_targs['velocity'], val_preds['velocity'])
            else:
                vel_mse = 0.0
            print(f"   🔹 VEL    -> MSE: {vel_mse:.6f}")

            csv_line = f"{epoch+1},{avg_t:.6f},{avg_v:.6f}," + \
                       f"{results['onset'][0]:.6f},{results['onset'][1]:.6f},{results['onset'][2]:.6f}," + \
                       f"{results['frame'][0]:.6f},{results['frame'][1]:.6f},{results['frame'][2]:.6f}," + \
                       f"{results['offset'][0]:.6f},{results['offset'][1]:.6f},{results['offset'][2]:.6f}," + \
                       f"{vel_mse:.6f}\n"
            log_file.write(csv_line)
            log_file.flush()

            current_onset_f1 = results['onset'][0]
            if current_onset_f1 > best_val_f1:
                best_val_f1 = current_onset_f1
                torch.save(final_model.state_dict(), MODEL_SAVE_NAME)
                print(f"   💾 ¡Nuevo Récord ({best_val_f1:.4f})! -> {MODEL_SAVE_NAME}")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")
        torch.save(final_model.state_dict(), INTERRUPTED_NAME)
    finally:
        log_file.close()
        if len(hist_train) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(hist_train, label='Train Loss')
            plt.plot(hist_val, label='Val Loss')
            plt.title("HPPNet 256 Loss")
            plt.legend()
            plt.savefig(PLOT_NAME)