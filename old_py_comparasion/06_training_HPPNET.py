import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import optuna
import sys

# ==========================================
# 0. CONFIGURACIÓN (HPPNet Paper Specs)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SEGMENT_FRAMES = 320  # ~10s
BINS_PER_OCTAVE = 12  
DATA_PATH = Path("processed_data_HPPNET")

# Configuración de Entrenamiento
N_TRIALS = 3            
MAX_EPOCHS_OPTUNA = 2   
FINAL_EPOCHS = 30       

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
                    self.segments.append((idx, start, end))
            except: continue

    def __len__(self): return len(self.segments)

    def __getitem__(self, idx):
        file_idx, start, end = self.segments[idx]
        fid = self.files[file_idx].name
        
        try:
            p_in = self.processed_dir / "inputs_hcqt" / fid
            p_on = self.processed_dir / "targets_onset" / fid
            p_fr = self.processed_dir / "targets_frame" / fid
            p_off = self.processed_dir / "targets_offset" / fid
            p_vel = self.processed_dir / "targets_velocity" / fid
            
            hcqt = np.load(p_in, mmap_mode='r')[start:end] 
            onset = np.load(p_on, mmap_mode='r')[start:end]
            frame = np.load(p_fr, mmap_mode='r')[start:end]
            offset = np.load(p_off, mmap_mode='r')[start:end]
            vel = np.load(p_vel, mmap_mode='r')[start:end]
            
            curr_len = hcqt.shape[0]
            if curr_len < SEGMENT_FRAMES:
                pad = SEGMENT_FRAMES - curr_len
                hcqt = np.pad(hcqt, ((0, pad), (0,0), (0,0)))
                onset = np.pad(onset, ((0, pad), (0,0)))
                frame = np.pad(frame, ((0, pad), (0,0)))
                offset = np.pad(offset, ((0, pad), (0,0)))
                vel = np.pad(vel, ((0, pad), (0,0)))
            
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
# 2. ARQUITECTURA HPPNET
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
# 3. UTILS & PLOT
# ==========================================
def calculate_simple_f1(model, loader):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 30: break 
            hcqt = batch['hcqt'].to(DEVICE)
            p_on, _, _, _ = model(hcqt)
            preds.append(torch.sigmoid(p_on).cpu().numpy().flatten())
            targs.append(batch['onset'].numpy().flatten())
    p_bin = np.concatenate(preds) > 0.4
    t_bin = np.concatenate(targs) > 0.5
    return f1_score(t_bin, p_bin, zero_division=0)

def evaluate_and_plot(model, loader, filename_suffix="HPPNET"):
    model.eval()
    metrics = {k: {'preds': [], 'targets': []} for k in ['onset', 'frame', 'offset', 'velocity']}
    
    print(f"\n📊 Generando Gráficas ({filename_suffix})...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval Plot"):
            hcqt = batch['hcqt'].to(DEVICE)
            p_on, p_fr, p_off, p_vel = model(hcqt)
            
            metrics['onset']['preds'].append(torch.sigmoid(p_on).cpu().numpy().flatten())
            metrics['onset']['targets'].append(batch['onset'].numpy().flatten())
            metrics['frame']['preds'].append(torch.sigmoid(p_fr).cpu().numpy().flatten())
            metrics['frame']['targets'].append(batch['frame'].numpy().flatten())
            metrics['offset']['preds'].append(torch.sigmoid(p_off).cpu().numpy().flatten())
            metrics['offset']['targets'].append(batch['offset'].numpy().flatten())
            metrics['velocity']['preds'].append(torch.sigmoid(p_vel).cpu().numpy().flatten())
            metrics['velocity']['targets'].append(batch['velocity'].numpy().flatten())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tasks = ['onset', 'frame', 'offset']
    
    print("\n" + "="*60)
    print(f"📢 REPORTE FINAL ({filename_suffix})")
    print("="*60)

    for i, task in enumerate(tasks):
        p_bin = np.concatenate(metrics[task]['preds']) > 0.4
        t_bin = np.concatenate(metrics[task]['targets']) > 0.5
        f1 = f1_score(t_bin, p_bin, zero_division=0)
        
        cm = confusion_matrix(t_bin, p_bin)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"{task.upper()} F1: {f1:.2f}")

    plt.tight_layout()
    plt.savefig(f"metrics_{filename_suffix}.png")
    print(f"✅ Gráfica guardada: metrics_{filename_suffix}.png")

# ==========================================
# 4. OPTUNA
# ==========================================
def objective(trial, train_ds, val_ds):
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 48])
    onset_w = trial.suggest_float("onset_weight", 1.5, 3.0) 
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    model = HPPNet(in_channels=3, lstm_hidden=lstm_hidden).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda') 
    
    # Usamos pesos equilibrados también en Optuna para que la métrica sea realista
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([onset_w]).to(DEVICE))
    
    best_f1 = 0.0
    for epoch in range(MAX_EPOCHS_OPTUNA):
        model.train()
        for i, batch in enumerate(train_loader):
            if i > 100: break 
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
        if f1 > best_f1: best_f1 = f1
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            
    return best_f1

# ==========================================
# 5. MAIN (TRAINING CON LOGS DETALLADOS)
# ==========================================
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\n🚀 HPPNET Training Start ({DEVICE})")
    
    print("\n📂 Inicializando Datasets...")
    full_train_ds = PianoDataset(DATA_PATH, split='train')
    full_val_ds = PianoDataset(DATA_PATH, split='val')
    print(f"✅ Clips Train: {len(full_train_ds)} | Clips Val: {len(full_val_ds)}")

    print("\n🔍 Optuna Search (Buscando hiperparámetros)...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, full_train_ds, full_val_ds), n_trials=N_TRIALS)
    bp = study.best_params
    print("\n🏆 Mejores Parámetros:", bp)
    
    print("\n🔄 Iniciando Entrenamiento Final...")
    
    final_model = HPPNet(in_channels=3, lstm_hidden=bp['lstm_hidden']).to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=bp['lr'])
    scaler = torch.amp.GradScaler('cuda') 
    
    # --- CONFIGURACIÓN DE PÉRDIDA ---
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bp['onset_weight']]).to(DEVICE))
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(DEVICE)) 
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))
    crit_vel = nn.MSELoss(reduction='none') 
    
    train_loader = DataLoader(full_train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(full_val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    
    # --- Logging System Completo ---
    log_file = open("training_log.csv", "w")
    # Cabecera CSV con TODO
    header = "epoch,train_loss,val_loss,"
    header += "onset_f1,onset_prec,onset_rec,"
    header += "frame_f1,frame_prec,frame_rec,"
    header += "offset_f1,offset_prec,offset_rec\n"
    log_file.write(header)
    
    hist_train, hist_val = [], []
    best_val_f1 = 0.0
    
    try:
        for epoch in range(FINAL_EPOCHS):
            final_model.train()
            t_loss = 0
            
            # --- TRAINING LOOP ---
            with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{FINAL_EPOCHS}")
                for batch in tepoch:
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        p_on, p_fr, p_off, p_vel = final_model(hcqt)
                        
                        l_on = crit_onset(p_on, targets['onset'])
                        l_fr = crit_frame(p_fr, targets['frame'])
                        l_off = crit_offset(p_off, targets['offset'])
                        
                        vel_loss_raw = crit_vel(torch.sigmoid(p_vel), targets['velocity'])
                        mask = targets['frame']
                        l_vel = (vel_loss_raw * mask).sum() / (mask.sum() + 1e-6)

                        loss = l_on + l_fr + l_off + l_vel
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    t_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())

            # --- VALIDATION LOOP ---
            final_model.eval()
            v_loss = 0
            
            # Diccionarios para acumular predicciones de las 3 tareas
            val_preds = {'onset': [], 'frame': [], 'offset': []}
            val_targs = {'onset': [], 'frame': [], 'offset': []}
            
            with torch.no_grad():
                for batch in val_loader:
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    
                    p_on, p_fr, p_off, p_vel = final_model(hcqt)
                    
                    # 1. Loss
                    l_on = crit_onset(p_on, targets['onset'])
                    l_fr = crit_frame(p_fr, targets['frame'])
                    l_off = crit_offset(p_off, targets['offset'])
                    vel_loss_raw = crit_vel(torch.sigmoid(p_vel), targets['velocity'])
                    mask = targets['frame']
                    l_vel = (vel_loss_raw * mask).sum() / (mask.sum() + 1e-6)
                    v_loss += (l_on + l_fr + l_off + l_vel).item()
                    
                    # 2. Guardar predicciones (Sigmoid y a CPU)
                    val_preds['onset'].append(torch.sigmoid(p_on).cpu().numpy().flatten())
                    val_targs['onset'].append(targets['onset'].cpu().numpy().flatten())
                    
                    val_preds['frame'].append(torch.sigmoid(p_fr).cpu().numpy().flatten())
                    val_targs['frame'].append(targets['frame'].cpu().numpy().flatten())
                    
                    val_preds['offset'].append(torch.sigmoid(p_off).cpu().numpy().flatten())
                    val_targs['offset'].append(targets['offset'].cpu().numpy().flatten())
            
            avg_t = t_loss/len(train_loader)
            avg_v = v_loss/len(val_loader)
            hist_train.append(avg_t)
            hist_val.append(avg_v)
            
            # --- CÁLCULO DE MÉTRICAS DETALLADAS ---
            tasks = ['onset', 'frame', 'offset']
            results = {} # Aquí guardaremos los resultados para imprimir y CSV
            
            print(f"\n✅ Ep {epoch+1} | T.Loss: {avg_t:.4f} | V.Loss: {avg_v:.4f}")
            
            for task in tasks:
                vp = np.concatenate(val_preds[task]) > 0.5
                vt = np.concatenate(val_targs[task]) > 0.5
                
                f1 = f1_score(vt, vp, zero_division=0)
                prec = precision_score(vt, vp, zero_division=0)
                rec = recall_score(vt, vp, zero_division=0)
                
                results[task] = (f1, prec, rec)
                print(f"   🔹 {task.upper():<6} -> F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

            # --- Escribir en CSV ---
            # onset_f1, onset_p, onset_r, frame_f1...
            csv_line = f"{epoch+1},{avg_t:.6f},{avg_v:.6f},"
            csv_line += f"{results['onset'][0]:.6f},{results['onset'][1]:.6f},{results['onset'][2]:.6f},"
            csv_line += f"{results['frame'][0]:.6f},{results['frame'][1]:.6f},{results['frame'][2]:.6f},"
            csv_line += f"{results['offset'][0]:.6f},{results['offset'][1]:.6f},{results['offset'][2]:.6f}\n"
            
            log_file.write(csv_line)
            log_file.flush()

            # --- Guardar Mejor Modelo (Basado en Onset F1) ---
            current_onset_f1 = results['onset'][0]
            if current_onset_f1 > best_val_f1:
                best_val_f1 = current_onset_f1
                torch.save(final_model.state_dict(), "best_model_HPPNET.pth")
                print(f"   💾 ¡Nuevo Récord Onset ({best_val_f1:.4f})! Modelo guardado.")

    except KeyboardInterrupt:
        print("\n🛑 Detenido por usuario. Guardando estado...")
        torch.save(final_model.state_dict(), "interrupted_model.pth")
    finally:
        log_file.close()

    # 4. PLOT FINAL
    if len(hist_train) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(hist_train, label='Train Loss')
        plt.plot(hist_val, label='Val Loss')
        plt.title("Curva de Aprendizaje HPPNet")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve_HPPNET.png")
        print("\n📈 Gráfica guardada.")
        
        if os.path.exists("best_model_HPPNET.pth"):
            print("🔄 Cargando mejor modelo para plots finales...")
            final_model.load_state_dict(torch.load("best_model_HPPNET.pth"))
            
        evaluate_and_plot(final_model, val_loader, filename_suffix="FINAL")
    else:
        print("⚠️ No se completó ninguna época.")