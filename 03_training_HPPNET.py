####CORREGIR ERRORES####

import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import optuna

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SEGMENT_FRAMES = 320 
N_TRIALS = 5            # Número de pruebas de Optuna
MAX_EPOCHS_OPTUNA = 5   # Épocas por prueba (rápido)
FINAL_EPOCHS = 40       # Épocas del entrenamiento final
DATA_PATH = Path("processed_data_HPPNET") 

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
        all_files = sorted(list((self.processed_dir / "inputs_hcqt").glob("*.npy")))
        
        if len(all_files) == 0:
            raise RuntimeError(f"❌ No se encontraron archivos en {self.processed_dir}/inputs_hcqt")

        random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - val_split))
        self.files = all_files[:split_idx] if split == 'train' else all_files[split_idx:]
        print(f"🔄 Cargando {len(self.files)} archivos para {split}...")
        
        self.data = []
        for f in tqdm(self.files, desc=f"Loading {split}"):
            fid = f.name
            try:
                self.data.append({
                    "hcqt": np.load(self.processed_dir / "inputs_hcqt" / fid).astype(np.float32), 
                    "onset": np.load(self.processed_dir / "targets_onset" / fid).astype(np.float32),
                    "frame": np.load(self.processed_dir / "targets_frame" / fid).astype(np.float32),
                    "offset": np.load(self.processed_dir / "targets_offset" / fid).astype(np.float32),
                    "velocity": np.load(self.processed_dir / "targets_velocity" / fid).astype(np.float32)
                })
            except: continue

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        total = item['hcqt'].shape[0]
        if total > SEGMENT_FRAMES:
            s = np.random.randint(0, total - SEGMENT_FRAMES)
            e = s + SEGMENT_FRAMES
        else:
            s, e = 0, total
        
        # (Time, 88, 3) -> (3, Time, 88)
        hcqt = torch.tensor(item['hcqt'][s:e]).permute(2, 0, 1)

        return {
            "hcqt": hcqt, 
            "onset": torch.tensor(item['onset'][s:e]),
            "frame": torch.tensor(item['frame'][s:e]),
            "offset": torch.tensor(item['offset'][s:e]),
            "velocity": torch.tensor(item['velocity'][s:e])
        }

# ==========================================
# 2. MODELO DUAL-STREAM (TIMING vs STATE)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 1)) if pool else None 

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.pool: x = self.pool(x)
        return x

class PianoCRNN(nn.Module):
    def __init__(self, hidden_lstm=64): 
        super().__init__()
        
        # --- ENCODER COMPARTIDO ---
        self.enc1 = ConvBlock(3, 16, pool=False) 
        self.enc2 = ConvBlock(16, 32, pool=True)
        self.enc3 = ConvBlock(32, 64, pool=True)
        
        self.lstm_input_size = 64 * 88 

        # --- RAMA 1: TIMING (Onset + Offset) ---
        self.lstm_time = nn.LSTM(self.lstm_input_size, hidden_lstm, batch_first=True, bidirectional=True)
        self.proj_time = nn.Linear(hidden_lstm * 2, self.lstm_input_size)
        self.up3_time = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.dec3_time = ConvBlock(64, 32)
        self.up2_time = nn.ConvTranspose2d(32, 16, kernel_size=(2, 1), stride=(2, 1))
        self.dec2_time = ConvBlock(32, 16)
        
        self.head_onset = nn.Conv2d(16, 1, 1)
        self.head_offset = nn.Conv2d(16, 1, 1)

        # --- RAMA 2: STATE (Frame + Velocity) ---
        self.lstm_state = nn.LSTM(self.lstm_input_size, hidden_lstm, batch_first=True, bidirectional=True)
        self.proj_state = nn.Linear(hidden_lstm * 2, self.lstm_input_size)
        self.up3_state = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.dec3_state = ConvBlock(64, 32)
        self.up2_state = nn.ConvTranspose2d(32, 16, kernel_size=(2, 1), stride=(2, 1))
        self.dec2_state = ConvBlock(32, 16)

        self.head_frame = nn.Conv2d(16, 1, 1)
        self.head_velocity = nn.Conv2d(16, 1, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 1. Encoder Compartido
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b, c, t, f = e3.shape
        lstm_in = e3.permute(0, 2, 1, 3).reshape(b, t, c*f)
        
        # 2. Rama Timing
        out_time, _ = self.lstm_time(lstm_in)
        out_time = self.dropout(self.relu(out_time))
        proj_time = self.proj_time(out_time).view(b, t, c, f).permute(0, 2, 1, 3)
        
        d3_t = self.dec3_time(torch.cat([self.up3_time(proj_time), e2], dim=1))
        d2_t = self.dec2_time(torch.cat([self.up2_time(d3_t), e1], dim=1))
        
        # 3. Rama State
        out_state, _ = self.lstm_state(lstm_in)
        out_state = self.dropout(self.relu(out_state))
        proj_state = self.proj_state(out_state).view(b, t, c, f).permute(0, 2, 1, 3)
        
        d3_s = self.dec3_state(torch.cat([self.up3_state(proj_state), e2], dim=1))
        d2_s = self.dec2_state(torch.cat([self.up2_state(d3_s), e1], dim=1))

        return (
            self.head_onset(d2_t).squeeze(1),
            self.head_frame(d2_s).squeeze(1),
            self.head_offset(d2_t).squeeze(1),
            self.head_velocity(d2_s).squeeze(1)
        )

# ==========================================
# 3. OPTUNA
# ==========================================
print("\n📂 Cargando Datos...")
full_train_ds = PianoDataset(DATA_PATH, split='train')
full_val_ds = PianoDataset(DATA_PATH, split='val')

def calculate_simple_f1(model, loader):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for batch in loader:
            hcqt = batch['hcqt'].to(DEVICE)
            p_on, _, _, _ = model(hcqt)
            preds.append(torch.sigmoid(p_on).cpu().numpy().flatten())
            targs.append(batch['onset'].cpu().numpy().flatten())
    p_bin = np.concatenate(preds) > 0.4
    t_bin = np.concatenate(targs) > 0.5
    return f1_score(t_bin, p_bin, zero_division=0)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 6]) 
    onset_w = trial.suggest_float("onset_weight", 15.0, 30.0)
    frame_w = trial.suggest_float("frame_weight", 1.0, 8.0)

    train_loader = DataLoader(full_train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(full_val_ds, batch_size=batch_size, shuffle=False)

    model = PianoCRNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([onset_w]).to(DEVICE))
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([frame_w]).to(DEVICE))
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(DEVICE))
    crit_vel = nn.MSELoss()

    best_f1 = 0.0
    pbar = tqdm(range(MAX_EPOCHS_OPTUNA), desc=f"Trial {trial.number}", leave=False)

    for epoch in pbar:
        model.train()
        train_loss = 0
        for batch in train_loader:
            hcqt = batch['hcqt'].to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
            
            optimizer.zero_grad()
            with autocast():
                p_on, p_fr, p_off, p_vel = model(hcqt)
                loss = (crit_onset(p_on, targets['onset']) + 
                        crit_frame(p_fr, targets['frame']) +
                        crit_offset(p_off, targets['offset']) +
                        crit_vel(torch.sigmoid(p_vel), targets['velocity']))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        f1 = calculate_simple_f1(model, val_loader)
        pbar.set_postfix({"Loss": f"{train_loss/len(train_loader):.3f}", "F1": f"{f1:.3f}"})
        
        trial.report(f1, epoch)
        if f1 > best_f1: best_f1 = f1
        if trial.should_prune():
            pbar.close()
            raise optuna.exceptions.TrialPruned()
            
    return best_f1

# ==========================================
# 4. PLOTTING FINAL (CORREGIDO)
# ==========================================
def evaluate_and_plot(model, loader, filename_suffix="DUAL_STREAM"):
    model.eval()
    metrics = {k: {'preds': [], 'targets': []} for k in ['onset', 'frame', 'offset', 'velocity']}
    
    print("\n📊 Generando Métricas Finales Detalladas...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
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
    print("📢 REPORTE FINAL DE MÉTRICAS")
    print("="*60)

    for i, task in enumerate(tasks):
        p_bin = np.concatenate(metrics[task]['preds']) > 0.4
        t_bin = np.concatenate(metrics[task]['targets']) > 0.5
        
        f1 = f1_score(t_bin, p_bin)
        prec = precision_score(t_bin, p_bin, zero_division=0)
        rec = recall_score(t_bin, p_bin, zero_division=0)
        
        # --- EL PRINT QUE PEDISTE ---
        print(f"🔹 {task.upper():<7} -> F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
        cm = confusion_matrix(t_bin, p_bin)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"{task.upper()} F1: {f1:.2f}")

    # Velocity
    v_p = np.concatenate(metrics['velocity']['preds'])
    v_t = np.concatenate(metrics['velocity']['targets'])
    mse = mean_squared_error(v_t, v_p)
    print(f"🔹 VELOCITY -> MSE: {mse:.6f}")
    print("="*60)

    plt.tight_layout()
    plt.savefig(f"metrics_{filename_suffix}.png")
    print(f"✅ Gráfica guardada: metrics_{filename_suffix}.png")

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    print(f"\n🚀 DUAL-STREAM HPPNET Start ({DEVICE})")
    
    # 1. OPTUNA
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS)
    print("🏆 Best Params:", study.best_params)
    
    # 2. TRAIN FINAL
    print("\n🔄 Training Final Model...")
    bp = study.best_params
    final_model = PianoCRNN().to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=bp['lr'])
    scaler = GradScaler()
    
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bp['onset_weight']]).to(DEVICE))
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bp['frame_weight']]).to(DEVICE))
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(DEVICE))
    crit_vel = nn.MSELoss()
    
    train_loader = DataLoader(full_train_ds, batch_size=bp['batch_size'], shuffle=True)
    val_loader = DataLoader(full_val_ds, batch_size=bp['batch_size'], shuffle=False)
    
    hist_train, hist_val = [], []
    
    for epoch in range(FINAL_EPOCHS):
        final_model.train()
        t_loss = 0
        
        with tqdm(train_loader, unit="batch", leave=False) as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{FINAL_EPOCHS}")
            for batch in tepoch:
                hcqt = batch['hcqt'].to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                
                optimizer.zero_grad()
                with autocast():
                    p_on, p_fr, p_off, p_vel = final_model(hcqt)
                    loss = (crit_onset(p_on, targets['onset']) + 
                            crit_frame(p_fr, targets['frame']) +
                            crit_offset(p_off, targets['offset']) +
                            crit_vel(torch.sigmoid(p_vel), targets['velocity']))
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                t_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        # Val
        final_model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                hcqt = batch['hcqt'].to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                p_on, p_fr, p_off, p_vel = final_model(hcqt)
                loss = (crit_onset(p_on, targets['onset']) + 
                        crit_frame(p_fr, targets['frame']) +
                        crit_offset(p_off, targets['offset']) +
                        crit_vel(torch.sigmoid(p_vel), targets['velocity']))
                v_loss += loss.item()
        
        avg_t = t_loss/len(train_loader)
        avg_v = v_loss/len(val_loader)
        hist_train.append(avg_t)
        hist_val.append(avg_v)
        
        print(f"✅ Ep {epoch+1} | T.Loss: {avg_t:.4f} | V.Loss: {avg_v:.4f}")

    # 3. SAVE
    torch.save(final_model.state_dict(), "best_model_DUAL.pth")
    
    plt.figure()
    plt.plot(hist_train, label='Train')
    plt.plot(hist_val, label='Val')
    plt.legend()
    plt.savefig("loss_curve_DUAL.png")
    
    evaluate_and_plot(final_model, val_loader)