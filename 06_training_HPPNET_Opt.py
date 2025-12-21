import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import optuna
import sys

# ==========================================
# 0. CONFIGURACIÓN OPTIMIZADA (GTX 1060 Ti)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ⚡ OPTIMIZACIÓN DE MEMORIA ⚡
SEGMENT_FRAMES = 160  # 5s para permitir batch size más alto

# 📂 RUTA
DATA_PATH = Path("processed_data_HPPNET_2")

# ⚙️ ARQUITECTURA
BINS_PER_OCTAVE = 48        
LSTM_HIDDEN = 128           
IN_CHANNELS = 3             

# 🛡️ VELOCIDAD DE ENTRENAMIENTO
PHYSICAL_BATCH_SIZE = 4     
VIRTUAL_BATCH_SIZE = 16     
ACCUM_STEPS = max(1, VIRTUAL_BATCH_SIZE // PHYSICAL_BATCH_SIZE)

# ⏱️ CONFIGURACIÓN "EXPRESS" (Optuna rápido -> Training largo)
N_TRIALS = 2                # ⚡ Solo 2 intentos para encontrar LR rápido
MAX_EPOCHS_OPTUNA = 2       # ⚡ Solo 2 épocas de prueba
FINAL_EPOCHS = 50           # Entrenamiento final completo

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
                    if (end - start) > 30: 
                        self.segments.append((idx, start, end))
            except: continue

    def __len__(self): return len(self.segments)

    def __getitem__(self, idx):
        file_idx, start, end = self.segments[idx]
        fid = self.files[file_idx].name
        
        try:
            p_in = self.processed_dir / "inputs_hcqt" / fid
            hcqt = np.load(p_in, mmap_mode='r')[start:end] 
            hcqt_t = torch.tensor(hcqt).permute(2, 1, 0).float()
            
            targets = {}
            for t_name in ["onset", "frame", "offset", "velocity"]:
                p_t = self.processed_dir / f"targets_{t_name}" / fid
                data = np.load(p_t, mmap_mode='r')[start:end]
                
                if data.shape[0] < SEGMENT_FRAMES:
                    pad = SEGMENT_FRAMES - data.shape[0]
                    data = np.pad(data, ((0, pad), (0,0)))
                targets[t_name] = torch.tensor(data).float()

            if hcqt_t.shape[2] < SEGMENT_FRAMES:
                pad = SEGMENT_FRAMES - hcqt_t.shape[2]
                hcqt_t = torch.nn.functional.pad(hcqt_t, (0, pad))
            
            return {"hcqt": hcqt_t, **targets}
        except: return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)

# ==========================================
# 2. ARQUITECTURA HPPNET-SP
# ==========================================
class HarmonicDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, bins_per_octave):
        super().__init__()
        self.convs = nn.ModuleList()
        harmonics = [1, 2, 3, 4] 
        dilations = [int(np.round(bins_per_octave * np.log2(h))) for h in harmonics]
        for d in dilations:
            d_safe = max(1, d)
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), 
                                        padding=(d_safe, 1), dilation=(d_safe, 1)))
            
    def forward(self, x):
        return sum([conv(x) for conv in self.convs])

class AcousticBlock(nn.Module):
    def __init__(self, in_channels, base_channels, bins_per_octave):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1),
                                   nn.InstanceNorm2d(base_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(base_channels, base_channels, 3, padding=1),
                                   nn.InstanceNorm2d(base_channels), nn.ReLU())
        self.hd_conv = HarmonicDilatedConv(base_channels, base_channels, bins_per_octave)
        self.hd_bn = nn.InstanceNorm2d(base_channels)
        self.hd_relu = nn.ReLU()
        self.out_conv = nn.Sequential(nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                                      nn.InstanceNorm2d(base_channels*2), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.hd_relu(self.hd_bn(self.hd_conv(x)))
        return self.out_conv(x)

class FG_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        b, c, f, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        out, _ = self.lstm(x)
        out = self.proj(out).reshape(b, f, t).permute(0, 2, 1)
        return out

class HPPNet_Separate(nn.Module):
    def __init__(self, in_channels=3, base_channels=24, lstm_hidden=128):
        super().__init__()
        self.acoustic_onset = AcousticBlock(in_channels, base_channels, BINS_PER_OCTAVE)
        self.acoustic_frame = AcousticBlock(in_channels, base_channels, BINS_PER_OCTAVE)
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        feat_dim = base_channels * 2 
        self.head_onset = FG_LSTM(feat_dim, lstm_hidden)
        self.head_frame = FG_LSTM(feat_dim * 2, lstm_hidden)
        self.head_offset = FG_LSTM(feat_dim * 2, lstm_hidden)
        self.head_velocity = FG_LSTM(feat_dim * 2, lstm_hidden)

    def forward(self, x):
        ft_onset = self.acoustic_onset(x)
        out_onset = self.head_onset(self.pool(ft_onset))
        ft_frame = self.acoustic_frame(x)
        ft_combined = torch.cat([ft_frame, ft_onset.detach()], dim=1) 
        ft_combined_pooled = self.pool(ft_combined)
        out_frame = self.head_frame(ft_combined_pooled)
        out_offset = self.head_offset(ft_combined_pooled)
        out_vel = self.head_velocity(ft_combined_pooled)
        return out_onset, out_frame, out_offset, out_vel

# ==========================================
# 3. UTILS
# ==========================================
def calculate_simple_f1(model, loader):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 25: break 
            if batch is None: continue
            hcqt = batch['hcqt'].to(DEVICE)
            p_on, _, _, _ = model(hcqt)
            preds.append(torch.sigmoid(p_on).cpu().numpy().flatten())
            targs.append(batch['onset'].numpy().flatten())
    if not preds: return 0.0
    p_bin = np.concatenate(preds) > 0.4
    t_bin = np.concatenate(targs) > 0.5
    return f1_score(t_bin, p_bin, zero_division=0)

def evaluate_and_plot(model, loader, filename_suffix="HPPNET_2"):
    model.eval()
    metrics = {k: {'preds': [], 'targets': []} for k in ['onset', 'frame', 'offset']}
    
    print(f"\n📊 Generando Gráficas Finales ({filename_suffix})...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval Plot"):
            if batch is None: continue
            hcqt = batch['hcqt'].to(DEVICE)
            p_on, p_fr, p_off, _ = model(hcqt)
            
            metrics['onset']['preds'].append(torch.sigmoid(p_on).cpu().numpy().flatten())
            metrics['onset']['targets'].append(batch['onset'].numpy().flatten())
            metrics['frame']['preds'].append(torch.sigmoid(p_fr).cpu().numpy().flatten())
            metrics['frame']['targets'].append(batch['frame'].numpy().flatten())
            metrics['offset']['preds'].append(torch.sigmoid(p_off).cpu().numpy().flatten())
            metrics['offset']['targets'].append(batch['offset'].numpy().flatten())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tasks = ['onset', 'frame', 'offset']
    
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
# 4. OPTUNA (OPTIMIZADO & MODO EXPRESS)
# ==========================================
def objective(trial, train_ds, val_ds):
    # ⚡ FIJAMOS LOS PESOS A MANO PARA AHORRAR TIEMPO
    onset_w = 4.0  # Valor equilibrado y seguro
    
    # ⚡ SOLO BUSCAMOS LR
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    
    model = HPPNet_Separate(in_channels=IN_CHANNELS, lstm_hidden=LSTM_HIDDEN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([onset_w]).to(DEVICE))
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE)) 
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(DEVICE))
    crit_vel = nn.MSELoss(reduction='none')
    
    # ⚡ WINDOWS FIX: num_workers=0
    train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0, pin_memory=False)
    
    best_f1 = 0.0
    
    for epoch in range(MAX_EPOCHS_OPTUNA):
        model.train()
        for i, batch in enumerate(train_loader):
            # ⚡ BUCLE CORTO OPTUNA: Solo mira 100 batches
            if i > 100: break 
            if batch is None: continue
            
            hcqt = batch['hcqt'].to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
            
            with torch.amp.autocast('cuda'):
                p_on, p_fr, p_off, p_vel = model(hcqt) 
                
                loss = crit_onset(p_on, targets['onset']) + \
                       crit_frame(p_fr, targets['frame']) + \
                       crit_offset(p_off, targets['offset'])
                
                v_raw = crit_vel(torch.sigmoid(p_vel), targets['velocity'])
                loss += (v_raw * targets['frame']).sum() / (targets['frame'].sum() + 1e-6)
                loss = loss / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        f1 = calculate_simple_f1(model, val_loader)
        trial.report(f1, epoch)
        if f1 > best_f1: best_f1 = f1
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            
    return best_f1

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\n🚀 HPPNET-SP V2 Start | ⚡ MODO OPTIMIZADO PARA GTX 1060 Ti ⚡")
    print(f"⚙️  Frames por Segmento: {SEGMENT_FRAMES} (Antes 320)")
    print(f"⚙️  Physical Batch: {PHYSICAL_BATCH_SIZE} | Workers: 0 (Windows Fix)")
    
    print("\n📂 Inicializando Datasets...")
    try:
        full_train_ds = PianoDataset(DATA_PATH, split='train')
        full_val_ds = PianoDataset(DATA_PATH, split='val')
        print(f"✅ Clips Train: {len(full_train_ds)} | Clips Val: {len(full_val_ds)}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n🔍 Optuna LR Search ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, full_train_ds, full_val_ds), n_trials=N_TRIALS)
    bp = study.best_params
    print(f"\n🏆 Mejor LR encontrado: {bp['lr']}")

    print(f"\n🏋️ Entrenamiento Final ({FINAL_EPOCHS} Épocas)...")
    final_model = HPPNet_Separate(in_channels=IN_CHANNELS, lstm_hidden=LSTM_HIDDEN).to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=bp['lr'])
    scaler = torch.amp.GradScaler('cuda')
    
    # ⚡ PESO ONSET FIJO (4.0)
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(DEVICE))
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE)) 
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(DEVICE))
    crit_vel = nn.MSELoss(reduction='none')

    # ⚡ DATA LOADERS SEGUROS (Workers=0)
    train_loader = DataLoader(full_train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=False, collate_fn=collate_fn)
    val_loader = DataLoader(full_val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, 
                            num_workers=0, pin_memory=False, collate_fn=collate_fn)
    
    # ⚡ NOMBRES _OPTIM MANTENIDOS
    log_file = open("training_log_optimized.csv", "w")
    header = "epoch,train_loss,val_loss,onset_f1,onset_prec,onset_rec,frame_f1,frame_prec,frame_rec,offset_f1,offset_prec,offset_rec\n"
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
                        loss += (v_raw * targets['frame']).sum() / (targets['frame'].sum() + 1e-6)
                        loss = loss / ACCUM_STEPS
                    
                    scaler.scale(loss).backward()
                    
                    if (i + 1) % ACCUM_STEPS == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    t_loss += loss.item() * ACCUM_STEPS 
                    tepoch.set_postfix(loss=loss.item() * ACCUM_STEPS)

            # --- VALIDACIÓN ---
            final_model.eval()
            v_loss = 0
            val_preds = {'onset': [], 'frame': [], 'offset': []}
            val_targs = {'onset': [], 'frame': [], 'offset': []}
            
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None: continue
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    
                    p_on, p_fr, p_off, p_vel = final_model(hcqt)
                    
                    l_total = crit_onset(p_on, targets['onset']) + \
                              crit_frame(p_fr, targets['frame']) + \
                              crit_offset(p_off, targets['offset'])
                    v_raw = crit_vel(torch.sigmoid(p_vel), targets['velocity'])
                    l_total += (v_raw * targets['frame']).sum() / (targets['frame'].sum() + 1e-6)
                    v_loss += l_total.item()
                    
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
            
            print(f"\n✅ Ep {epoch+1} | T.Loss: {avg_t:.4f} | V.Loss: {avg_v:.4f}")
            
            tasks = ['onset', 'frame', 'offset']
            results = {} 
            for task in tasks:
                vp = np.concatenate(val_preds[task]) > 0.4
                vt = np.concatenate(val_targs[task]) > 0.5
                f1 = f1_score(vt, vp, zero_division=0)
                prec = precision_score(vt, vp, zero_division=0)
                rec = recall_score(vt, vp, zero_division=0)
                results[task] = (f1, prec, rec)
                print(f"   🔹 {task.upper():<6} -> F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

            csv_line = f"{epoch+1},{avg_t:.6f},{avg_v:.6f}," + \
                       f"{results['onset'][0]:.6f},{results['onset'][1]:.6f},{results['onset'][2]:.6f}," + \
                       f"{results['frame'][0]:.6f},{results['frame'][1]:.6f},{results['frame'][2]:.6f}," + \
                       f"{results['offset'][0]:.6f},{results['offset'][1]:.6f},{results['offset'][2]:.6f}\n"
            log_file.write(csv_line)
            log_file.flush()

            current_onset_f1 = results['onset'][0]
            if current_onset_f1 > best_val_f1:
                best_val_f1 = current_onset_f1
                torch.save(final_model.state_dict(), "best_model_HPPNET_OPTIM.pth")
                print(f"   💾 ¡Nuevo Récord ({best_val_f1:.4f})!")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")
        torch.save(final_model.state_dict(), "interrupted_model_optim.pth")
    finally:
        log_file.close()
        if len(hist_train) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(hist_train, label='Train Loss')
            plt.plot(hist_val, label='Val Loss')
            plt.title("HPPNet Optimized Loss")
            plt.legend()
            plt.savefig("loss_curve_optim.png")