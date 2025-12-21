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
import sys

# INTENTAR IMPORTAR MIR_EVAL
try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("⚠️ ADVERTENCIA: 'mir_eval' no instalado. Instala con 'pip install mir_eval'")

# ==========================================
# 0. CONFIGURACIÓN Y HIPERPARÁMETROS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# --- CONFIGURACIÓN DE TIEMPO ---
SR = 16000
HOP_LENGTH = 512
FPS = SR / HOP_LENGTH   # 31.25 fps
SEGMENT_FRAMES = 320    # 10.24 segundos
BINS_PER_OCTAVE = 12   
DATA_PATH = Path("processed_data_HPPNET")

# --- Hiperparámetros Fijos ---
LEARNING_RATE = 6e-4    
LSTM_HIDDEN = 48        
ONSET_WEIGHT = 2.0      
BATCH_SIZE = 4          
FINAL_EPOCHS = 50       

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
    def __init__(self, in_channels=3, base_channels=24, lstm_hidden=48):
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
# 3. UTILS (MIR EVAL & METRICS)
# ==========================================
def grid_to_intervals(grid, threshold=0.5):
    """
    Decodifica matriz binaria (88, Time) a notas (start_time, end_time, pitch_midi).
    """
    notes = []
    for pitch in range(88):
        row = grid[pitch, :]
        is_active = row > threshold
        diff = np.diff(is_active.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            start_time = s / FPS
            end_time = e / FPS
            if end_time > start_time:
                notes.append([start_time, end_time, pitch + 21]) 
    return np.array(notes)

def compute_batch_metrics(p_on, p_fr, p_off, p_vel, t_on, t_fr, t_off, t_vel):
    """
    Calcula TODAS las métricas detalladas para un batch.
    Retorna un diccionario con listas de resultados.
    """
    results = {
        'frame_f1': 0, 'frame_p': 0, 'frame_r': 0,
        'onset_f1': 0, 'onset_p': 0, 'onset_r': 0,
        'offset_f1': 0, 'offset_p': 0, 'offset_r': 0,
        'vel_mse': 0
    }
    
    # 1. Pixel-wise Metrics (Frames)
    p_fr_np = torch.sigmoid(p_fr).cpu().numpy()
    t_fr_np = t_fr.cpu().numpy()
    
    p_bin = p_fr_np.flatten() > 0.5
    t_bin = t_fr_np.flatten() > 0.5
    
    results['frame_f1'] = f1_score(t_bin, p_bin, zero_division=0)
    results['frame_p'] = precision_score(t_bin, p_bin, zero_division=0)
    results['frame_r'] = recall_score(t_bin, p_bin, zero_division=0)
    
    # 2. Velocity MSE
    p_vel_sig = torch.sigmoid(p_vel)
    mask = t_fr > 0.5
    if mask.sum() > 0:
        results['vel_mse'] = mean_squared_error(t_vel[mask].cpu().numpy(), p_vel_sig[mask].cpu().numpy())
    else:
        results['vel_mse'] = 0.0

    # 3. MIR_EVAL Metrics (Note-Level)
    if HAS_MIR_EVAL:
        batch_size = p_fr_np.shape[0]
        # Listas temporales para promediar el batch
        on_f1, on_p, on_r = [], [], []
        off_f1, off_p, off_r = [], [], []
        
        for i in range(batch_size):
            est_notes = grid_to_intervals(p_fr_np[i])
            ref_notes = grid_to_intervals(t_fr_np[i])
            
            # Casos bordes (sin notas)
            if len(ref_notes) == 0 and len(est_notes) == 0: continue
            if len(ref_notes) == 0: 
                on_f1.append(0); on_p.append(0); on_r.append(0)
                off_f1.append(0); off_p.append(0); off_r.append(0)
                continue
            if len(est_notes) == 0:
                on_f1.append(0); on_p.append(0); on_r.append(0)
                off_f1.append(0); off_p.append(0); off_r.append(0)
                continue
                
            ref_int = ref_notes[:, 0:2]
            ref_pit = ref_notes[:, 2]
            est_int = est_notes[:, 0:2]
            est_pit = est_notes[:, 2]
            
            # --- CORRECCIÓN AQUÍ: AÑADIDO ", _" PARA IGNORAR EL 4º VALOR ---
            
            # A. ONSET
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_int, ref_pit, est_int, est_pit, 
                onset_tolerance=0.05, offset_ratio=None
            )
            on_p.append(p); on_r.append(r); on_f1.append(f)
            
            # B. OFFSET
            p_off_val, r_off_val, f_off_val, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_int, ref_pit, est_int, est_pit, 
                onset_tolerance=0.05, offset_ratio=0.2
            )
            off_p.append(p_off_val); off_r.append(r_off_val); off_f1.append(f_off_val)
            
        # Promedios del batch
        if on_f1:
            results['onset_f1'] = np.mean(on_f1); results['onset_p'] = np.mean(on_p); results['onset_r'] = np.mean(on_r)
            results['offset_f1'] = np.mean(off_f1); results['offset_p'] = np.mean(off_p); results['offset_r'] = np.mean(off_r)
        
    return results

def evaluate_and_plot(model, loader, filename_suffix="HPPNET"):
    model.eval()
    metrics_plot = {k: {'preds': [], 'targets': []} for k in ['onset', 'frame', 'offset', 'velocity']}
    
    print(f"\n📊 Generando Gráficas ({filename_suffix})...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Eval Plot")):
            if i > 50: break 
            hcqt = batch['hcqt'].to(DEVICE)
            p_on, p_fr, p_off, p_vel = model(hcqt)
            
            metrics_plot['onset']['preds'].append(torch.sigmoid(p_on).cpu().numpy().flatten())
            metrics_plot['onset']['targets'].append(batch['onset'].numpy().flatten())
            metrics_plot['frame']['preds'].append(torch.sigmoid(p_fr).cpu().numpy().flatten())
            metrics_plot['frame']['targets'].append(batch['frame'].numpy().flatten())
            metrics_plot['offset']['preds'].append(torch.sigmoid(p_off).cpu().numpy().flatten())
            metrics_plot['offset']['targets'].append(batch['offset'].numpy().flatten())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tasks = ['onset', 'frame', 'offset']
    
    print("\n" + "="*60)
    print(f"📢 REPORTE GRÁFICO FINAL")
    print("="*60)

    for i, task in enumerate(tasks):
        p_bin = np.concatenate(metrics_plot[task]['preds']) > 0.4
        t_bin = np.concatenate(metrics_plot[task]['targets']) > 0.5
        f1 = f1_score(t_bin, p_bin, zero_division=0)
        
        cm = confusion_matrix(t_bin, p_bin)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"{task.upper()} (Pixel) F1: {f1:.2f}")

    plt.tight_layout()
    plt.savefig(f"metrics_{filename_suffix}.png")
    print(f"✅ Gráfica guardada: metrics_{filename_suffix}.png")

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    print(f"\n🚀 HPPNET Training Start ({DEVICE})")
    print(f"⚙️ Params: Batch={BATCH_SIZE} | 10s Crops | LSTM={LSTM_HIDDEN}")
    if HAS_MIR_EVAL: print("✅ mir_eval activo")
    
    print("\n📂 Inicializando Datasets...")
    full_train_ds = PianoDataset(DATA_PATH, split='train')
    full_val_ds = PianoDataset(DATA_PATH, split='val')
    print(f"✅ Clips Train: {len(full_train_ds)} | Clips Val: {len(full_val_ds)}")

    print("\n🔄 Iniciando Entrenamiento...")
    final_model = HPPNet(in_channels=3, lstm_hidden=LSTM_HIDDEN).to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') 
    
    # Loss
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ONSET_WEIGHT]).to(DEVICE))
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(DEVICE)) 
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))
    crit_vel = nn.MSELoss(reduction='none') 
    
    train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(full_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # CSV Logging
    log_file = open("training_log_final_48.csv", "w")
    # Header completo con P/R
    header = "epoch,train_loss,val_loss,"
    header += "onset_f1,onset_p,onset_r,"
    header += "offset_f1,offset_p,offset_r,"
    header += "frame_f1,frame_p,frame_r,"
    header += "vel_mse\n"
    log_file.write(header)
    
    hist_train, hist_val = [], []
    best_onset_f1 = 0.0
    
    try:
        for epoch in range(FINAL_EPOCHS):
            final_model.train()
            t_loss = 0
            
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

            # --- VALIDATION (Full Metrics) ---
            final_model.eval()
            v_loss = 0
            
            # Diccionario para acumular métricas de toda la época
            epoch_metrics = {k: [] for k in ['frame_f1', 'frame_p', 'frame_r', 
                                             'onset_f1', 'onset_p', 'onset_r',
                                             'offset_f1', 'offset_p', 'offset_r', 'vel_mse']}

            with torch.no_grad():
                for batch in val_loader:
                    hcqt = batch['hcqt'].to(DEVICE)
                    targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'hcqt'}
                    
                    p_on, p_fr, p_off, p_vel = final_model(hcqt)
                    
                    # Calc Loss
                    l_on = crit_onset(p_on, targets['onset'])
                    l_fr = crit_frame(p_fr, targets['frame'])
                    l_off = crit_offset(p_off, targets['offset'])
                    vel_loss_raw = crit_vel(torch.sigmoid(p_vel), targets['velocity'])
                    mask = targets['frame']
                    l_vel = (vel_loss_raw * mask).sum() / (mask.sum() + 1e-6)
                    v_loss += (l_on + l_fr + l_off + l_vel).item()
                    
                    # Calc Detailed Metrics
                    batch_res = compute_batch_metrics(
                        p_on, p_fr, p_off, p_vel,
                        targets['onset'], targets['frame'], targets['offset'], targets['velocity']
                    )
                    
                    for k, v in batch_res.items():
                        epoch_metrics[k].append(v)

            # Promedios de la época
            avg_t = t_loss/len(train_loader)
            avg_v = v_loss/len(val_loader)
            hist_train.append(avg_t)
            hist_val.append(avg_v)
            
            # Extraer promedios limpios
            M = {k: np.mean(v) for k, v in epoch_metrics.items()}

            # --- PRINT REPORT ---
            print(f"\n✅ Ep {epoch+1} | T.Loss: {avg_t:.4f} | V.Loss: {avg_v:.4f}")
            print(f"   🎹 MIR ONSET  -> F1: {M['onset_f1']:.4f} | P: {M['onset_p']:.4f} | R: {M['onset_r']:.4f}")
            print(f"   🎹 MIR OFFSET -> F1: {M['offset_f1']:.4f} | P: {M['offset_p']:.4f} | R: {M['offset_r']:.4f}")
            print(f"   🖼️ FRAME (Px) -> F1: {M['frame_f1']:.4f} | P: {M['frame_p']:.4f} | R: {M['frame_r']:.4f}")
            print(f"   📉 VEL MSE    -> {M['vel_mse']:.5f}")

            # CSV Log
            csv_str = f"{epoch+1},{avg_t:.4f},{avg_v:.4f},"
            csv_str += f"{M['onset_f1']:.4f},{M['onset_p']:.4f},{M['onset_r']:.4f},"
            csv_str += f"{M['offset_f1']:.4f},{M['offset_p']:.4f},{M['offset_r']:.4f},"
            csv_str += f"{M['frame_f1']:.4f},{M['frame_p']:.4f},{M['frame_r']:.4f},"
            csv_str += f"{M['vel_mse']:.5f}\n"
            log_file.write(csv_str)
            log_file.flush()

            # Guardar Mejor Modelo (Basado en MIR ONSET F1, lo más importante)
            if M['onset_f1'] > best_onset_f1:
                best_onset_f1 = M['onset_f1']
                torch.save(final_model.state_dict(), "best_model_HPPNET_48.pth")
                print(f"   💾 ¡Nuevo Récord ({best_onset_f1:.4f})! Modelo guardado.")

    except KeyboardInterrupt:
        print("\n🛑 Detenido por usuario.")
        torch.save(final_model.state_dict(), "interrupted_model_48.pth")
    finally:
        log_file.close()

    # Plots Finales
    if len(hist_train) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(hist_train, label='Train')
        plt.plot(hist_val, label='Val')
        plt.title("HPPNet Training Loss")
        plt.legend(); plt.grid(True)
        plt.savefig("loss_curve_final.png")
        
        if os.path.exists("best_model_HPPNET_48.pth"):
            print("🔄 Cargando mejor modelo para plots finales...")
            final_model.load_state_dict(torch.load("best_model_HPPNET_48.pth"))
        evaluate_and_plot(final_model, val_loader, filename_suffix="FINAL")