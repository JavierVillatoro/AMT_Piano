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
HOP_LENGTH = 512     
SEGMENT_FRAMES = 320 
BINS_PER_OCTAVE = 12  
# Nota: En Kaggle working directory es donde descomprimimos
DATA_PATH = Path("/kaggle/input/maestro2/processed_data_HPPNET_4_h") 

# Hyperparámetros Optimizados para Kaggle
BATCH_SIZE = 32           # Subido de 4 a 32 para aprovechar GPU P100/T4
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
            
            hcqt_t = torch.tensor(hcqt).permute(2, 1, 0).float()
            
            return {
                "hcqt": hcqt_t,
                "onset": torch.tensor(onset).float(),
                "frame": torch.tensor(frame).float(),
                "offset": torch.tensor(offset).float(),
                "velocity": torch.tensor(vel).float()
            }
        except Exception as e:
            print(f"Error loading {fid}: {e}")
            z = torch.zeros(SEGMENT_FRAMES, 88)
            return {"hcqt": torch.zeros(3, 88, SEGMENT_FRAMES), "onset": z, "frame": z, "offset": z, "velocity": z}

# ==========================================
# 2. ARQUITECTURA FINAL (RESIDUAL + HDCONV + INSTANCENORM)
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

# --- NUEVO: LOSSES AVANZADAS ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Aplicamos Sigmoid porque la salida del modelo son Logits
        inputs = torch.sigmoid(inputs)
        
        # Aplanar tensores para calcular intersección global
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class ComboLoss(nn.Module):
    def __init__(self, bce_weight=5.0):
        super(ComboLoss, self).__init__()
        # BCE se encarga del Recall (el peso alto fuerza a encontrar notas)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_weight]))
        # Dice se encarga de la Precisión (la forma exacta)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        # Sumamos ambas: BCE mantiene el Recall alto, Dice limpia la suciedad
        return self.bce(inputs, targets) + self.dice(inputs, targets)

# --- NUEVO: Bloque Residual ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_c, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
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
        out += identity  # <--- RESIDUAL CONNECTION
        out = self.relu(out)
        return out

# --- HDConv Corregida (Sin dilatación negativa) ---
class HDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        # AÑADIDO: Armónicos 5 y 6 para mayor definición de timbre
        harmonics = [1, 2, 3, 4, 5, 6] 
        
        for h in harmonics:
            if h == 1:
                d = 1 
            else:
                # Cuidado: log2(h) debe dar un entero razonable. 
                # Para 3, 5, 6 redondeamos.
                d = int(np.round(BINS_PER_OCTAVE * np.log2(h)))
            
            # Padding para mantener dimensiones: (d, 1)
            self.convs.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=(3, 3), 
                padding=(d, 1), 
                dilation=(d, 1)
            ))
            
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        # Sumamos todas las convoluciones armónicas
        x_sum = sum([conv(x) for conv in self.convs])
        return self.fusion(x_sum)

class AcousticModel(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        # Entrada: Adapta HCQT (3 canales) a Base Channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU()
        )
        
        # Bloques Residuales para profundizar sin perder señal
        self.res1 = ResidualBlock(base_channels, base_channels)
        self.res2 = ResidualBlock(base_channels, base_channels)
        
        # Visión Armónica
        self.hdc = HDConv(base_channels, base_channels)
        self.hdc_bn = nn.InstanceNorm2d(base_channels, affine=True)
        self.hdc_relu = nn.ReLU()
        
        # Contexto (Más residuales)
        self.context = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        
        # Sumamos la visión armónica (Residual Style)
        x_hdc = self.hdc(x)
        x_hdc = self.hdc_relu(self.hdc_bn(x_hdc))
        x = x + x_hdc 
        
        x = self.context(x)
        return x

class FG_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # LSTM Bidireccional
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        b, c, f, t = x.shape
        # Permutar para LSTM (Batch*Freq, Time, Channels)
        x = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        output = self.proj(output)
        output = output.view(b, f, t)
        return output.permute(0, 2, 1) 

class HPPNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=24, lstm_hidden=128):
        super().__init__()
        self.acoustic_onset = AcousticModel(in_channels, base_channels)
        self.acoustic_other = AcousticModel(in_channels, base_channels)
        
        self.head_onset = FG_LSTM(base_channels, lstm_hidden)
        
        concat_dim = base_channels * 2
        self.head_frame = FG_LSTM(concat_dim, lstm_hidden)
        
        # El offset recibe (Features + Predicción Frame) = concat_dim + 1
        self.head_offset = FG_LSTM(concat_dim + 1, lstm_hidden) 
        
        self.head_velocity = FG_LSTM(concat_dim, lstm_hidden)

    def forward(self, x):
        feat_onset = self.acoustic_onset(x)
        logits_onset = self.head_onset(feat_onset)
        
        feat_onset_detached = feat_onset.detach()
        feat_other = self.acoustic_other(x)
        feat_combined = torch.cat([feat_other, feat_onset_detached], dim=1)
        
        logits_frame = self.head_frame(feat_combined)
        
        # --- CORRECCIÓN DE DIMENSIONES (Tu código) ---
        # 1. Obtenemos probabilidad y quitamos gradiente
        prob_frame = torch.sigmoid(logits_frame).detach() # [Batch, Time, Freq]
        
        # 2. Permutamos para que Time quede al final: [Batch, Freq, Time]
        prob_frame = prob_frame.permute(0, 2, 1)
        
        # 3. Añadimos dimensión de canal: [Batch, 1, Freq, Time]
        prob_frame = prob_frame.unsqueeze(1)
        
        # 4. Concatenamos: [B, C, F, T] + [B, 1, F, T] -> [B, C+1, F, T]
        feat_offset_in = torch.cat([feat_combined, prob_frame], dim=1)
        
        logits_offset = self.head_offset(feat_offset_in)
        logits_velocity = self.head_velocity(feat_combined)
        
        return logits_onset, logits_frame, logits_offset, logits_velocity
    

# ==========================================
# 3. UTILS (CON PEAK PICKING)
# ==========================================
def tensor_to_notes(onset_pred, frame_pred, offset_pred, velocity_pred=None, t_onset=0.35, t_frame=0.6, t_offset=0.4):
    """
    Decodificación con condicionamiento explícito de Offset.
    Nota: He añadido offset_pred a los argumentos.
    """
    notes = []
    for pitch in range(88):
        # 1. Peak Picking (Igual que antes)
        peaks, _ = find_peaks(onset_pred[:, pitch], height=t_onset, distance=2)
        
        for onset_frame in peaks:
            # 2. Validación de Sustain (Igual que antes)
            check_frame = min(onset_frame + 1, frame_pred.shape[0] - 1)
            if frame_pred[check_frame, pitch] < t_frame:
                continue 
                
            # 3. Buscar Offset (MEJORADO)
            end_frame = onset_frame + 1
            # Buscamos hasta que se acabe el frame O encontremos un offset fuerte
            while end_frame < frame_pred.shape[0]:
                # Condición A: El frame sigue activo
                frame_active = frame_pred[end_frame, pitch] > t_frame
                
                # Condición B: NO hay un offset fuerte en este punto
                # (Si offset > t_offset, is_offset_hit es True, y paramos el loop)
                is_offset_hit = offset_pred[end_frame, pitch] > t_offset
                
                if frame_active and not is_offset_hit:
                    end_frame += 1
                else:
                    # Hemos encontrado el final (ya sea por caída de frame o por presencia de offset)
                    break
            
            # Ajuste fino: Si paramos por un Offset, incluimos ese frame como el final
            if end_frame < frame_pred.shape[0] and offset_pred[end_frame, pitch] > t_offset:
                 end_frame += 1

            # 4. Filtro duración (Igual)
            if end_frame - onset_frame > 2:
                onset_time = onset_frame * HOP_LENGTH / SR
                offset_time = end_frame * HOP_LENGTH / SR
                
                # 5. Velocity (Igual)
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

        # Para mir_eval, columnas 0 y 1 son tiempos, columna 2 es pitch
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
    """
    Calcula métricas ESTÁNDAR (mir_eval) para Note Onset + Offset.
    Requiere que tanto el inicio como el final sean correctos.
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    for ref_notes, est_notes in zip(ref_notes_batch, est_notes_batch):
        ref_arr = np.array(ref_notes)
        est_arr = np.array(est_notes)
        
        # Casos vacíos
        if len(ref_arr) == 0 and len(est_arr) == 0: continue
        if len(ref_arr) == 0:
            total_fp += len(est_arr); continue
        if len(est_arr) == 0:
            total_fn += len(ref_arr); continue

        ref_int, ref_p = ref_arr[:, :2], ref_arr[:, 2]
        est_int, est_p = est_arr[:, :2], est_arr[:, 2]
        
        # Métrica estricta: Onset (50ms) + Offset (20% duración o 50ms)
        matched = mir_eval.transcription.match_notes(
            ref_int, ref_p, est_int, est_p, 
            onset_tolerance=0.05, 
            offset_ratio=0.2, # <--- ESTO ES LO NUEVO
            offset_min_tolerance=0.05
        )
        
        tp = len(matched)
        total_tp += tp
        total_fp += (len(est_p) - tp)
        total_fn += (len(ref_p) - tp)
        
    p = total_tp / (total_tp + total_fp + 1e-8)
    r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r

def plot_training_history(csv_path="training_log_kaggle.csv"):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('HPPNet Optimized (Residual + PeakPicking) Metrics', fontsize=16)

    sns.lineplot(data=df, x='epoch', y='train_loss', label='Train', ax=axes[0,0])
    sns.lineplot(data=df, x='epoch', y='val_loss', label='Val', ax=axes[0,0], linestyle='--')
    axes[0,0].set_title('Loss')

    sns.lineplot(data=df, x='epoch', y='onset_f1', label='F1', ax=axes[0,1], color='g')
    sns.lineplot(data=df, x='epoch', y='onset_p', label='Precision', ax=axes[0,1], linestyle=':', alpha=0.6)
    sns.lineplot(data=df, x='epoch', y='onset_r', label='Recall', ax=axes[0,1], linestyle=':', alpha=0.6)
    axes[0,1].set_title('Onset Metrics')

    sns.lineplot(data=df, x='epoch', y='frame_f1', label='Frame F1', ax=axes[1,0])
    #sns.lineplot(data=df, x='epoch', y='offset_f1', label='Offset F1', ax=axes[1,0], color='orange')
    sns.lineplot(data=df, x='epoch', y='offset_f1', label='Offset Pix', ax=axes[1,0], color='orange', alpha=0.4, linestyle='--')
    sns.lineplot(data=df, x='epoch', y='note_off_f1', label='Offset Note (Real)', ax=axes[1,0], color='red', linewidth=2)
    axes[1,0].set_title('Frame & Offset F1')

    sns.lineplot(data=df, x='epoch', y='velocity_mse', color='purple', ax=axes[1,1])
    axes[1,1].set_title('Velocity MSE')

    plt.tight_layout()
    plt.savefig("training_results_kaggle.png", dpi=200)
    print("📊 Gráficas guardadas.")

# ==========================================
# 4. MAIN (ESTRUCTURA ORIGINAL OPTIMIZADA)
# ==========================================
if __name__ == "__main__":
    print(f"\n🚀 HPPNET-SP FINAL KAGGLE Training ({DEVICE})")
    print(f"🔹 Config: Batch Size {BATCH_SIZE} | T4x2 (DataParallel) | Partial Val | Real-time Plotting")
    
    # 1. Cargar Datos
    train_ds = PianoDataset(DATA_PATH, split='train')
    val_ds = PianoDataset(DATA_PATH, split='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 2. Inicializar Modelo
    model = HPPNet(in_channels=4, lstm_hidden=128).to(DEVICE)   #ANTES 3 , AHORA CAMBIA 
    
    # --- CAMBIO: ACTIVAR MULTI-GPU ---
    # Esto reparte el Batch Size 32 entre las dos tarjetas (16 y 16)
    if torch.cuda.device_count() > 1:
        print(f"🔥 ¡Activando Turbo! Usando {torch.cuda.device_count()} GPUs en DataParallel")
        model = nn.DataParallel(model)
    # ---------------------------------

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=FACTOR_LR, patience=PATIENCE_LR)
    scaler = torch.amp.GradScaler('cuda') 
    
    # 3. Losses
    crit_onset = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)
    crit_frame = ComboLoss(bce_weight=2.0).to(DEVICE)
    #crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(DEVICE))
    crit_offset = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)
    crit_vel = nn.MSELoss(reduction='none')

    # 4. Logs
    log_file = open("training_log_kaggle.csv", "w")
    header = "epoch,train_loss,val_loss,onset_f1,onset_p,onset_r,frame_f1,frame_p,frame_r,offset_f1,offset_p,offset_r,velocity_mse,lr,note_off_f1,note_off_p,note_off_r\n"
    log_file.write(header)
    log_file.flush()
    
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
                        
                        loss = (10.0 * l_on) + l_fr + (10.0 * l_off) + l_vel
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    t_loss += loss.item()
                    bar.set_postfix(loss=loss.item())
            
            avg_t_loss = t_loss / len(train_loader)

            # --- ESTRATEGIA: VALIDAR CADA 3 EPOCAS ---
            should_validate = ((epoch + 1) % 3 == 0) or ((epoch + 1) == FINAL_EPOCHS)

            if should_validate:
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
                        
                        v_loss += ((10.0 * l_on) + l_fr + (10.0 * l_off) + l_vel).item()
                        
                        pr_on = torch.sigmoid(p_on)
                        pr_fr = torch.sigmoid(p_fr)
                        pr_off = torch.sigmoid(p_off)
                        
                        # Notes decoding
                        for i in range(len(hcqt)):
                            v_map = torch.sigmoid(p_vel[i]).cpu().numpy()
                            est = tensor_to_notes(pr_on[i].cpu().numpy(), pr_fr[i].cpu().numpy(),pr_off[i].cpu().numpy(), v_map, t_onset=THRESHOLD_ONSET, t_frame=THRESHOLD_FRAME,t_offset=THRESHOLD_OFFSET)
                            
                            ref = []
                            ref_on = targets['onset'][i].cpu().numpy()
                            ref_fr = targets['frame'][i].cpu().numpy()
                            for pitch in range(88):
                                ons = np.where(ref_on[:, pitch] > 0.5)[0]
                                for o in ons:
                                    e = o + 1
                                    while e < ref_fr.shape[0] and ref_fr[e, pitch] > 0.5: e += 1
                                    if e - o > 1: ref.append([o*HOP_LENGTH/SR, e*HOP_LENGTH/SR, pitch+21])
                            est_all.append(est)
                            ref_all.append(ref)
                        
                        # Pixel-wise decoding
                        fr_preds.append((pr_fr > THRESHOLD_FRAME).cpu().numpy().flatten())
                        fr_targs.append((targets['frame'] > 0.5).cpu().numpy().flatten())
                        
                        off_preds.append((pr_off > THRESHOLD_OFFSET).cpu().numpy().flatten())
                        off_targs.append((targets['offset'] > 0.5).cpu().numpy().flatten())
                        
                        v_p = torch.sigmoid(p_vel).cpu().numpy().flatten()
                        v_t = targets['velocity'].cpu().numpy().flatten()
                        m = mask.cpu().numpy().flatten().astype(bool)
                        if m.sum() > 0:
                            vel_accum += mean_squared_error(v_t[m], v_p[m]) * m.sum()
                            vel_count += m.sum()

                # --- METRICS ---
                avg_v_loss = v_loss / len(val_loader)
                
                # 1. Metrica onset
                onset_f1, onset_p, onset_r = compute_metrics_standard(ref_all, est_all)
                
                # 2. NUEVA: Métrica de Note Onset+Offset (Estándar)
                note_off_f1, note_off_p, note_off_r = compute_note_offset_metrics(ref_all, est_all)
                
                # 3. Tus métricas pixel-wise (Frame)
                f_p = np.concatenate(fr_preds); f_t = np.concatenate(fr_targs)
                frame_f1 = f1_score(f_t, f_p, zero_division=0)
                frame_p = precision_score(f_t, f_p, zero_division=0)
                frame_r = recall_score(f_t, f_p, zero_division=0)
                
                # 4. Tu métrica pixel-wise (Offset)
                o_p = np.concatenate(off_preds); o_t = np.concatenate(off_targs)
                offset_f1 = f1_score(o_t, o_p, zero_division=0)
                offset_p = precision_score(o_t, o_p, zero_division=0)
                offset_r = recall_score(o_t, o_p, zero_division=0)
                
                vel_mse = vel_accum / (vel_count + 1e-8)
                curr_lr = optimizer.param_groups[0]['lr']

                print("-" * 80)
                print(f"🏁 Epoch {epoch+1} Results:")
                print(f"   📉 Loss    : Train={avg_t_loss:.4f} | Val={avg_v_loss:.4f}")
                print(f"   🎹 Onset   : F1={onset_f1:.4f} | P={onset_p:.4f} | R={onset_r:.4f}")
                print(f"   🖼️ Frame   : F1={frame_f1:.4f} | P={frame_p:.4f} | R={frame_r:.4f}")
                # AQUI ESTÁ LA METRICA QUE FALTABA
                print(f"   🏁 Offset  : F1={offset_f1:.4f} | P={offset_p:.4f} | R={offset_r:.4f}")
                print(f"   ⏱️ Note Offset : F1={note_off_f1:.4f} | P={note_off_p:.4f} | R={note_off_r:.4f}")
                print(f"   ⚡ Vel     : MSE={vel_mse:.4f}")
                print(f"   🧠 LR      : {curr_lr:.2e}")
                print("-" * 80)
                
                log_line = f"{epoch+1},{avg_t_loss},{avg_v_loss},{onset_f1},{onset_p},{onset_r},{frame_f1},{frame_p},{frame_r},{offset_f1},{offset_p},{offset_r},{vel_mse},{curr_lr},{note_off_f1},{note_off_p},{note_off_r}\n"
                
                scheduler.step(onset_f1)
                
                if onset_f1 > best_f1:
                    best_f1 = onset_f1
                    # Guardado seguro con DataParallel
                    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save(state_dict, "best_hppnet_kaggle.pth")
                    print(f"   💾 Nuevo Récord! Modelo guardado (F1: {best_f1:.4f})")

                # --- GUARDAR GRÁFICA EN TIEMPO REAL ---
                log_file.write(log_line)
                log_file.flush()
                # Llamamos a plot aquí para que se actualice la imagen png cada vez que validamos
                plot_training_history("training_log_kaggle.csv")
                print("   📊 Gráfica actualizada.")

            else:
                # --- NO VALIDAR (Ahorro de tiempo) ---
                print(f"⏩ Epoch {epoch+1}: Train Loss={avg_t_loss:.4f} (Validación saltada)")
                # Log con huecos vacíos para mantener formato CSV
                log_line = f"{epoch+1},{avg_t_loss},,,,,,,,,,,,{optimizer.param_groups[0]['lr']}\n"
                log_file.write(log_line)
                log_file.flush()
                
                # Guardar backup simple
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, "latest_checkpoint.pth")
    
    except KeyboardInterrupt:
            print("\n🛑 Entrenamiento detenido.")
            
    finally:
        log_file.close()
            # Asegurar gráfica final
        plot_training_history("training_log_kaggle.csv")