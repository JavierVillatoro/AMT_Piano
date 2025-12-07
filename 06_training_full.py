import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SEGMENT_FRAMES = 320 
BATCH_SIZE = 6       # Conservador para tu GTX 1650 (si te sobra VRAM, sube a 8)
LEARNING_RATE = 5e-4 
EPOCHS = 100          # Un poco más largo para que aprenda los matices

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
        # Busca en processed_data_hcqt (o processed_data si usas CQT normal)
        # Asumimos que usas HCQT porque pediste precisión máxima
        base = self.processed_dir / "inputs_cqt"
        if not base.exists(): 
             # Fallback por si acaso
             base = Path("processed_data_hhcqt") / "inputs_cqt"
             self.processed_dir = Path("processed_data_hhcqt")

        all_files = sorted(list(base.glob("*.npy")))
        random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - val_split))
        
        self.files = all_files[:split_idx] if split == 'train' else all_files[split_idx:]
        print(f"🔄 Cargando {len(self.files)} archivos para {split}...")
        
        self.data = []
        for f in tqdm(self.files):
            fid = f.name
            try:
                # Cargamos TODO
                d = {
                    "cqt": np.load(self.processed_dir / "inputs_cqt" / fid).astype(np.float32),
                    "onset": np.load(self.processed_dir / "targets_onset" / fid).astype(np.float32),
                    "frame": np.load(self.processed_dir / "targets_frame" / fid).astype(np.float32),
                    "offset": np.load(self.processed_dir / "targets_offset" / fid).astype(np.float32),
                    "velocity": np.load(self.processed_dir / "targets_velocity" / fid).astype(np.float32)
                }
                self.data.append(d)
            except: continue

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        total_frames = item['cqt'].shape[0]
        
        # Smart Sampling
        for _ in range(10):
            if total_frames > SEGMENT_FRAMES:
                start = np.random.randint(0, total_frames - SEGMENT_FRAMES)
                end = start + SEGMENT_FRAMES
            else:
                start = 0; end = total_frames
            
            # Si hay actividad, nos quedamos con el crop
            if item['frame'][start:end].sum() > 0:
                break
        
        return {
            "cqt": torch.tensor(item['cqt'][start:end]).unsqueeze(0), # [1, T, F] or [1, T, F, 3]
            "onset": torch.tensor(item['onset'][start:end]),
            "frame": torch.tensor(item['frame'][start:end]),
            "offset": torch.tensor(item['offset'][start:end]),
            "velocity": torch.tensor(item['velocity'][start:end])
        }

# ==========================================
# 2. MODELO "FULL STACK" (4 HEADS)
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
    def __init__(self, input_channels=3, hidden_lstm=128):
        super().__init__()
        # input_channels=3 para HCQT, 1 para CQT
        self.enc1 = ConvBlock(input_channels, 16, pool=False) 
        self.enc2 = ConvBlock(16, 32, pool=True)
        self.enc3 = ConvBlock(32, 64, pool=True)
        
        self.lstm_input_size = 64 * 88 
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_lstm, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_lstm * 2, self.lstm_input_size)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.dec3 = ConvBlock(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 1), stride=(2, 1))
        self.dec2 = ConvBlock(32, 16)
        
        # --- 4 CABEZAS DE SALIDA ---
        self.head_onset = nn.Conv2d(16, 1, 1)
        self.head_frame = nn.Conv2d(16, 1, 1)
        self.head_offset = nn.Conv2d(16, 1, 1) # Saber cuándo acaba la nota
        self.head_vel = nn.Conv2d(16, 1, 1)    # Dinámica (volumen)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b, c, t, f = e3.shape
        lstm_in = e3.permute(0, 2, 1, 3).reshape(b, t, c*f)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.dropout(self.relu(lstm_out))
        
        proj = self.lstm_proj(lstm_out).view(b, t, c, f).permute(0, 2, 1, 3)
        
        d3 = self.dec3(torch.cat([self.up3(proj), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        
        # Outputs
        onset = self.head_onset(d2).squeeze(1)
        frame = self.head_frame(d2).squeeze(1)
        offset = self.head_offset(d2).squeeze(1)
        vel = torch.sigmoid(self.head_vel(d2).squeeze(1)) # Vel siempre entre 0 y 1
        
        return onset, frame, offset, vel

# ==========================================
# 3. ENTRENAMIENTO DE ALTA PRECISIÓN
# ==========================================
def train():
    # Detectar carpeta automáticamente
    if Path("processed_data_hcqt").exists():
        data_path = "processed_data_hcqt"
        in_ch = 3
        print("🧠 Modo detectado: HCQT (3 canales)")
    else:
        data_path = "processed_data"
        in_ch = 1
        print("🧠 Modo detectado: CQT Standard (1 canal)")

    train_ds = PianoDataset(data_path, split='train')
    val_ds = PianoDataset(data_path, split='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PianoCRNN(input_channels=in_ch).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()
    
    # Pesos de Loss
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(DEVICE)) # Para Onset/Offset
    criterion_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE)) # Para Frame
    criterion_mse = nn.MSELoss() # Para Velocity
    
    print("\n🚀 Iniciando entrenamiento FULL-STACK...")
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            cqt = batch['cqt'].to(DEVICE)
            # Fix dimensiones si es HCQT
            if in_ch == 3:
                cqt = cqt.squeeze(1).permute(0, 3, 1, 2)
            
            onset_target = batch['onset'].to(DEVICE)
            frame_target = batch['frame'].to(DEVICE)
            offset_target = batch['offset'].to(DEVICE)
            vel_target = batch['velocity'].to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                p_onset, p_frame, p_offset, p_vel = model(cqt)
                
                # 1. Losses Básicas
                l_onset = criterion_bce(p_onset, onset_target)
                l_frame = criterion_frame(p_frame, frame_target)
                l_offset = criterion_bce(p_offset, offset_target)
                
                # 2. MASKED VELOCITY LOSS (Clave para precisión)
                # Solo aprendemos velocity donde realmente hay nota (frame > 0.5)
                mask = frame_target > 0.5
                if mask.sum() > 0:
                    l_vel = criterion_mse(p_vel[mask], vel_target[mask])
                else:
                    l_vel = 0.0
                
                loss = l_onset + l_frame + (l_offset * 0.5) + l_vel

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        if (epoch+1) % 3 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
            f1 = validate(model, val_loader, in_ch)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), "best_piano_full_2.pth")
                print("💾 Modelo Full-Stack guardado.")

def validate(model, loader, in_ch):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            cqt = batch['cqt'].to(DEVICE)
            if in_ch == 3: cqt = cqt.squeeze(1).permute(0, 3, 1, 2)
            
            p_onset, _, _, _ = model(cqt) # Validamos por Onset F1
            all_preds.append(torch.sigmoid(p_onset).cpu().numpy().flatten())
            all_targets.append(batch['onset'].cpu().numpy().flatten())
            
    p_bin = np.concatenate(all_preds) > 0.5
    t_bin = np.concatenate(all_targets) > 0.5
    f1 = f1_score(t_bin, p_bin)
    print(f"   >>> Val Onset F1: {f1:.4f}")
    return f1

if __name__ == "__main__":
    train()