import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SEGMENT_FRAMES = 320 
BATCH_SIZE = 8       # Probamos subir un poco si te cabe en VRAM, si no baja a 6
LEARNING_RATE = 5e-4 # Un poco más fino
EPOCHS = 30          

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ==========================================
# 1. DATASET CON "SMART SAMPLING" (Anti-Silencio)
# ==========================================
class PianoDataset(Dataset):
    def __init__(self, processed_dir, split='train', val_split=0.15):
        self.processed_dir = Path(processed_dir)
        all_files = sorted(list((self.processed_dir / "inputs_cqt").glob("*.npy")))
        random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - val_split))
        
        self.files = all_files[:split_idx] if split == 'train' else all_files[split_idx:]
        print(f"🔄 Cargando {len(self.files)} archivos para {split}...")
        
        self.data = []
        for f in tqdm(self.files):
            fid = f.name
            try:
                # Cargar datos
                cqt = np.load(self.processed_dir / "inputs_cqt" / fid).astype(np.float32)
                onset = np.load(self.processed_dir / "targets_onset" / fid).astype(np.float32)
                
                # OPTIMIZACIÓN: Solo guardamos lo necesario para entrenar Onset/Frame
                # (Offset y Velocity los ignoramos temporalmente para ahorrar RAM)
                frame = np.load(self.processed_dir / "targets_frame" / fid).astype(np.float32)
                
                self.data.append({"cqt": cqt, "onset": onset, "frame": frame})
            except: continue

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        total_frames = item['cqt'].shape[0]
        
        # --- SMART SAMPLING ---
        # Intentamos hasta 10 veces encontrar un trozo que NO sea silencio puro
        for _ in range(10):
            if total_frames > SEGMENT_FRAMES:
                start = np.random.randint(0, total_frames - SEGMENT_FRAMES)
                end = start + SEGMENT_FRAMES
            else:
                start = 0; end = total_frames
            
            # Recorte temporal
            onset_crop = item['onset'][start:end]
            frame_crop = item['frame'][start:end]
            
            # Si hay al menos UNA nota (actividad > 0), nos vale este trozo
            if frame_crop.sum() > 0 or onset_crop.sum() > 0:
                break
            # Si es silencio total, el bucle repite y busca otro trozo
        
        # Preparar tensores
        cqt_crop = item['cqt'][start:end]
        
        return {
            "cqt": torch.tensor(cqt_crop).unsqueeze(0),       # [1, T, 88]
            "onset": torch.tensor(onset_crop),
            "frame": torch.tensor(frame_crop)
        }

# ==========================================
# 2. MODELO CRNN (Igual que antes)
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
    def __init__(self, hidden_lstm=128):
        super().__init__()
        self.enc1 = ConvBlock(1, 16, pool=False) 
        self.enc2 = ConvBlock(16, 32, pool=True)
        self.enc3 = ConvBlock(32, 64, pool=True)
        
        self.lstm_input_size = 64 * 88 
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_lstm, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_lstm * 2, self.lstm_input_size)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.dec3 = ConvBlock(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 1), stride=(2, 1))
        self.dec2 = ConvBlock(32, 16)
        
        self.head_onset = nn.Conv2d(16, 1, 1)
        self.head_frame = nn.Conv2d(16, 1, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25) # Dropout un poco más bajo

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
        
        return self.head_onset(d2).squeeze(1), self.head_frame(d2).squeeze(1)

# ==========================================
# 3. ENTRENAMIENTO EQUILIBRADO
# ==========================================
def train():
    processed_path = "processed_data_hcqt"
    train_ds = PianoDataset(processed_path, split='train')
    val_ds = PianoDataset(processed_path, split='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PianoCRNN().to(DEVICE)
    
    # Weight Decay ayuda a reducir Falsos Positivos (castiga pesos muy locos)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()
    
    # --- AJUSTE FINO DE PESOS ---
    # Bajamos de 20.0 a 10.0 para reducir Falsos Positivos
    onset_weight = torch.tensor([10.0]).to(DEVICE) 
    frame_weight = torch.tensor([5.0]).to(DEVICE)
    
    criterion_onset = nn.BCEWithLogitsLoss(pos_weight=onset_weight)
    criterion_frame = nn.BCEWithLogitsLoss(pos_weight=frame_weight)
    
    print("\n🚀 Iniciando entrenamiento con Smart Sampling...")
    
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            cqt = batch['cqt'].to(DEVICE)
            onset_target = batch['onset'].to(DEVICE)
            frame_target = batch['frame'].to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                pred_onset, pred_frame = model(cqt)
                l_onset = criterion_onset(pred_onset, onset_target)
                l_frame = criterion_frame(pred_frame, frame_target)
                loss = l_onset + l_frame

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        # Validación
        if (epoch+1) % 3 == 0: # Validar cada 3 épocas para ir más rápido
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
            current_f1 = evaluate_and_find_threshold(model, val_loader, epoch)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                torch.save(model.state_dict(), "best_piano_crnn_smart.pth")
                print("💾 Nuevo mejor modelo guardado.")

def evaluate_and_find_threshold(model, loader, epoch):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            cqt = batch['cqt'].to(DEVICE)
            pred_onset, _ = model(cqt)
            pred_prob = torch.sigmoid(pred_onset)
            
            all_preds.append(pred_prob.cpu().numpy().flatten())
            all_targets.append(batch['onset'].cpu().numpy().flatten())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets) > 0.5
    
    # Mostramos métricas para Threshold 0.5 (estándar)
    p_bin = preds > 0.5
    f1 = f1_score(targets, p_bin)
    prec = precision_score(targets, p_bin)
    rec = recall_score(targets, p_bin)
    
    print(f"   >>> Val (Th=0.5) F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    
    # Guardar matriz de confusión de la última época
    if epoch == EPOCHS - 1:
        cm = confusion_matrix(targets, p_bin)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix Smart (F1: {f1:.2f})")
        plt.savefig("confusion_matrix_smart.png")
        plt.close()
        
    return f1

if __name__ == "__main__":
    train()