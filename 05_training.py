import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN GENERAL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("processed_data")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# Semilla
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"⚡ Usando dispositivo: {DEVICE}")

# --- 1. DATASET ---
class PianoRollDataset(Dataset):
    def __init__(self, file_list, segment_length=256):
        self.file_list = file_list
        self.segment_length = segment_length
        self.data_cache = []

        if len(file_list) > 0:
            # print(f"📦 Cargando {len(file_list)} archivos...")
            pass
        
        for f_cqt in file_list:
            fname = f_cqt.name
            try:
                cqt = np.load(BASE_DIR / "inputs_cqt" / fname)
                onset = np.load(BASE_DIR / "targets_onset" / fname)
                offset = np.load(BASE_DIR / "targets_offset" / fname)
                frame = np.load(BASE_DIR / "targets_frame" / fname)
                vel = np.load(BASE_DIR / "targets_velocity" / fname)

                self.data_cache.append({
                    "cqt": torch.tensor(cqt, dtype=torch.float32),
                    "onset": torch.tensor(onset, dtype=torch.float32),
                    "offset": torch.tensor(offset, dtype=torch.float32),
                    "frame": torch.tensor(frame, dtype=torch.float32),
                    "vel": torch.tensor(vel, dtype=torch.float32)
                })
            except Exception:
                pass 

    def __len__(self):
        return len(self.data_cache) * 5

    def __getitem__(self, idx):
        track_idx = idx % len(self.data_cache)
        data = self.data_cache[track_idx]
        total_frames = data["cqt"].shape[0]
        
        if total_frames <= self.segment_length:
            start = 0
            real_len = total_frames
            pad_len = self.segment_length - total_frames
        else:
            start = random.randint(0, total_frames - self.segment_length)
            real_len = self.segment_length
            pad_len = 0

        def slice_pad(tensor, start, length, padding):
            sliced = tensor[start : start + length]
            if padding > 0:
                pad = torch.zeros((padding, 88), dtype=sliced.dtype)
                sliced = torch.cat([sliced, pad], dim=0)
            return sliced

        cqt_seg = slice_pad(data["cqt"], start, real_len, pad_len)
        onset_seg = slice_pad(data["onset"], start, real_len, pad_len)
        offset_seg = slice_pad(data["offset"], start, real_len, pad_len)
        frame_seg = slice_pad(data["frame"], start, real_len, pad_len)
        vel_seg = slice_pad(data["vel"], start, real_len, pad_len)

        return cqt_seg.unsqueeze(0), onset_seg, offset_seg, frame_seg, vel_seg

# --- 2. MODELO ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(3,3), pad=(1,1)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    def forward(self, x): return self.conv(x)

class PianoCRNN(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_layers=2):
        super().__init__()
        self.conv1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv3 = ConvBlock(64, 128)
        
        self.rnn_input_dim = 128 * 22 
        self.lstm = nn.LSTM(input_size=self.rnn_input_dim, hidden_size=lstm_hidden_size, 
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.lstm_fc = nn.Linear(lstm_hidden_size * 2, 256)
        self.relu = nn.ReLU()
        
        self.head_onset = nn.Linear(256, 88)
        self.head_offset = nn.Linear(256, 88)
        self.head_frame = nn.Linear(256, 88)
        self.head_velocity = nn.Linear(256, 88)

    def forward(self, x):
        b, c, t, f = x.shape
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x) 
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, -1) 
        x, _ = self.lstm(x)
        x = self.relu(self.lstm_fc(x)) 
        
        return self.head_onset(x), self.head_offset(x), self.head_frame(x), torch.sigmoid(self.head_velocity(x))

# --- 3. METRICAS Y ENTRENAMIENTO ---
def calculate_metrics(pred_logits, targets, threshold=0.4): # Umbral bajado a 0.4 para ayudar al F1
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()
    y_true = targets.cpu().numpy().flatten()
    y_pred = preds.cpu().numpy().flatten()
    return precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)

def train_epoch(model, loader, optimizer, criterion_bce, criterion_mse):
    model.train()
    total_loss = 0
    for cqt, y_onset, y_offset, y_frame, y_vel in loader:
        cqt, y_onset, y_offset, y_frame, y_vel = cqt.to(DEVICE), y_onset.to(DEVICE), y_offset.to(DEVICE), y_frame.to(DEVICE), y_vel.to(DEVICE)
        optimizer.zero_grad()
        p_onset, p_offset, p_frame, p_vel = model(cqt)
        
        loss = criterion_bce(p_onset, y_onset) + criterion_bce(p_offset, y_offset) + \
               criterion_bce(p_frame, y_frame) + criterion_mse(p_vel, y_vel)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion_bce, criterion_mse):
    model.eval()
    total_loss = 0
    all_f1 = []
    with torch.no_grad():
        for cqt, y_onset, y_offset, y_frame, y_vel in loader:
            cqt, y_onset, y_offset, y_frame, y_vel = cqt.to(DEVICE), y_onset.to(DEVICE), y_offset.to(DEVICE), y_frame.to(DEVICE), y_vel.to(DEVICE)
            p_onset, p_offset, p_frame, p_vel = model(cqt)
            loss = criterion_bce(p_onset, y_onset) + criterion_bce(p_offset, y_offset) + \
                   criterion_bce(p_frame, y_frame) + criterion_mse(p_vel, y_vel)
            total_loss += loss.item()
            _, _, f1 = calculate_metrics(p_frame, (y_frame > 0.5).float())
            all_f1.append(f1)
    return total_loss / len(loader), np.mean(all_f1)

def plot_confusion_matrix_binary(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    print("\n📊 Generando Matriz de Confusión...")
    with torch.no_grad():
        for cqt, _, _, y_frame, _ in loader:
            cqt, y_frame = cqt.to(DEVICE), y_frame.to(DEVICE)
            _, _, p_frame, _ = model(cqt)
            probs = torch.sigmoid(p_frame)
            preds = (probs > 0.4).cpu().numpy().flatten() # Umbral consistente
            targets = (y_frame > 0.5).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Silencio', 'Nota'], yticklabels=['Silencio', 'Nota'])
    plt.title('Matriz de Confusión (Frames)')
    plt.ylabel('Realidad'); plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "confusion_matrix.png")
    print(f"   📸 Guardada en: {CHECKPOINT_DIR / 'confusion_matrix.png'}")
    
    p = precision_score(all_targets, all_preds, zero_division=0)
    r = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    print(f"\n🏆 FINAL METRICS: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")

def run_full_training(best_params):
    print("\n" + "="*40 + "\n🚀 INICIANDO ENTRENAMIENTO FINAL (100 EPOCHS)\n" + "="*40)
    
    segment_len = best_params["segment_len"]
    batch_size = best_params["batch_size"]
    lr = best_params["lr"]
    lstm_hidden = best_params["lstm_hidden"]
    
    all_files = list(Path(BASE_DIR / "inputs_cqt").glob("*.npy"))
    if not all_files: return

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=SEED)
    
    print("📦 Cargando datos en RAM...")
    train_ds = PianoRollDataset(train_files, segment_length=segment_len)
    val_ds = PianoRollDataset(val_files, segment_length=segment_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = PianoCRNN(lstm_hidden_size=lstm_hidden).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    crit_bce = nn.BCEWithLogitsLoss()
    crit_mse = nn.MSELoss()
    
    best_f1 = -1.0 # Inicializar en negativo para forzar guardado al menos una vez si es 0
    history = {'loss': [], 'val_loss': [], 'f1': []}
    
    pbar = tqdm(range(1, 101), desc="Training")
    
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, crit_bce, crit_mse)
        val_loss, val_f1 = validate(model, val_loader, crit_bce, crit_mse)
        
        scheduler.step(val_f1)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['f1'].append(val_f1)
        
        pbar.set_postfix({'T_Loss': f"{train_loss:.3f}", 'V_F1': f"{val_f1:.3f}"})
        
        # LOGICA CORREGIDA: Guardar si mejora O si es igual (para atrapar el 0.0 al principio)
        # Y asegurarnos de que el archivo se crea
        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    # CORRECCIÓN FINAL: Si por alguna razón extraña no se guardó, guardamos el último
    if not BEST_MODEL_PATH.exists():
        print("⚠️ Advertencia: F1 muy bajo, guardando último modelo por seguridad.")
        torch.save(model.state_dict(), BEST_MODEL_PATH)
            
    # Graficar
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['f1'], label='Val F1', color='orange')
    plt.title("F1 Score")
    plt.legend()
    plt.savefig(CHECKPOINT_DIR / "training_history.png")
    
    print("\nCargando mejor modelo para evaluación...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    plot_confusion_matrix_binary(model, val_loader)

# --- OPTUNA PLACEHOLDER ---
def objective(trial): return 0.0 # Simplificado

if __name__ == "__main__":
    best_params = {
        "lr": 0.001,
        "batch_size": 16,     
        "lstm_hidden": 256,
        "segment_len": 256    
    }
    run_full_training(best_params)