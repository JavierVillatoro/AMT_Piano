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
# 0. CONFIGURACIÓN GLOBAL
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SEGMENT_FRAMES = 320 
PROCESSED_DIR = "processed_data_full_dataset" 

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
        # Buscar archivos
        all_files = sorted(list((self.processed_dir / "inputs_cqt").glob("*.npy")))
        
        # Mezclar siempre igual para consistencia
        random.Random(SEED).shuffle(all_files)
        
        # Dividir
        split_idx = int(len(all_files) * (1 - val_split))
        self.files = all_files[:split_idx] if split == 'train' else all_files[split_idx:]
        
        print(f"🔄 Cargando {len(self.files)} archivos para {split}...")
        
        # Cargar en RAM
        self.data = []
        for f in tqdm(self.files):
            fid = f.name
            try:
                self.data.append({
                    "cqt": np.load(self.processed_dir / "inputs_cqt" / fid).astype(np.float32), 
                    "onset": np.load(self.processed_dir / "targets_onset" / fid).astype(np.float32),
                    "frame": np.load(self.processed_dir / "targets_frame" / fid).astype(np.float32),
                    "offset": np.load(self.processed_dir / "targets_offset" / fid).astype(np.float32),
                    "velocity": np.load(self.processed_dir / "targets_velocity" / fid).astype(np.float32)
                })
            except Exception as e:
                continue

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        total_frames = item['cqt'].shape[0]
        
        # Random Crop si es muy largo
        if total_frames > SEGMENT_FRAMES:
            start = np.random.randint(0, total_frames - SEGMENT_FRAMES)
            end = start + SEGMENT_FRAMES
        else:
            start = 0
            end = total_frames
            
        cqt_tensor = torch.tensor(item['cqt'][start:end])
        # Asegurar dimensiones (1, Time, Freq) para Conv2d
        if cqt_tensor.ndim == 2: cqt_tensor = cqt_tensor.unsqueeze(0)

        return {
            "cqt": cqt_tensor,
            "onset": torch.tensor(item['onset'][start:end]),
            "offset": torch.tensor(item['offset'][start:end]),
            "frame": torch.tensor(item['frame'][start:end]),
            "velocity": torch.tensor(item['velocity'][start:end])
        }

# ==========================================
# 2. MODELO  (4 SALIDAS)
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
    def __init__(self, hidden_lstm=128, dropout_rate=0.3):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 16, pool=False) 
        self.enc2 = ConvBlock(16, 32, pool=True)
        self.enc3 = ConvBlock(32, 64, pool=True)
        
        # LSTM
        self.lstm_input_size = 64 * 88 
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_lstm, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_lstm * 2, self.lstm_input_size)
        
        # Decoder (U-Net style skip connections)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.dec3 = ConvBlock(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 1), stride=(2, 1))
        self.dec2 = ConvBlock(32, 16)
        
        # Cabezales de salida
        self.head_onset = nn.Conv2d(16, 1, 1)
        self.head_offset = nn.Conv2d(16, 1, 1)
        self.head_frame = nn.Conv2d(16, 1, 1)
        self.head_velocity = nn.Conv2d(16, 1, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [Batch, 1, Time, Freq]
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
        
        return (
            self.head_onset(d2).squeeze(1), 
            self.head_offset(d2).squeeze(1),
            self.head_frame(d2).squeeze(1),
            self.head_velocity(d2).squeeze(1)
        )

# ==========================================
# 3. OPTUNA OBJECTIVE
# ==========================================
def objective(trial):
    # Definir espacio de búsqueda
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    onset_pos_weight = trial.suggest_float("onset_weight", 5.0, 30.0) # Peso alto para notas raras
    
    # Dataloaders (usando variables globales para eficiencia)
    train_loader = DataLoader(train_ds_global, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds_global, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = PianoCRNN(dropout_rate=dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    
    # Losses
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([onset_pos_weight]).to(DEVICE))
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(DEVICE))
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))
    crit_vel = nn.MSELoss()
    
    # Loop
    n_epochs_optuna = 10
    for epoch in range(n_epochs_optuna):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            cqt = batch['cqt'].to(DEVICE)
            with autocast():
                p_on, p_off, p_fr, p_vel = model(cqt)
                loss = (crit_onset(p_on, batch['onset'].to(DEVICE)) + 
                        crit_offset(p_off, batch['offset'].to(DEVICE)) + 
                        crit_frame(p_fr, batch['frame'].to(DEVICE)) + 
                        crit_vel(torch.sigmoid(p_vel), batch['velocity'].to(DEVICE)))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validar
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                cqt = batch['cqt'].to(DEVICE)
                p_on, _, _, _ = model(cqt)
                # Optimizamos minimizando la loss de Onset (la más crítica)
                val_loss += crit_onset(p_on, batch['onset'].to(DEVICE)).item()
        
        # Reportar a Optuna
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

# ==========================================
# 4. ENTRENAMIENTO FINAL Y EVALUACIÓN
# ==========================================
def evaluate_full_metrics(model, loader):
    model.eval()
    metrics = {k: {'preds': [], 'targets': []} for k in ['onset', 'offset', 'frame', 'velocity']}
    
    print("\n📊 Generando reporte detallado y matrices de confusión...")
    with torch.no_grad():
        for batch in tqdm(loader):
            cqt = batch['cqt'].to(DEVICE)
            p_on, p_off, p_fr, p_vel = model(cqt)
            
            # Aplicar Sigmoid y guardar
            metrics['onset']['preds'].append(torch.sigmoid(p_on).cpu().numpy().flatten())
            metrics['onset']['targets'].append(batch['onset'].numpy().flatten())
            
            metrics['offset']['preds'].append(torch.sigmoid(p_off).cpu().numpy().flatten())
            metrics['offset']['targets'].append(batch['offset'].numpy().flatten())
            
            metrics['frame']['preds'].append(torch.sigmoid(p_fr).cpu().numpy().flatten())
            metrics['frame']['targets'].append(batch['frame'].numpy().flatten())
            
            metrics['velocity']['preds'].append(torch.sigmoid(p_vel).cpu().numpy().flatten())
            metrics['velocity']['targets'].append(batch['velocity'].numpy().flatten())

    # --- GRAFICAR ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tasks = ['onset', 'offset', 'frame']
    
    for i, task in enumerate(tasks):
        preds = np.concatenate(metrics[task]['preds'])
        targets = np.concatenate(metrics[task]['targets'])
        
        # Threshold 0.5
        preds_bin = preds > 0.5
        targets_bin = targets > 0.5
        
        # Métricas
        f1 = f1_score(targets_bin, preds_bin)
        prec = precision_score(targets_bin, preds_bin)
        rec = recall_score(targets_bin, preds_bin)
        
        print(f"🔹 {task.upper()} -> F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
        # Matriz de Confusión
        cm = confusion_matrix(targets_bin, preds_bin)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"{task.upper()} (F1: {f1:.2f})")
        axes[i].set_xlabel("Predicción")
        axes[i].set_ylabel("Real")

    # Velocity MSE
    v_preds = np.concatenate(metrics['velocity']['preds'])
    v_targets = np.concatenate(metrics['velocity']['targets'])
    mse = mean_squared_error(v_targets, v_preds)
    print(f"🔹 VELOCITY -> MSE: {mse:.6f}")
    
    plt.tight_layout()
    plt.savefig("final_metrics_all_outputs_full.png")
    print("✅ Gráfico de métricas guardado como 'final_metrics_all_outputs_full.png'")
    plt.close()

def train_final_model(best_params):
    print("\n🏆 Entrenando modelo final con mejores parámetros:", best_params)
    
    batch_size = best_params['batch_size']
    lr = best_params['lr']
    dropout = best_params['dropout']
    onset_weight = best_params['onset_weight']
    
    train_loader = DataLoader(train_ds_global, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds_global, batch_size=batch_size, shuffle=False)
    
    model = PianoCRNN(dropout_rate=dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    
    crit_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([onset_weight]).to(DEVICE))
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(DEVICE))
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE)) 
    crit_vel = nn.MSELoss()
    
    EPOCHS_FINAL = 100
    
    # Historial para graficar Loss
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(EPOCHS_FINAL):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_FINAL}")
        epoch_train_loss = 0
        
        for batch in loop:
            optimizer.zero_grad()
            cqt = batch['cqt'].to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'cqt'}
            
            with autocast():
                p_on, p_off, p_fr, p_vel = model(cqt)
                loss = (crit_onset(p_on, targets['onset']) + 
                        crit_offset(p_off, targets['offset']) + 
                        crit_frame(p_fr, targets['frame']) + 
                        crit_vel(torch.sigmoid(p_vel), targets['velocity']))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # Guardar loss promedio train
        avg_train = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train)
        
        # Validación Loss
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                cqt = batch['cqt'].to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in batch.items() if k != 'cqt'}
                p_on, p_off, p_fr, p_vel = model(cqt)
                loss = (crit_onset(p_on, targets['onset']) + 
                        crit_offset(p_off, targets['offset']) + 
                        crit_frame(p_fr, targets['frame']) + 
                        crit_vel(torch.sigmoid(p_vel), targets['velocity']))
                epoch_val_loss += loss.item()
        
        avg_val = epoch_val_loss / len(val_loader)
        history['val_loss'].append(avg_val)
        
        loop.set_postfix(train_loss=avg_train, val_loss=avg_val)

    # --- GRAFICAR LOSS ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='x')
    plt.title("Curva de Aprendizaje (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_history_full.png")
    print("\n📈 Gráfico de Loss guardado como 'loss_history_full.png'")
    plt.close()

    # Guardar modelo y evaluar
    torch.save(model.state_dict(), "best_piano_crnn_optuna_full_dataset.pth")
    evaluate_full_metrics(model, val_loader)

# ==========================================
# 5. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print(f"🚀 Iniciando en dispositivo: {DEVICE}")
    
    # 1. Cargar Dataset Global
    try:
        full_ds = PianoDataset(PROCESSED_DIR, split='train', val_split=0.0)
        
        # Split simple para Optuna
        train_size = int(0.85 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds_global, val_ds_global = torch.utils.data.random_split(full_ds, [train_size, val_size])
        
    except Exception as e:
        print(f"❌ Error fatal cargando datos: {e}")
        exit()
        
    # 2. Optuna (Búsqueda de hiperparámetros)
    print("\n🔍 --- FASE 1: Búsqueda de Hiperparámetros con Optuna ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10) # 10 Pruebas
    
    print("\n✅ Mejores Parámetros:", study.best_params)
    
    # 3. Entrenamiento Final
    print("\n💪 --- FASE 2: Entrenamiento Final del Modelo ---")
    train_final_model(study.best_params)
    
    print("\n✨ Proceso finalizado con éxito.")