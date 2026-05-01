import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mir_eval

# ========================================================
# 0. CONFIGURACIÓN BÁSICA (Misma que tu entrenamiento)
# ========================================================
SR = 16000            
HOP_LENGTH = 320    
SEGMENT_FRAMES = 512   
BINS_PER_OCTAVE = 48  
INPUT_BINS = 352
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ========================================================
# 1. DATASET
# ========================================================
class PianoDataset(Dataset):
    def __init__(self, processed_dir, split='validation'):
        from pathlib import Path
        self.processed_dir = Path(processed_dir) / split 
        p = self.processed_dir / "inputs_hcqt"
        if not p.exists(): raise RuntimeError(f"❌ Ruta no existe: {p}")
        
        self.files = sorted(list(p.glob("*.npy")))
        if len(self.files) == 0: raise RuntimeError(f"❌ No se encontraron archivos .npy en {p}")
        
        self.segments = []
        print(f"   Calculando segmentos para {split}...")
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
        print(f"   ✅ {split.upper()}: {len(self.segments)} segmentos cargados.")

    def __len__(self): return len(self.segments)

    def __getitem__(self, idx):
        file_idx, start, end = self.segments[idx]
        fid = self.files[file_idx].name
        try:
            base = self.processed_dir
            def load_slice(folder):
                path = base / folder / fid
                m = np.load(path, mmap_mode='r')
                chunk = np.array(m[start:end])
                if hasattr(m, '_mmap'): m._mmap.close()
                del m
                return chunk

            hcqt = load_slice("inputs_hcqt")
            ped_onset = load_slice("targets_pedal_onset")
            
            curr_len = hcqt.shape[0]
            if curr_len < SEGMENT_FRAMES:
                pad = SEGMENT_FRAMES - curr_len
                hcqt = np.pad(hcqt, ((0, pad), (0,0), (0,0)))
                ped_onset = np.pad(ped_onset, ((0, pad), (0,0)))
            
            hcqt_t = torch.tensor(hcqt).permute(2, 1, 0).float()
            
            return {
                "hcqt": hcqt_t,
                "pedal_onset": torch.tensor(ped_onset).float()
                # (Omitimos cargar targets de notas para ahorrar RAM en esta búsqueda)
            }
        except Exception as e:
            zp = torch.zeros(SEGMENT_FRAMES, 1) 
            return {
                "hcqt": torch.zeros(1, INPUT_BINS, SEGMENT_FRAMES), 
                "pedal_onset": zp
            }


# ========================================================
# 2. ARQUITECTURA HPPNET-SP (Completa)
# ========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation=(1, 1)):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=padding,dilation=dilation, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_c, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=padding,dilation=dilation, bias=False)
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
        if self.downsample is not None: identity = self.downsample(x)
        out += identity 
        out = self.relu(out)
        return out

class HDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        harmonics = [1, 2, 3, 4, 5, 6, 7, 8] 
        for h in harmonics:
            d_math = int(np.round(BINS_PER_OCTAVE * np.log2(h)))
            dil_pt = (d_math, 1) if d_math != 0 else (1, 1)
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=dil_pt, dilation=dil_pt))
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fusion(sum([conv(x) for conv in self.convs]))

class AcousticModel(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(7, 7), padding=(3,3), bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(base_channels, base_channels)
        self.hdc = HDConv(base_channels, base_channels)
        self.hdc_bn = nn.InstanceNorm2d(base_channels, affine=True)
        self.hdc_relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.context = nn.Sequential(
            ResidualBlock(base_channels, base_channels, dilation=(1, 1)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 2)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 4))
        )

    def forward(self, x):
        x = self.res1(self.input_conv(x))
        x = x + self.hdc_relu(self.hdc_bn(self.hdc(x)))
        return self.context(self.pool(x))

class FG_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        b, c, f, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        return self.proj(output).view(b, f, t).permute(0, 2, 1) 

class PedalAcousticModel(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(7, 7), padding=(3,3), bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(base_channels, base_channels)
        self.pool = nn.AdaptiveMaxPool2d((1, None)) 
        self.context = nn.Sequential(
            ResidualBlock(base_channels, base_channels, dilation=(1, 1)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 2)),
            ResidualBlock(base_channels, base_channels, dilation=(1, 4))
        )

    def forward(self, x):
        return self.context(self.pool(self.res1(self.input_conv(x))))

class HPPNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=24, lstm_hidden=128):
        super().__init__()
        self.acoustic_onset = AcousticModel(in_channels, base_channels)
        self.acoustic_other = AcousticModel(in_channels, base_channels)
        self.head_onset = FG_LSTM(base_channels, lstm_hidden)
        
        concat_dim = base_channels * 2
        self.head_frame = FG_LSTM(concat_dim, lstm_hidden)
        self.head_offset = FG_LSTM(concat_dim + 1, lstm_hidden) 
        self.head_velocity = FG_LSTM(concat_dim, lstm_hidden)

        self.pedal_acoustic = PedalAcousticModel(in_channels, base_channels)
        self.pedal_head_onset = FG_LSTM(base_channels, lstm_hidden // 2)
        self.pedal_head_frame = FG_LSTM(base_channels, lstm_hidden // 2)
        self.pedal_head_offset = FG_LSTM(base_channels, lstm_hidden // 2)

    def forward(self, x):
        feat_onset = self.acoustic_onset(x)
        logits_onset = self.head_onset(feat_onset)
        
        feat_combined = torch.cat([self.acoustic_other(x), feat_onset.detach()], dim=1)
        logits_frame = self.head_frame(feat_combined)
        
        prob_frame = torch.sigmoid(logits_frame).detach().permute(0, 2, 1).unsqueeze(1)
        feat_offset_in = torch.cat([feat_combined, prob_frame], dim=1)
        
        logits_offset = self.head_offset(feat_offset_in)
        logits_velocity = self.head_velocity(feat_combined)
        
        feat_pedal = self.pedal_acoustic(x)
        logits_pedal_on = self.pedal_head_onset(feat_pedal)
        logits_pedal_fr = self.pedal_head_frame(feat_pedal)
        logits_pedal_off = self.pedal_head_offset(feat_pedal)
        
        return logits_onset, logits_frame, logits_offset, logits_velocity, logits_pedal_on, logits_pedal_fr, logits_pedal_off


# ========================================================
# 3. FUNCIÓN DE BÚSQUEDA
# ========================================================
def find_best_pedal_thresholds(val_loader, model, device):
    model.eval()
    print("\n🔍 Escaneando validación para encontrar umbrales óptimos...")
    
    all_ped_on_probs, all_ped_on_targets = [], []
    
    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(val_loader, desc="Extrayendo predicciones"):
            hcqt = batch['hcqt'].to(device)
            targets = batch['pedal_onset'].to(device)
            
            # Pasamos CQT y tomamos solo la 5ta salida (pedal_on)
            _, _, _, _, ped_on, _, _ = model(hcqt)
            
            probs = torch.sigmoid(ped_on).cpu().numpy().flatten()
            targs = targets.cpu().numpy().flatten()
            
            all_ped_on_probs.append(probs)
            all_ped_on_targets.append(targs)
            
    all_probs = np.concatenate(all_ped_on_probs)
    all_targs = np.concatenate(all_ped_on_targets) > 0.5
    
    thresholds_to_try = np.arange(0.10, 0.65, 0.05)
    best_f1, best_th = 0, 0
    
    print("\n📊 Resultados de la búsqueda:")
    for th in thresholds_to_try:
        preds = all_probs > th
        tp = np.sum(preds & all_targs)
        fp = np.sum(preds & ~all_targs)
        fn = np.sum(~preds & all_targs)
        
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        
        print(f"   Umbral: {th:.2f} -> F1: {f1:.4f} | Precisión: {p:.4f} | Recall: {r:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    print(f"\n🏆 UMBRAL GANADOR: {best_th:.2f} (F1 Máximo: {best_f1:.4f})")
    print("👉 Cambia la variable TH_PED_ON en tu script principal por este valor.")


# ========================================================
# 4. EJECUCIÓN PRINCIPAL
# ========================================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Iniciando en: {DEVICE}")

    # ----------------------------------------------------
    # ⚠️ 1. REVISA ESTAS DOS RUTAS ANTES DE EJECUTAR
    # ----------------------------------------------------
    RUTA_DATASET = "c:/Users/franc/Desktop/proyectos/AMT_Piano_Sheet_Music/processed_data_cqt_pedal_100" # Ajusta a donde esté tu carpeta 'validation'
    RUTA_PESOS = "c:/Users/franc/Desktop/proyectos/AMT_Piano_Sheet_Music/best_hppnet_phase3_pedal_2.pth"
    BATCH_SIZE = 16 

    try:
        # Cargar Dataset
        val_ds = PianoDataset(RUTA_DATASET, split='validation')
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

        # Construir y cargar Modelo
        print("🏗️ 1. Construyendo la arquitectura de la red...")
        modelo_entrenado = HPPNet(in_channels=1, lstm_hidden=128).to(DEVICE)
        
        print(f"🧠 2. Cargando diccionario de pesos desde: {RUTA_PESOS}")
        checkpoint = torch.load(RUTA_PESOS, map_location=DEVICE)
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
        modelo_entrenado.load_state_dict(state_dict, strict=False)
        print("✅ Pesos inyectados con éxito. ¡Modelo listo!")
        
        # Ejecutar búsqueda
        find_best_pedal_thresholds(val_loader, modelo_entrenado, DEVICE)
        
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")