import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
import warnings
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# --- CONFIG ---
SR = 16000
HOP_LENGTH = 512
MIN_MIDI = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MISMO MODELO QUE EN TRAINING ---
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
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))) if not self.pool else self.pool(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))))

class PianoCRNN(nn.Module):
    def __init__(self, input_channels=3, hidden_lstm=128):
        super().__init__()
        self.enc1 = ConvBlock(input_channels, 16, pool=False) 
        self.enc2 = ConvBlock(16, 32, pool=True)
        self.enc3 = ConvBlock(32, 64, pool=True)
        self.lstm = nn.LSTM(64*88, hidden_lstm, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_lstm * 2, 64*88)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.dec3 = ConvBlock(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 1), stride=(2, 1))
        self.dec2 = ConvBlock(32, 16)
        
        # 4 Salidas
        self.head_onset = nn.Conv2d(16, 1, 1)
        self.head_frame = nn.Conv2d(16, 1, 1)
        self.head_offset = nn.Conv2d(16, 1, 1)
        self.head_vel = nn.Conv2d(16, 1, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b, c, t, f = e3.shape
        lstm_out, _ = self.lstm(e3.permute(0, 2, 1, 3).reshape(b, t, c*f))
        proj = self.lstm_proj(self.dropout(self.relu(lstm_out))).view(b, t, c, f).permute(0, 2, 1, 3)
        d3 = self.dec3(torch.cat([self.up3(proj), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return self.head_onset(d2).squeeze(1), self.head_frame(d2).squeeze(1), self.head_offset(d2).squeeze(1), torch.sigmoid(self.head_vel(d2).squeeze(1))

# --- PROCESADO ---
def compute_cqt_layer(y, harmonics_multiplier):
    base_fmin = librosa.note_to_hz('A0')
    target_fmin = base_fmin * harmonics_multiplier
    nyquist = SR / 2
    max_bins = int(np.floor(12 * np.log2(nyquist / target_fmin)))
    actual_bins = min(88, max_bins)
    if actual_bins <= 0: return np.zeros((88, int(len(y)/HOP_LENGTH) + 1), dtype=np.float32).T
    cqt = librosa.cqt(y=y, sr=SR, hop_length=HOP_LENGTH, fmin=target_fmin, n_bins=actual_bins, bins_per_octave=12)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).astype(np.float32)
    cqt_db = np.clip((cqt_db + 80.0) / 80.0, 0, 1)
    if actual_bins < 88:
        full = np.zeros((88, cqt_db.shape[1]), dtype=np.float32)
        full[:actual_bins, :] = cqt_db
        cqt_db = full
    return cqt_db.T

# --- NUEVA FUNCIÓN: CARGA INTELIGENTE ---
def smart_load_audio(path):
    """
    Intenta cargar el audio. Si es MP3 y librosa falla, 
    usa pydub para convertirlo a WAV temporalmente.
    """
    path_obj = Path(path)
    temp_wav = None

    # Caso A: Es MP3 -> Intentamos conversión preventiva
    if path_obj.suffix.lower() == '.mp3':
        print(f"🎵 Detectado MP3. Intentando conversión automática a WAV...")
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_mp3(path)
                # Exportamos a un archivo temporal en la misma carpeta
                temp_wav = path_obj.with_suffix(".temp_convert.wav")
                audio.export(temp_wav, format="wav")
                print(f"✅ Conversión exitosa. Usando temporal: {temp_wav.name}")
                path = str(temp_wav) # Cambiamos la ruta a cargar por el WAV
            except Exception as e:
                print(f"⚠️ Falló pydub ({e}). Intentando carga directa...")
        else:
            print("⚠️ No tienes 'pydub' instalado (pip install pydub). Se intentará carga directa.")

    # Carga con Librosa
    try:
        # Ignoramos warnings de PySoundFile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(path, sr=SR)
            
        # Limpieza: Si creamos un temporal, lo borramos
        if temp_wav and temp_wav.exists():
            try:
                os.remove(temp_wav)
                print("🧹 Archivo temporal eliminado.")
            except: pass
            
        return y
        
    except Exception as e:
        # Si falló, explicamos por qué
        if temp_wav and temp_wav.exists(): os.remove(temp_wav)
        raise RuntimeError(f"No se pudo leer el audio. Si es MP3 y falla, instala ffmpeg en Windows.\nError original: {e}")

def compute_input(audio_path, mode):
    print(f"🔄 Cargando audio: {audio_path}")
    
    # Usamos la nueva carga inteligente
    y = smart_load_audio(audio_path)

    if mode == "CQT":
        tensor = torch.tensor(compute_cqt_layer(y, 1)).unsqueeze(0).unsqueeze(0)
    else:
        layers = [compute_cqt_layer(y, h) for h in [1, 2, 3]]
        min_len = min([l.shape[0] for l in layers])
        data = np.stack([l[:min_len, :] for l in layers], axis=-1)
        tensor = torch.tensor(data).permute(2, 0, 1).unsqueeze(0)
    
    pad_needed = (4 - (tensor.shape[2] % 4)) % 4
    if pad_needed > 0: tensor = F.pad(tensor, (0, 0, 0, pad_needed))
    return tensor

def matrix_to_midi(onset, frame, offset, velocity, output_path):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    onsets_bin = onset > 0.4
    frames_bin = frame > 0.4
    offsets_bin = offset > 0.4
    
    time_per_frame = HOP_LENGTH / SR
    print("🎹 Generando MIDI de alta precisión...")

    for pitch in range(88):
        midi_num = pitch + MIN_MIDI
        onset_idxs = np.where(np.diff(onsets_bin[:, pitch].astype(int), prepend=0) == 1)[0]
        
        for start_idx in onset_idxs:
            end_idx = start_idx + 1
            while end_idx < len(frames_bin):
                if not frames_bin[end_idx, pitch]: break
                if offsets_bin[end_idx, pitch]: break
                end_idx += 1
            
            vel_val = velocity[start_idx, pitch]
            final_vel = int(np.clip(vel_val * 127, 20, 127))
            
            start_t = start_idx * time_per_frame
            end_t = end_idx * time_per_frame
            
            if end_t - start_t > 0.05:
                note = pretty_midi.Note(velocity=final_vel, pitch=midi_num, start=start_t, end=end_t)
                piano.notes.append(note)

    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"✅ ¡Guardado en {output_path} con dinámica real!")

def main():
    print("--- INFERENCIA PRECISIÓN TOTAL (AUTO MP3 -> WAV) ---")
    
    raw_model = input("Archivo .pth: ").strip()
    model_path = raw_model.replace("&", "").replace("'", "").replace('"', "").strip()
    
    mode = "HCQT" if input("Modo (1=CQT, 2=HCQT): ").strip() == "2" else "CQT"
    in_ch = 3 if mode == "HCQT" else 1
    
    try:
        model = PianoCRNN(input_channels=in_ch).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("✅ Modelo cargado.")
    except Exception:
        print("❌ Error cargando modelo.")
        return

    while True:
        raw_input = input("\n🎵 Audio (arrastra archivo aquí): ").strip()
        if raw_input.lower() == 'q': break
        
        # Limpieza de ruta Windows
        p = raw_input.replace("&", "").replace("'", "").replace('"', "").strip()
        
        if not os.path.exists(p):
            print(f"❌ Archivo no encontrado: {p}")
            continue
            
        try:
            tens = compute_input(p, mode).to(DEVICE)
            with torch.no_grad():
                o, f, off, v = model(tens)
                
            out = Path("mis_midis_generados") / (Path(p).stem + "_full.mid")
            out.parent.mkdir(exist_ok=True)
            
            matrix_to_midi(
                torch.sigmoid(o).cpu().numpy()[0],
                torch.sigmoid(f).cpu().numpy()[0],
                torch.sigmoid(off).cpu().numpy()[0],
                v.cpu().numpy()[0], 
                out
            )
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()