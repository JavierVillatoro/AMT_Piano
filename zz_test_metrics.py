import numpy as np
import torch
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error

# --- 1. CONFIGURACIÓN MOCK ---
try:
    import mir_eval
    HAS_MIR_EVAL = True
    print("✅ mir_eval encontrado.")
except ImportError:
    HAS_MIR_EVAL = False
    print("❌ mir_eval NO encontrado. Este test no sirve de mucho sin él.")

FPS = 31.25

# --- 2. TUS FUNCIONES (COPIADAS DEL SCRIPT CORREGIDO) ---

def grid_to_intervals(grid, threshold=0.5):
    """Decodifica matriz binaria a notas (start, end, pitch)"""
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

# --- 3. EJECUCIÓN DE PRUEBA ---
print("\n🧪 Generando datos falsos...")
# Batch=4, 88 notas, 320 frames (10 segs)
B, F, T = 4, 88, 320

# Predicciones (Logits aleatorios)
mock_p_on = torch.randn(B, F, T)
mock_p_fr = torch.randn(B, F, T) # Algunos serán > 0.5 tras sigmoid
mock_p_off = torch.randn(B, F, T)
mock_p_vel = torch.randn(B, F, T)

# Targets (0 o 1)
mock_t_on = torch.randint(0, 2, (B, F, T)).float()
mock_t_fr = torch.randint(0, 2, (B, F, T)).float() # Frame targets
mock_t_off = torch.randint(0, 2, (B, F, T)).float()
mock_t_vel = torch.rand(B, F, T)

print("🧪 Ejecutando compute_batch_metrics...")
try:
    metrics = compute_batch_metrics(
        mock_p_on, mock_p_fr, mock_p_off, mock_p_vel,
        mock_t_on, mock_t_fr, mock_t_off, mock_t_vel
    )
    
    print("\n✅ ¡ÉXITO! La función ha terminado sin errores.")
    print("📊 Resultados de prueba:")
    for k, v in metrics.items():
        print(f"   - {k}: {v:.4f}")
    
    print("\n🚀 CONCLUSIÓN: Puedes iniciar el entrenamiento con seguridad.")

except Exception as e:
    print("\n❌ FALLO: El código sigue roto.")
    print(e)
    # Imprimir traza completa si quieres
    import traceback
    traceback.print_exc()