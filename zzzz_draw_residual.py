import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_professional_architecture(save_path="hcqt_residual_hppnet_paper.png"):
    # --- Configuración del Lienzo ---
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # --- Paleta de Colores (Estilo Académico Moderno) ---
    c_h05 = '#FFCDD2'  # Rojo suave (Canal 1)
    c_h1  = '#C8E6C9'  # Verde suave (Canal 2)
    c_h2  = '#BBDEFB'  # Azul suave (Canal 3)
    
    c_onset_bg = '#E3F2FD' # Fondo azulado para rama Onset
    c_other_bg = '#FFF3E0' # Fondo anaranjado para rama Other
    
    c_block = '#FFFFFF'    # Bloques blancos
    c_hdc   = '#D1C4E9'    # HDConv (Violeta suave)
    c_lstm  = '#F0F0F0'    # Gris muy claro
    c_edge  = '#333333'    # Bordes oscuros
    
    # --- Funciones de Dibujo ---
    def draw_box(x, y, w, h, color, label, sublabel=None, fontsize=10, style="round,pad=0.3"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=style, 
                                     linewidth=1.2, edgecolor=c_edge, facecolor=color, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + (1.5 if sublabel else 0), label, 
                ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#222', zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 1.5, sublabel, 
                    ha='center', va='center', fontsize=fontsize-2, style='italic', color='#555', zorder=4)
        return (x + w/2, y) # Return bottom center

    def draw_arrow(x1, y1, x2, y2, style='->', ls='-', color='#444'):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.5, linestyle=ls, shrinkA=0, shrinkB=0),
                    zorder=2)

    # ==========================================
    # 1. INPUT HCQT (REPRESENTACIÓN 3D)
    # ==========================================
    # Dibujamos 3 capas apiladas para que sea OBVIO que es HCQT
    start_x, start_y = 42, 90
    w_lay, h_lay = 16, 6
    offset = 1.5
    
    # Capa h=2
    draw_box(start_x + 2*offset, start_y + 2*offset, w_lay, h_lay, c_h2, "Ch 2: h=2.0", "Harmonic", style="square")
    # Capa h=1
    draw_box(start_x + offset, start_y + offset, w_lay, h_lay, c_h1, "Ch 1: h=1.0", "Fundamental", style="square")
    # Capa h=0.5
    draw_box(start_x, start_y, w_lay, h_lay, c_h05, "Ch 0: h=0.5", "Sub-Harmonic", style="square")
    
    # Etiqueta Global Input
    ax.text(start_x + w_lay + 5, start_y + h_lay/2, "Input HCQT Tensor\n(Batch, 3, 88, Time)", 
            va='center', fontsize=11, fontweight='bold')

    # Punto de salida del input
    p_input_out = (start_x + w_lay/2, start_y)

    # ==========================================
    # 2. RAMAS (ONSET vs OTHER)
    # ==========================================
    y_start_branches = 82
    w_col = 30
    h_col = 48
    
    # Fondo Rama Onset (Izquierda)
    rect_on = patches.FancyBboxPatch((10, 30), w_col, h_col, boxstyle="round,pad=1", fc=c_onset_bg, ec='none', alpha=0.5)
    ax.add_patch(rect_on)
    ax.text(25, y_start_branches, "ONSET Branch\n(Shape Expert)", ha='center', fontsize=12, fontweight='bold', color='#1565C0')

    # Fondo Rama Other (Derecha)
    rect_off = patches.FancyBboxPatch((60, 30), w_col, h_col, boxstyle="round,pad=1", fc=c_other_bg, ec='none', alpha=0.5)
    ax.add_patch(rect_off)
    ax.text(75, y_start_branches, "OTHER Branch\n(Duration Expert)", ha='center', fontsize=12, fontweight='bold', color='#E65100')

    # --- Bloques Internos ---
    block_h = 5
    gap = 4
    y_curr = 72
    
    for i in range(2): # Resblocks iniciales
        draw_box(13, y_curr, 24, block_h, c_block, f"Residual Block {i+1}", "Conv-IN-ReLU")
        draw_box(63, y_curr, 24, block_h, c_block, f"Residual Block {i+1}", "Conv-IN-ReLU")
        # Flechas intermedias
        if i == 0:
            draw_arrow(25, y_curr, 25, y_curr - gap)
            draw_arrow(75, y_curr, 75, y_curr - gap)
        y_curr -= (block_h + gap)

    # HDConv (Diferenciada)
    draw_box(13, y_curr, 24, block_h, c_hdc, "HDConv Layer", "Dilated Harmonics")
    draw_box(63, y_curr, 24, block_h, c_hdc, "HDConv Layer", "Dilated Harmonics")
    # Flechas a HDConv
    draw_arrow(25, y_curr + block_h + gap, 25, y_curr + block_h)
    draw_arrow(75, y_curr + block_h + gap, 75, y_curr + block_h)
    
    y_curr -= (block_h + gap)
    
    # Context
    draw_box(13, y_curr, 24, block_h, c_block, "Context Stack", "3x Residual Blocks")
    draw_box(63, y_curr, 24, block_h, c_block, "Context Stack", "3x Residual Blocks")
    # Flechas a Context
    draw_arrow(25, y_curr + block_h + gap, 25, y_curr + block_h)
    draw_arrow(75, y_curr + block_h + gap, 75, y_curr + block_h)

    # Conectar Input a Ramas
    draw_arrow(p_input_out[0], p_input_out[1], 25, 77, color='#555')
    draw_arrow(p_input_out[0], p_input_out[1], 75, 77, color='#555')

    # ==========================================
    # 3. FUSIÓN Y DETACH
    # ==========================================
    y_fusion = 20
    
    # Caja de Concatenación
    draw_box(40, y_fusion, 20, 5, '#FFF', "Concatenate", "dim=1")
    
    # Salidas de las ramas
    p_onset_end = (25, y_curr)
    p_other_end = (75, y_curr)
    
    # Flecha Onset -> Concat (CON DETACH)
    ax.annotate("Stop Gradient\n(.detach())", xy=(40, y_fusion+2.5), xytext=(25, y_fusion+12),
                arrowprops=dict(arrowstyle="->", color='#D32F2F', lw=2, linestyle="--"),
                ha='center', fontsize=9, color='#D32F2F', fontweight='bold', 
                bbox=dict(boxstyle="round", fc="white", ec="#D32F2F"))
    # Línea visual desde Onset
    draw_arrow(25, y_curr, 25, y_fusion+12) 
    
    # Flecha Other -> Concat
    draw_arrow(75, y_curr, 60, y_fusion+2.5)

    # ==========================================
    # 4. CABEZAS (LSTMs)
    # ==========================================
    y_lstm = 8
    lstm_w = 18
    
    # Onset Head (Directa desde rama Onset)
    draw_box(5, y_lstm, lstm_w, 6, c_lstm, "Onset Head", "Bi-LSTM (128)")
    draw_arrow(25, y_curr, 14, y_lstm+6) # Rama Onset -> Head Onset
    
    # Otras Heads (Desde Concat)
    draw_box(29, y_lstm, lstm_w, 6, c_lstm, "Frame Head", "Bi-LSTM (128)")
    draw_box(53, y_lstm, lstm_w, 6, c_lstm, "Offset Head", "Bi-LSTM (128)")
    draw_box(77, y_lstm, lstm_w, 6, c_lstm, "Velocity Head", "Bi-LSTM (128)")
    
    # Flechas desde Concat
    draw_arrow(50, y_fusion, 38, y_lstm+6)
    draw_arrow(50, y_fusion, 62, y_lstm+6)
    draw_arrow(50, y_fusion, 86, y_lstm+6)

    # Títulos finales
    ax.text(50, 98, "Residual HCQT-HPPNet Architecture", fontsize=18, fontweight='bold', ha='center')
    ax.text(50, 1, "Output: MIDI Events (Onset, Frame, Offset, Velocity)", fontsize=12, style='italic', ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Diagrama PRO guardado: {save_path}")
    plt.show()

if __name__ == "__main__":
    draw_professional_architecture()