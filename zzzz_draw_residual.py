import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram():
    # Configuración de la figura
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # --- Estilos ---
    # Colores académicos (suaves)
    c_input = '#FFDDC1'   # Naranja suave (Datos)
    c_high = '#C1E1FF'    # Azul suave (High Res Processing)
    c_pool = '#FFABAB'    # Rojo suave (Cambio dimensión)
    c_low = '#C1C6FF'     # Violeta suave (Low Res Context)
    c_head = '#D0F0C0'    # Verde suave (Salidas)
    c_midi = '#E1C1FF'    # Morado (Resultado final)
    
    # Estilo de caja
    def draw_box(x, y, w, h, title, sub, color):
        # Sombra
        shadow = patches.FancyBboxPatch((x+0.1, y-0.1), w, h, boxstyle="round,pad=0.2", 
                                       ec="none", fc='gray', alpha=0.3)
        ax.add_patch(shadow)
        # Caja principal
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                     ec="black", fc=color, linewidth=1.5)
        ax.add_patch(box)
        # Texto
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(x + w/2, y + h*0.35, sub, ha='center', va='center', fontsize=9, family='monospace')

    # Estilo de flecha
    def draw_arrow(x_from, y_from, x_to, y_to):
        ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

    # ==========================================
    # DIBUJO DEL PIPELINE (De arriba a abajo)
    # ==========================================
    
    # 1. RAW AUDIO
    draw_box(4, 22, 4, 1, "Raw Audio Input", "Waveform (16kHz)", c_input)
    draw_arrow(6, 22, 6, 21.2)
    
    # 2. CQT PREPROCESSING
    draw_box(4, 20, 4, 1, "CQT Extraction", "High-Res: 352 Bins\n(B, 1, 352, 512)", c_input)
    draw_arrow(6, 20, 6, 19.2)
    
    # 3. ACOUSTIC MODEL (HIGH RES)
    draw_box(3, 16.5, 6, 2.5, "Acoustic Model (High-Res)", 
             "Input Conv (7x7)\n+\nResidual Block (3x3)\n+\nHD-Conv (Armónicos)", c_high)
    ax.text(9.2, 17.75, "Tensor:\n(B, 24, 352, 512)", fontsize=9, color='blue', ha='left')
    draw_arrow(6, 16.5, 6, 15.2)
    
    # 4. BOTTLENECK (MAXPOOL)
    draw_box(4, 14, 4, 1, "📉 MaxPool Downsampling", "Kernel=(4,1) | Stride=(4,1)", c_pool)
    ax.text(8.2, 14.5, "¡Freq / 4!", fontsize=10, fontweight='bold', color='red')
    draw_arrow(6, 14, 6, 12.7)
    
    # 5. ACOUSTIC MODEL (LOW RES / CONTEXT)
    draw_box(3, 10.5, 6, 2, "Temporal Context", 
             "3x Dilated Residual Blocks\nDilation Time: [1, 2, 4]", c_low)
    ax.text(9.2, 11.5, "Tensor:\n(B, 24, 88, 512)", fontsize=9, color='blue', ha='left')
    draw_arrow(6, 10.5, 6, 9.5)
    
    # --- RAMIFICACIÓN A CABEZAS (LSTMs) ---
    # Línea horizontal distribuidora
    plt.plot([2, 10], [9.5, 9.5], color='black', lw=1.5)
    
    # Cabezas
    # Onset
    draw_arrow(2, 9.5, 2, 8.5)
    draw_box(1, 7, 2, 1.5, "Onset Head", "Bi-LSTM\n(B, 88, 512)", c_head)
    
    # Frame
    draw_arrow(4.66, 9.5, 4.66, 8.5)
    draw_box(3.66, 7, 2, 1.5, "Frame Head", "Bi-LSTM\n(B, 88, 512)", c_head)
    
    # Offset
    draw_arrow(7.33, 9.5, 7.33, 8.5)
    draw_box(6.33, 7, 2, 1.5, "Offset Head", "Bi-LSTM\n(B, 88, 512)", c_head)
    
    # Velocity
    draw_arrow(10, 9.5, 10, 8.5)
    draw_box(9, 7, 2, 1.5, "Velocity Head", "Bi-LSTM\n(B, 88, 512)", c_head)
    
    # --- CONEXIONES ENTRE CABEZAS (Dependencies) ---
    # HPPNet usa Onset para ayudar a Frame, etc.
    # Dibujamos flechas curvas punteadas para indicar dependencia condicional
    ax.annotate("", xy=(3.66, 7.75), xytext=(3, 7.75), arrowprops=dict(arrowstyle="->", ls="--", color='gray'))
    ax.annotate("", xy=(6.33, 7.75), xytext=(5.66, 7.75), arrowprops=dict(arrowstyle="->", ls="--", color='gray'))
    
    # 7. POST-PROCESSING
    # Recogemos todas las flechas en un punto
    draw_arrow(2, 7, 6, 5.5) # Desde Onset
    draw_arrow(4.66, 7, 6, 5.5) # Desde Frame
    draw_arrow(7.33, 7, 6, 5.5) # Desde Offset
    draw_arrow(10, 7, 6, 5.5) # Desde Vel
    
    draw_box(3, 4, 6, 1.5, "Decoding & Thresholding", 
             "Sigmoid > Thresh\nPeak Picking Algorithm", c_input)
             
    draw_arrow(6, 4, 6, 2.5)
    
    # 8. OUTPUT
    draw_box(4.5, 1, 3, 1.5, "OUTPUT", "MIDI File\n(.mid)", c_midi)

    # Título
    plt.title("Arquitectura: Res-HPPNet High-Res\n(Batch=32, Seg=512 Frames)", fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig("modelo_arquitectura_final.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_architecture_diagram()