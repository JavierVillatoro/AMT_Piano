import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    # Configuración del lienzo
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    ax.set_title("Arquitectura Propuesta: Harmonic Res-U-Net\n(Fusión de U-Net + HPPNet + Onsets&Frames)", 
                 fontsize=16, fontweight='bold', pad=20)

    # Estilos de cajas
    style_encoder = {'boxstyle': 'round,pad=0.3', 'facecolor': '#bbdefb', 'edgecolor': '#0d47a1'}
    style_bottleneck = {'boxstyle': 'round,pad=0.3', 'facecolor': '#fff9c4', 'edgecolor': '#fbc02d'}
    style_decoder = {'boxstyle': 'round,pad=0.3', 'facecolor': '#c8e6c9', 'edgecolor': '#1b5e20'}
    style_head = {'boxstyle': 'round,pad=0.3', 'facecolor': '#e1bee7', 'edgecolor': '#4a148c'}
    style_input = {'boxstyle': 'round,pad=0.3', 'facecolor': '#e0e0e0', 'edgecolor': '#424242'}

    # --- 1. ENCODER (Izquierda) ---
    # Input
    ax.text(2, 14.5, "Input: CQT\n(Freq x Time)", ha='center', va='center', bbox=style_input, fontsize=10)
    
    # Bloques Encoder
    enc_positions = [(2, 12), (2, 9), (2, 6)]
    labels_enc = ["HDC Block 1\n(Harmonic Dilated Conv)", 
                  "HDC Block 2\n(Dilation++)", 
                  "HDC Block 3\n(Deep Harmonic Features)"]
    
    for i, (x, y) in enumerate(enc_positions):
        ax.text(x, y, labels_enc[i], ha='center', va='center', bbox=style_encoder, fontsize=9, fontweight='bold')
        # Flechas hacia abajo (Pooling)
        if i < len(enc_positions) - 1:
            ax.arrow(x, y - 0.8, 0, -1.4, head_width=0.3, head_length=0.3, fc='gray', ec='gray')
            ax.text(x + 0.5, y - 1.5, "Pool", fontsize=8, color='gray')

    # Flecha Input -> Encoder
    ax.arrow(2, 13.8, 0, -1.0, head_width=0.3, head_length=0.3, fc='black', ec='black')

    # --- 2. BOTTLENECK (Fondo) ---
    # Conexión Encoder -> Bottleneck
    ax.arrow(2, 5.2, 0, -1.5, head_width=0.3, head_length=0.3, fc='gray', ec='gray')
    
    # Bloque Central
    ax.text(6, 2, "Bottleneck: Bi-LSTM / Transformer\n(Captura Contexto Temporal & Duración)", 
            ha='center', va='center', bbox=style_bottleneck, fontsize=10, fontweight='bold')

    # --- 3. DECODER (Derecha) ---
    # Conexión Bottleneck -> Decoder
    ax.arrow(9, 2, 1, 0, head_width=0.3, head_length=0.3, fc='gray', ec='gray')
    ax.text(9.5, 2.3, "Reshape", fontsize=8, color='gray')
    
    # Bloques Decoder
    dec_positions = [(10, 6), (10, 9), (10, 12)]
    labels_dec = ["Conv Block 3\n(Upsample + Concat)", 
                  "Conv Block 2\n(Upsample + Concat)", 
                  "Conv Block 1\n(Recupera Resolución)"]

    for i, (x, y) in enumerate(dec_positions):
        ax.text(x, y, labels_dec[i], ha='center', va='center', bbox=style_decoder, fontsize=9, fontweight='bold')
        # Flechas hacia arriba (Upsample)
        if i < len(dec_positions) - 1:
            ax.arrow(x, y + 0.8, 0, 1.4, head_width=0.3, head_length=0.3, fc='gray', ec='gray')
    
    # Flecha inicial hacia arriba desde abajo
    ax.arrow(10, 3, 0, 2.2, head_width=0.3, head_length=0.3, fc='gray', ec='gray')

    # --- 4. SKIP CONNECTIONS (Horizontal) ---
    for i in range(3):
        y = enc_positions[2-i][1] # Conectar niveles correspondientes
        # Dibujar flecha curva o recta
        ax.annotate("", xy=(8.5, y), xytext=(3.5, y),
                    arrowprops=dict(arrowstyle="->", linestyle="dashed", color="#ef6c00", lw=2))
        if i == 1:
            ax.text(6, y + 0.2, "Skip Connections\n(Preserva info de Onset)", ha='center', fontsize=8, color="#ef6c00")

    # --- 5. HEADS (Salidas) ---
    head_start_x = 10
    head_start_y = 12.8
    
    # Flechas distribuidoras
    ax.plot([head_start_x, head_start_x], [head_start_y, 14], color='black') # Tronco vertical
    ax.plot([head_start_x - 3, head_start_x + 3], [14, 14], color='black') # Rama horizontal
    
    # Onset Head
    ax.arrow(head_start_x - 3, 14, 0, 0.5, head_width=0.2, fc='black')
    ax.text(head_start_x - 3, 15, "ONSET Head\n(Sigmoid)", ha='center', va='center', bbox=style_head, fontsize=9)
    
    # Frame Head
    ax.arrow(head_start_x, 14, 0, 0.5, head_width=0.2, fc='black')
    ax.text(head_start_x, 15, "FRAME Head\n(Sigmoid)", ha='center', va='center', bbox=style_head, fontsize=9)
    
    # Velocity Head
    ax.arrow(head_start_x + 3, 14, 0, 0.5, head_width=0.2, fc='black')
    ax.text(head_start_x + 3, 15, "VELOCITY Head\n(Regresión 0-1)", ha='center', va='center', bbox=style_head, fontsize=9)

    # Condicionales (Flechas curvas extra)
    # Onset -> Frame y Velocity
    ax.annotate("", xy=(head_start_x - 0.5, 15), xytext=(head_start_x - 2, 15),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color="purple", linestyle=":"))
    ax.text(head_start_x - 1.2, 15.3, "Condición", fontsize=7, color='purple')

    # Leyenda explicativa en el gráfico
    text_desc = (
        "LEYENDA:\n"
        "• HDC (Azul): Convoluciones Armónicas Dilatadas [HPPNet].\n"
        "  Aprenden patrones de acordes invariantes.\n"
        "• Skip Connections (Naranja): Pasan detalles finos [U-Net].\n"
        "  Cruciales para detectar el milisegundo exacto del ataque.\n"
        "• Bi-LSTM (Amarillo): Entiende la duración y secuencia [Onsets&Frames].\n"
        "• Heads (Morado): Salidas especializadas."
    )
    ax.text(14, 8, text_desc, ha='left', va='center', fontsize=10, 
            bbox={'boxstyle': 'square,pad=0.5', 'facecolor': '#f5f5f5', 'edgecolor': '#9e9e9e'})

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_architecture()