import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_net_diagram():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')  # Ocultar ejes
    ax.set_title("Arquitectura Dual-Stream PianoCRNN", fontsize=16, pad=20)

    # --- Configuración de Estilos ---
    styles = {
        'input': {'color': '#E0E0E0', 'ec': '#333333'},    # Gris
        'encoder': {'color': '#BBDEFB', 'ec': '#1976D2'},  # Azul
        'lstm': {'color': '#FFF9C4', 'ec': '#FBC02D'},     # Amarillo
        'decoder': {'color': '#E1BEE7', 'ec': '#7B1FA2'},  # Morado
        'head': {'color': '#FFCCBC', 'ec': '#BF360C'},     # Naranja
        'arrow': {'fc': 'black', 'ec': 'black', 'head_width': 1.5, 'head_length': 2}
    }

    # --- Función Helper para dibujar cajas ---
    def draw_box(x, y, w, h, text, style_key, subtext=""):
        style = styles[style_key]
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=2, edgecolor=style['ec'], facecolor=style['color'])
        ax.add_patch(rect)
        cx, cy = x + w/2, y + h/2
        ax.text(cx, cy + 1, text, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(cx, cy - 1.5, subtext, ha='center', va='center', fontsize=8, color='#333333')
        return (cx, y) # Retorna punto inferior para conectar (bottom center)
    
    # --- Función Helper para flechas ---
    def connect(p1, p2, style='-'):
        # p1 = (x, y) start, p2 = (x, y) end
        ax.annotate("", xy=p2, xytext=p1, 
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='black', ls=style))

    # ==========================================
    # 1. ENCODER COMPARTIDO (SHARED)
    # ==========================================
    w, h = 16, 6
    cx = 42 # Centro X general
    
    # Input
    p_in = draw_box(cx, 90, w, h, "INPUT (HCQT)", 'input', "(3, T, 88)")
    
    # Encoder Layers
    p_e1 = draw_box(cx, 80, w, h, "Encoder 1", 'encoder', "Conv2d (16ch)")
    p_e2 = draw_box(cx, 70, w, h, "Encoder 2", 'encoder', "Conv + Pool (32ch)")
    p_e3 = draw_box(cx, 60, w, h, "Encoder 3", 'encoder', "Conv + Pool (64ch)")
    
    # Conexiones Encoder
    connect((p_in[0], p_in[1]), (p_e1[0], p_e1[1]+h+0.4)) # In -> E1
    connect((p_e1[0], p_e1[1]), (p_e2[0], p_e2[1]+h+0.4)) # E1 -> E2
    connect((p_e2[0], p_e2[1]), (p_e3[0], p_e3[1]+h+0.4)) # E2 -> E3

    # ==========================================
    # BIFURCACIÓN (SPLIT)
    # ==========================================
    # Puntos de anclaje para las ramas
    split_y = 55
    ax.text(50, 57, "Split & Flatten", ha='center', fontsize=9, style='italic')
    
    # Líneas de división
    ax.plot([50, 25], [p_e3[1], split_y], color='black', lw=1.5) # A la izquierda
    ax.plot([50, 75], [p_e3[1], split_y], color='black', lw=1.5) # A la derecha

    # ==========================================
    # 2. RAMA IZQUIERDA: TIMING (Onset/Offset)
    # ==========================================
    lx = 17 # Left X center
    
    # LSTM Time
    p_lt = draw_box(lx, 45, w, h, "LSTM TIMING", 'lstm', "Bi-LSTM (Hidden=64)")
    connect((25, split_y), (p_lt[0], p_lt[1]+h+0.4)) # Conexión flecha
    
    # Decoder Time
    p_dt3 = draw_box(lx, 35, w, h, "Dec Time 3", 'decoder', "Up + Cat(E2)")
    p_dt2 = draw_box(lx, 25, w, h, "Dec Time 2", 'decoder', "Up + Cat(E1)")
    
    connect((p_lt[0], p_lt[1]), (p_dt3[0], p_dt3[1]+h+0.4))
    connect((p_dt3[0], p_dt3[1]), (p_dt2[0], p_dt2[1]+h+0.4))
    
    # Heads Time
    p_h_on = draw_box(lx-5, 10, 10, 5, "ONSET", 'head', "Sigmoid")
    p_h_off = draw_box(lx+7, 10, 10, 5, "OFFSET", 'head', "Sigmoid")
    
    # Conexiones finales
    ax.plot([p_dt2[0], p_h_on[0]+5], [p_dt2[1], p_h_on[1]+5.4], color='black', lw=1)
    ax.plot([p_dt2[0], p_h_off[0]+5], [p_dt2[1], p_h_off[1]+5.4], color='black', lw=1)

    # ==========================================
    # 3. RAMA DERECHA: STATE (Frame/Vel)
    # ==========================================
    rx = 67 # Right X center
    
    # LSTM State
    p_ls = draw_box(rx, 45, w, h, "LSTM STATE", 'lstm', "Bi-LSTM (Hidden=64)")
    connect((75, split_y), (p_ls[0], p_ls[1]+h+0.4))
    
    # Decoder State
    p_ds3 = draw_box(rx, 35, w, h, "Dec State 3", 'decoder', "Up + Cat(E2)")
    p_ds2 = draw_box(rx, 25, w, h, "Dec State 2", 'decoder', "Up + Cat(E1)")
    
    connect((p_ls[0], p_ls[1]), (p_ds3[0], p_ds3[1]+h+0.4))
    connect((p_ds3[0], p_ds3[1]), (p_ds2[0], p_ds2[1]+h+0.4))
    
    # Heads State
    p_h_fr = draw_box(rx-5, 10, 10, 5, "FRAME", 'head', "Sigmoid")
    p_h_vel = draw_box(rx+7, 10, 10, 5, "VELOCITY", 'head', "MSE / Sigmoid")
    
    # Conexiones finales
    ax.plot([p_ds2[0], p_h_fr[0]+5], [p_ds2[1], p_h_fr[1]+5.4], color='black', lw=1)
    ax.plot([p_ds2[0], p_h_vel[0]+5], [p_ds2[1], p_h_vel[1]+5.4], color='black', lw=1)

    # ==========================================
    # SKIP CONNECTIONS (U-NET)
    # ==========================================
    # Skip E2 -> Dec3 (Time & State)
    # Dibujamos líneas curvas discontinuas
    e2_y = 70 + 3 # Altura media de E2
    
    # Skip a la izquierda
    ax.annotate("", xy=(lx+16, 38), xytext=(42, e2_y),
                arrowprops=dict(arrowstyle="->", color='#999999', lw=1.5, ls="--", connectionstyle="arc3,rad=-0.3"))
    ax.text(32, 50, "Skip E2", fontsize=8, color='gray', rotation=45)

    # Skip a la derecha
    ax.annotate("", xy=(rx, 38), xytext=(58, e2_y),
                arrowprops=dict(arrowstyle="->", color='#999999', lw=1.5, ls="--", connectionstyle="arc3,rad=0.3"))
    ax.text(68, 50, "Skip E2", fontsize=8, color='gray', rotation=-45)

    # Nota: No dibujo Skip E1 para no saturar, pero la lógica es la misma

    # Leyenda simple
    plt.figtext(0.5, 0.02, "Diagrama generado con Matplotlib - Arquitectura Dual CRNN", ha="center", fontsize=12)
    
    plt.show()

if __name__ == "__main__":
    draw_neural_net_diagram()