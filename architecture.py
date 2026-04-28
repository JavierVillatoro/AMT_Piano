from graphviz import Digraph

def draw_hppnet_sp_pedal_architecture():
    dot = Digraph(name='HPPNet_SP_Pedal', format='png')
    
    # Ajustes globales para que todo quepa y sea legible
    dot.attr(dpi='300', rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.6')
    dot.attr('node', shape='box', style='filled', fillcolor='white', 
             fontname='Arial', fontsize='11', margin='0.15,0.1', penwidth='1.2')
    dot.attr('edge', fontname='Arial', fontsize='9')

    # ==========================================
    # 1. ENTRADA (AUDIO -> CQT)
    # ==========================================
    dot.node('audio', 'Audio Waveform (16 kHz)', shape='cylinder', fillcolor='#f0f0f0')
    dot.node('cqt', 'High-Res CQT\n(B, 1, 352, T)', shape='cylinder', fillcolor='#e8f4f8')
    dot.edge('audio', 'cqt', label=' Preprocesamiento')

    # Nodos de bifurcación principal
    dot.node('split', '', shape='point', width='0')
    dot.edge('cqt', 'split')

    # ==========================================
    # RAMA A1: NOTAS - ONSET (AcousticModel)
    # ==========================================
    with dot.subgraph(name='cluster_notes_onset') as c1:
        c1.attr(label='Rama Notas: Onset Prior', style='dashed', color='blue', fontname='Arial bold')
        c1.node('ac_onset', 'AcousticModel (Onset)\n[Conv7x7 -> HDConv -> MaxPool(4,1) -> Context]', fillcolor='#e3f2fd')
        c1.node('head_on', 'FG-LSTM: Onset Head\n(B, 88, T)', fillcolor='#bbdefb')
        
        c1.edge('ac_onset', 'head_on', label=' feat_onset')

    # ==========================================
    # RAMA A2: NOTAS - OTHER (AcousticModel)
    # ==========================================
    with dot.subgraph(name='cluster_notes_other') as c2:
        c2.attr(label='Rama Notas: Resto', style='dashed', color='darkblue', fontname='Arial bold')
        c2.node('ac_other', 'AcousticModel (Other)\n[Conv7x7 -> HDConv -> MaxPool(4,1) -> Context]', fillcolor='#e3f2fd')
        
        # Concat 1
        c2.node('cat1', 'Concat (dim=1)', shape='invhouse', fillcolor='#ffe0b2')
        
        # Cabeceras
        c2.node('head_fr', 'FG-LSTM: Frame Head\n(B, 88, T)', fillcolor='#bbdefb')
        c2.node('head_vel', 'FG-LSTM: Velocity Head\n(B, 88, T)', fillcolor='#bbdefb')
        
        # Concat 2
        c2.node('cat2', 'Concat (dim=1)', shape='invhouse', fillcolor='#ffe0b2')
        
        # Cabecera Offset
        c2.node('head_off', 'FG-LSTM: Offset Head\n(B, 88, T)', fillcolor='#bbdefb')
        
        # Conexiones internas Rama A2
        c2.edge('ac_other', 'cat1', label=' feat_other')
        c2.edge('cat1', 'head_fr', label=' feat_combined')
        c2.edge('cat1', 'head_vel')
        c2.edge('cat1', 'cat2', label=' feat_combined')
        c2.edge('cat2', 'head_off', label=' feat_offset_in')

    # ==========================================
    # LAS SKIP CONNECTIONS (La magia de HPPNet-SP)
    # ==========================================
    # Skip 1: Onset -> Concat 1
    dot.edge('ac_onset', 'cat1', label=' detach()', color='red', style='dashed')
    
    # Skip 2: Frame Prob -> Concat 2
    dot.edge('head_fr', 'cat2', label=' Sigmoid + detach()', color='red', style='dashed')

    # ==========================================
    # RAMA B: PEDAL DE SOSTENIDO
    # ==========================================
    with dot.subgraph(name='cluster_pedal') as p:
        p.attr(label='Rama Pedal', style='dashed', color='magenta', fontname='Arial bold')
        
        p.node('ac_pedal', 'PedalAcousticModel\n[Conv7x7 -> AdaptiveMaxPool(1,None) -> Context]', fillcolor='#fce4ec')
        
        p.node('p_head_on', 'FG-LSTM: Pedal Onset\n(B, 1, T)', fillcolor='#f8bbd0')
        p.node('p_head_fr', 'FG-LSTM: Pedal Frame\n(B, 1, T)', fillcolor='#f8bbd0')
        p.node('p_head_off', 'FG-LSTM: Pedal Offset\n(B, 1, T)', fillcolor='#f8bbd0')
        
        p.edge('ac_pedal', 'p_head_on')
        p.edge('ac_pedal', 'p_head_fr')
        p.edge('ac_pedal', 'p_head_off')

    # Conectar Split inicial con los 3 modelos acústicos
    dot.edge('split', 'ac_onset')
    dot.edge('split', 'ac_other')
    dot.edge('split', 'ac_pedal')

    # ==========================================
    # POST-PROCESAMIENTO Y SALIDA
    # ==========================================
    dot.node('dec_notes', 'Note Decoding\n(Peak Picking & Tracking)', shape='component', fillcolor='#e8efe8')
    dot.node('dec_pedal', 'Pedal Decoding\n(Algoritmo Sub-frame)', shape='component', fillcolor='#e8efe8')
    
    # Juntar salidas Notas
    dot.edge('head_on', 'dec_notes')
    dot.edge('head_fr', 'dec_notes')
    dot.edge('head_off', 'dec_notes')
    dot.edge('head_vel', 'dec_notes')
    
    # Juntar salidas Pedal
    dot.edge('p_head_on', 'dec_pedal')
    dot.edge('p_head_fr', 'dec_pedal')
    dot.edge('p_head_off', 'dec_pedal')

    dot.node('midi', 'Archivo MIDI', shape='note', fillcolor='#fff9c4', penwidth='2')
    
    dot.edge('dec_notes', 'midi', label=' 88 Teclas')
    dot.edge('dec_pedal', 'midi', label=' CC 64 (Sustain)')

    # Renderizar
    output_filename = 'HPPNet_SP_Pedal_Architecture'
    dot.render(output_filename, view=False, cleanup=True)
    print(f"✅ Arquitectura dibujada: '{output_filename}.png'")

if __name__ == "__main__":
    draw_hppnet_sp_pedal_architecture()