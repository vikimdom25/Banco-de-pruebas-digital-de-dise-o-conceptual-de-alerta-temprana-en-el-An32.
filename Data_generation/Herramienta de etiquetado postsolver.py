import h5py
import numpy as np
import pandas as pd
import sys
import os

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
HDF5_PATH = r"C:\Users\santi\OneDrive\Documentos\visualizer\Base_de_datos_GOLD_Ultimate_PCHIP.h5"
CSV_PATH = r"C:\Users\santi\OneDrive\Documentos\visualizer\METADATA_VUELOS_FINAL.csv"

# Índices
IDX_ALT = 16; IDX_ROLL = 17; IDX_PITCH = 18; IDX_ALPHA = 20
IDX_VZ = 31; IDX_G = 38; IDX_FLAPS = 7
IDX_LABEL = 66; IDX_FUTURE = 67; IDX_TTS = 68

# Umbrales
SAFE_ALTITUDE = 200.0    # Pies (Debe superar esto para activar búsqueda de impacto)
IMPACT_ALTITUDE = 50.0   # Pies (Cota de choque)
STRUCTURAL_G = 9.0       
MAX_ABS_ALPHA = 85.0     
BUFFER_FRAMES = 5        

# Umbrales Etiquetado v7
WARN_G = 1.4; CRIT_ROLL = 80.0; CRIT_DIVE_VZ = -40.0; CRIT_DIVE_GAMMA = -20.0
KEEP_DIVE_VZ = -15.0; KEEP_DIVE_GAMMA = -5.0; RESET_VZ = -5.0; RESET_PITCH = -5.0

# Cargar CSV
print(f"Cargando CSV Ground Truth...")
try: df_meta = pd.read_csv(CSV_PATH, sep=',', encoding='utf-8')
except: df_meta = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')

df_meta['clean_name'] = df_meta['nombre_archivo'].astype(str).str.replace('.csv', '', regex=False)
outcome_map = dict(zip(df_meta['clean_name'], df_meta['outcome']))

# ==========================================
# 2. FUNCIONES LÓGICAS
# ==========================================
def calcular_etiquetas_v7_numpy(alpha, g_load, vz, pitch, roll, flaps_val):
    n = len(alpha)
    states = np.zeros(n, dtype=np.int8)
    gamma = pitch - alpha
    stall_limit = 16.0 - (3.0 * flaps_val) if not np.isscalar(flaps_val) else np.full(n, 16.0 - 3.0*float(flaps_val))
    event_active = False 
    for i in range(n):
        trig_red = (alpha[i] > stall_limit[i]) or (abs(roll[i]) > CRIT_ROLL)
        trig_dive = (vz[i] < CRIT_DIVE_VZ) or (gamma[i] < CRIT_DIVE_GAMMA)
        if trig_red: s=2; event_active=True
        elif trig_dive: s=3; event_active=True
        elif event_active:
            if (vz[i] < KEEP_DIVE_VZ) or (gamma[i] < KEEP_DIVE_GAMMA): s=3
            elif (vz[i] > RESET_VZ) and (pitch[i] > RESET_PITCH) and (not trig_red): s=0; event_active=False
            else: s=1 if ((alpha[i]>stall_limit[i]-3) or (g_load[i]>WARN_G)) else 0
        else: s=1 if ((alpha[i]>stall_limit[i]-3) or (g_load[i]>WARN_G)) else 0
        states[i] = s
    return pd.Series(states).rolling(5, center=True).median().fillna(0).astype(np.int8).values

def encontrar_punto_impacto_inteligente(data):
    """
    Encuentra el impacto PERO respetando el despegue.
    Solo busca el choque DESPUÉS de haber volado (Alt > SAFE_ALTITUDE).
    """
    n = data.shape[0]
    alt = data[:, IDX_ALT]
    g = data[:, IDX_G]
    alpha = data[:, IDX_ALPHA]

    # 1. ¿Cuándo despegó realmente?
    mask_airborne = alt > SAFE_ALTITUDE
    
    if not np.any(mask_airborne):
        # CASO RARO: El CSV dice "IMPACTO" pero el avión nunca subió de 200ft.
        # Puede ser un accidente en despegue o taxi.
        # En este caso, no confiamos en "SAFE_ALTITUDE" y buscamos fallas desde el frame 50 (1 seg)
        start_search = 50 
    else:
        # CASO NORMAL: Despegó, voló y luego se estrelló.
        # Empezamos a buscar problemas SOLO después de cruzar 200ft.
        start_search = np.argmax(mask_airborne)

    # Analizamos solo la fase post-despegue
    alt_phase = alt[start_search:]
    g_phase = g[start_search:]
    alpha_phase = alpha[start_search:]

    # A. Impacto Terreno (Baja de 50ft DESPUÉS de haber subido)
    mask_ground = alt_phase < IMPACT_ALTITUDE
    idx_ground = np.argmax(mask_ground) if np.any(mask_ground) else n
    
    # B. Falla Estructural
    mask_struct = g_phase > STRUCTURAL_G
    idx_struct = np.argmax(mask_struct) if np.any(mask_struct) else n
    
    # C. Caos Numérico
    mask_chaos = np.abs(alpha_phase) > MAX_ABS_ALPHA
    idx_chaos = np.argmax(mask_chaos) if np.any(mask_chaos) else n

    # El primer fallo relativo al inicio de búsqueda
    first_fail_rel = min(idx_ground, idx_struct, idx_chaos)
    
    # Si no encontró nada en la fase de vuelo, devolvemos todo (no corta)
    if first_fail_rel == n:
        return n

    # Índice absoluto
    cut_idx_abs = start_search + first_fail_rel
    
    return max(0, cut_idx_abs - BUFFER_FRAMES)

def recalculate_ml_targets(labels):
    n = len(labels)
    if n==0: return np.array([]), np.array([])
    horizon = 250
    fut = np.zeros(n, dtype=np.float32)
    fut[:-horizon] = labels[horizon:] if n>horizon else labels[-1]
    fut[-horizon:] = labels[-1]
    
    stall_idxs = np.where(labels==2)[0]
    tts = np.full(n, 30.0, dtype=np.float32)
    if len(stall_idxs)>0:
        idxs = np.arange(n)
        ptr = np.searchsorted(stall_idxs, idxs)
        mask = ptr < len(stall_idxs)
        valid = idxs[mask]
        next_ev = stall_idxs[ptr[mask]]
        tts[mask] = np.clip((next_ev - valid)*0.02, 0.0, 30.0)
    return fut, tts

# ==========================================
# 3. EJECUCIÓN
# ==========================================
print(f"--- Fusión Final: CSV Truth + Protección de Despegue ---")

with h5py.File(HDF5_PATH, 'r+') as f:
    grp = f['dynamic_states']
    sim_keys = list(grp.keys())
    
    stats = {'RECUPERACION': 0, 'IMPACTO': 0, 'UNKNOWN': 0}
    frames_trimmed = 0
    
    for i, key in enumerate(sim_keys):
        dset = grp[key]
        
        # Identificar Outcome
        orig_name = dset.attrs.get('original_file', b'').decode() if isinstance(dset.attrs.get('original_file'), bytes) else dset.attrs.get('original_file', '')
        outcome_csv = outcome_map.get(orig_name.replace('.csv',''), 'UNKNOWN')
        
        # Cargar
        data = dset[:]
        orig_len = len(data)
        
        # Decisión de Corte
        if outcome_csv == 'IMPACTO':
            # AQUÍ ESTÁ LA CORRECCIÓN: Usamos la función inteligente
            cut_idx = encontrar_punto_impacto_inteligente(data)
            stats['IMPACTO'] += 1
        elif outcome_csv == 'RECUPERACION':
            cut_idx = orig_len
            stats['RECUPERACION'] += 1
        else:
            cut_idx = orig_len
            stats['UNKNOWN'] += 1
            
        # Aplicar
        final_data = data[:cut_idx]
        if len(final_data) < 10: final_data = data # Safety check: si borra todo, mejor no cortar
        frames_trimmed += (orig_len - len(final_data))
        
        # Etiquetas v7 y ML
        new_labels = calcular_etiquetas_v7_numpy(final_data[:, IDX_ALPHA], final_data[:, IDX_G], final_data[:, IDX_VZ], final_data[:, IDX_PITCH], final_data[:, IDX_ROLL], final_data[:, IDX_FLAPS])
        final_data[:, IDX_LABEL] = new_labels
        final_data[:, IDX_FUTURE], final_data[:, IDX_TTS] = recalculate_ml_targets(new_labels)
        
        # Guardar
        saved_attrs = dict(dset.attrs); saved_attrs['outcome'] = outcome_csv
        del grp[key]
        new_dset = grp.create_dataset(key, data=final_data, dtype='float32', compression='gzip')
        for k,v in saved_attrs.items(): new_dset.attrs[k] = v
            
        print(f"[{i+1}/{len(sim_keys)}] {key}: {outcome_csv} | Trim: {orig_len}->{len(final_data)}", end='\r')

print(f"\nTerminado. Frames recortados: {frames_trimmed}")