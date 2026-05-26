import pandas as pd
import numpy as np
import h5py
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# ==============================================================================
# --- CONFIGURACIÓN DE RUTAS ---
# ==============================================================================
RUTA_ENTRADA = r"C:\Users\santi\OneDrive\Documentos\visualizer\EDAs\dataset_optimizado_transformer.h5"
RUTA_SALIDA = r"C:\Users\santi\OneDrive\Documentos\visualizer\EDAs\dataset_Transformer_Final_Escalado2.h5"
RUTA_GRAFICA = r"C:\Users\santi\OneDrive\Documentos\visualizer\EDAs\Distribuciones_Escaladas_Finales2.png"

# Vector estricto de entrada para el modelo
VECTOR_ORDENADO = [
    'throttle', 'flap-pos-norm', 'elevator-pos-norm', 'left-aileron-pos-norm', 
    'rudder-pos-norm', 'altitude-ft', 'pitch-deg', 'roll-deg', 'alpha-deg', 
    'side-slip-deg', 'airspeed-kt', 'vertical-speed-fps', 'q_rad_sec', 
    'p_rad_sec', 'r_rad_sec', 'nlf', 'airspeed-kt_dot', 'nlf_dot', 'alpha-deg_dot'
]

COLUMNAS_ETIQUETAS = ['timestamp_ms', 'ID_Vuelo', 'label', 'future_label', 'time_to_stall']

def graficar_panel_escalado(df_plot, ruta_guardado):
    """Genera un panel 5x4 con las distribuciones ya escaladas."""
    print("\n   -> Generando panel de distribuciones escaladas (Solo datos de Train)...")
    
    vars_a_graficar = VECTOR_ORDENADO + ['time_to_stall']
    vars_presentes = [v for v in vars_a_graficar if v in df_plot.columns]
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 18))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")
    
    for i, col in enumerate(vars_presentes):
        color = '#E74C3C' if col == 'time_to_stall' else '#2ECC71'
        sns.histplot(df_plot[col].dropna(), bins=60, kde=True, color=color, ax=axes[i], edgecolor='none')
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='both', which='major', labelsize=8)

    for j in range(len(vars_presentes), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Distribuciones Definitivas (Train Set - Escalado Físico + Soft-Clipping)", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> ¡Gráfica guardada en: {ruta_guardado}!")

def preprocesar_sin_leakage(ruta_in, ruta_out, ruta_img):
    print("1. Cargando dataset masivo en memoria (~1GB)...")
    df = pd.read_hdf(ruta_in, key='data')
    
    col_nz = 'nlf' if 'nlf' in df.columns else 'pilot-z-accel-fps_sec'
    col_nz_dot = f"{col_nz}_dot"

    # ==============================================================================
    # --- FASE 1: SPLIT FÍSICO (VUELOS COMPLETOS) ---
    # ==============================================================================
    print("\n2. Realizando SPLIT por Vuelos (Evitando Data Leakage)...")
    todos_los_vuelos = df['ID_Vuelo'].unique()
    
    # 70% Train, 30% Temp (Val + Test)
    vuelos_train, vuelos_temp = train_test_split(todos_los_vuelos, test_size=0.30, random_state=42)
    # Del 30% Temp -> 15% Val, 15% Test
    vuelos_val, vuelos_test = train_test_split(vuelos_temp, test_size=0.50, random_state=42)
    
    print(f"   -> Train: {len(vuelos_train)} vuelos | Val: {len(vuelos_val)} vuelos | Test: {len(vuelos_test)} vuelos")

    # ==============================================================================
    # --- FASE 2: FIT (ENTRENAMIENTO DE SCALERS SOLO EN TRAIN) ---
    # ==============================================================================
    print("\n3. Extrayendo muestra SOLO DE TRAIN para entrenar Scalers...")
    df_train_solo = df[df['ID_Vuelo'].isin(vuelos_train)]
    df_sample_train = df_train_solo.sample(frac=0.05, random_state=42).copy()
    
    del df_train_solo
    gc.collect()

    print("   -> Ajustando Scalers estrictamente con datos de Entrenamiento...")
    cols_cinematicas = [col_nz_dot, 'alpha-deg_dot', 'airspeed-kt_dot', 
                        'p_rad_sec', 'q_rad_sec', 'r_rad_sec', 
                        'side-slip-deg', 'vertical-speed-fps', 'pitch-deg', 'roll-deg']
    
    standard_scaler = StandardScaler().fit(df_sample_train[cols_cinematicas])
    standard_scaler_nz = StandardScaler().fit(df_sample_train[[col_nz]] - 1.0)
    minmax_speed_scaler = MinMaxScaler().fit(df_sample_train[['airspeed-kt']])
    minmax_alt_scaler = MinMaxScaler().fit(df_sample_train[['altitude-ft']])

    # === NUEVO: DEFINICIÓN DE LÍMITES FÍSICOS PARA ALPHA ===
    # Valores extraídos del EDA empírico (Percentil 1% a 99.9%)
    ALPHA_MIN_FISICO = -6.0
    ALPHA_MAX_FISICO = 24.0

    directorio_base = os.path.dirname(RUTA_SALIDA)

    import joblib 
    # 2. Creamos las rutas absolutas para cada archivo .pkl
    ruta_std = os.path.join(directorio_base, 'standard_scaler_cinematica.pkl')
    ruta_nz = os.path.join(directorio_base, 'standard_scaler_nz.pkl')
    ruta_spd = os.path.join(directorio_base, 'minmax_speed_scaler.pkl')
    ruta_alt = os.path.join(directorio_base, 'minmax_alt_scaler.pkl')

    # 3. Guardamos los scalers directamente en la carpeta EDAs
    joblib.dump(standard_scaler, ruta_std)
    joblib.dump(standard_scaler_nz, ruta_nz)
    joblib.dump(minmax_speed_scaler, ruta_spd)
    joblib.dump(minmax_alt_scaler, ruta_alt)

    print(f"\n -> Scalers guardados exitosamente en: {directorio_base}")

    # --- APLICAR A LA MUESTRA PARA GRAFICAR ---
    
    # === NUEVO: TRANSFORMACIÓN DE ALPHA ===
    # 1. Hard Clip a los límites físicos
    alpha_clipped_sample = np.clip(df_sample_train['alpha-deg'], ALPHA_MIN_FISICO, ALPHA_MAX_FISICO)
    # 2. Escalado MinMax manual a rango [-1, 1] usando los límites físicos
    df_sample_train['alpha-deg'] = 2.0 * ((alpha_clipped_sample - ALPHA_MIN_FISICO) / (ALPHA_MAX_FISICO - ALPHA_MIN_FISICO)) - 1.0

    df_sample_train[cols_cinematicas] = np.tanh(standard_scaler.transform(df_sample_train[cols_cinematicas]) / 3.0)
    df_sample_train[col_nz] = np.tanh(standard_scaler_nz.transform(df_sample_train[[col_nz]] - 1.0) / 3.0)
    df_sample_train[['airspeed-kt']] = minmax_speed_scaler.transform(df_sample_train[['airspeed-kt']])
    df_sample_train[['altitude-ft']] = minmax_alt_scaler.transform(df_sample_train[['altitude-ft']])
    df_sample_train['time_to_stall'] = df_sample_train['time_to_stall'] / 30.0
    
    if col_nz != 'nlf':
        df_sample_train.rename(columns={col_nz: 'nlf', col_nz_dot: 'nlf_dot'}, inplace=True)
        
    graficar_panel_escalado(df_sample_train, ruta_img)
    del df_sample_train
    gc.collect()

    # ==============================================================================
    # --- FASE 3: TRANSFORM (APLICAR A TODO EL DATASET) ---
    # ==============================================================================
    print("\n4. Aplicando transformaciones (Escalado Físico + Soft-Clipping) a todo el dataset...")

    # === NUEVO: TRANSFORMACIÓN FINAL DE ALPHA PARA EL DATASET ===
    alpha_clipped = np.clip(df['alpha-deg'], ALPHA_MIN_FISICO, ALPHA_MAX_FISICO)
    df['alpha-deg'] = (2.0 * ((alpha_clipped - ALPHA_MIN_FISICO) / (ALPHA_MAX_FISICO - ALPHA_MIN_FISICO)) - 1.0).astype('float32')
    
    df_cinematicas_std = standard_scaler.transform(df[cols_cinematicas])
    df[cols_cinematicas] = np.tanh(df_cinematicas_std / 3.0).astype('float32')
    del df_cinematicas_std
    
    df[col_nz] = df[col_nz] - 1.0
    nz_std = standard_scaler_nz.transform(df[[col_nz]])
    df[[col_nz]] = np.tanh(nz_std / 3.0).astype('float32')
    del nz_std
    
    df[['airspeed-kt']] = minmax_speed_scaler.transform(df[['airspeed-kt']]).astype('float32')
    df[['altitude-ft']] = minmax_alt_scaler.transform(df[['altitude-ft']]).astype('float32')
    df['time_to_stall'] = (df['time_to_stall'] / 30.0).astype('float32')

    gc.collect()

    # ==============================================================================
    # --- FASE 4: GUARDADO JERÁRQUICO DIVIDIDO ---
    # ==============================================================================
    print("\n5. Reconstruyendo estructura HDF5 dividida (Train/Val/Test)...")
    os.makedirs(os.path.dirname(ruta_out), exist_ok=True)
    
    if col_nz != 'nlf':
        df.rename(columns={col_nz: 'nlf', col_nz_dot: 'nlf_dot'}, inplace=True)
        
    vuelos = df.groupby('ID_Vuelo')
    
    with h5py.File(ruta_out, 'w') as hf:
        grupo_train = hf.create_group('train')
        grupo_val = hf.create_group('val')
        grupo_test = hf.create_group('test')
        
        for id_vuelo, df_vuelo in vuelos:
            df_vuelo = df_vuelo.sort_values('timestamp_ms')
            
            matriz_features = df_vuelo[VECTOR_ORDENADO].values.astype('float32')
            matriz_labels = df_vuelo[COLUMNAS_ETIQUETAS].values.astype('float32')
            
            nombre_sim = f"sim_{int(id_vuelo):03d}"
            
            # Enrutar el vuelo a su carpeta correspondiente
            if id_vuelo in vuelos_train:
                grupo_sim = grupo_train.create_group(nombre_sim)
            elif id_vuelo in vuelos_val:
                grupo_sim = grupo_val.create_group(nombre_sim)
            else:
                grupo_sim = grupo_test.create_group(nombre_sim)
            
            dset_features = grupo_sim.create_dataset('features', data=matriz_features, compression="gzip")
            dset_labels = grupo_sim.create_dataset('labels', data=matriz_labels, compression="gzip")
            
            dset_features.attrs['feature_names'] = VECTOR_ORDENADO
            dset_labels.attrs['label_names'] = COLUMNAS_ETIQUETAS
            dset_labels.attrs['ID_Vuelo'] = id_vuelo

    print(f"\n¡ÉXITO! Dataset salvado de forma estricta y sin Data Leakage.")
    print("Estructura final:")
    print("  /train/sim_XXX")
    print("  /val/sim_YYY")
    print("  /test/sim_ZZZ")

if __name__ == "__main__":
    preprocesar_sin_leakage(RUTA_ENTRADA, RUTA_SALIDA, RUTA_GRAFICA)
