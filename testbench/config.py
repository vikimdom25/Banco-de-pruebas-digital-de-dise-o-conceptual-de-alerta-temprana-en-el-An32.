import os

# ==========================================
# CONSTANTES DE DOMINIO Y FÍSICAS
# ==========================================
FRECUENCIA_HZ = 50
DT = 1.0 / FRECUENCIA_HZ
HORIZONTE_PREDICCION_S = 5.0
VENTANA_BUFFER = 500  # 10 segundos a 50Hz

# Límites físicos para escalado (Soft-Clipping)
ALPHA_MIN_FISICO = -6.0
ALPHA_MAX_FISICO = 24.0

# Reglas de FSM y Vuelo
MTOW_KG = 27000
SAFE_ALTITUDE_FT = 200.0

# Vector de entrada estricto esperado por el modelo V8
VECTOR_ORDENADO = [
    'throttle', 'flap-pos-norm', 'elevator-pos-norm', 'left-aileron-pos-norm',
    'rudder-pos-norm', 'altitude-ft', 'pitch-deg', 'roll-deg', 'alpha-deg',
    'side-slip-deg', 'airspeed-kt', 'vertical-speed-fps', 'q_rad_sec',
    'p_rad_sec', 'r_rad_sec', 'nlf', 'airspeed-kt_dot', 'nlf_dot', 'alpha-deg_dot'
]

# Columnas requeridas para calcular derivadas temporales
COLS_CINEMATICAS = [
    'nlf_dot', 'alpha-deg_dot', 'airspeed-kt_dot',
    'p_rad_sec', 'q_rad_sec', 'r_rad_sec',
    'side-slip-deg', 'vertical-speed-fps', 'pitch-deg', 'roll-deg'
]

# ==========================================
# RUTAS RELATIVAS DEL PROYECTO
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Datos
RUTA_HDF5 = os.path.join(BASE_DIR, 'DATA', 'dataset_Transformer_Final_Escalado.h5')
RUTA_CSV_FALLBACK = os.path.join(BASE_DIR, 'DATA', 'log_datos_2026-02-13_23-14-49.csv')

# Modelo y Scalers
RUTA_MODELO_V8 = os.path.join(BASE_DIR, 'Models', 'antonov_v8_produccion (1).pt')
DIR_SCALERS = os.path.join(BASE_DIR, 'Models', 'scalers')
