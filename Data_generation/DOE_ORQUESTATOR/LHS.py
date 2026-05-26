import numpy as np
import pandas as pd
from scipy.stats import qmc
import os

# ================= CONFIGURACIÓN GLOBAL =================
N = 300
SEED = 42   # <-- SEMILLA FIJA (reproducibilidad total)

# >>>>> AJUSTA ESTA RUTA A TU GUSTO <<<<<
OUTPUT_DIR = r"C:\Users\santi\OneDrive\Documentos\Orquestador v1"

rng = np.random.default_rng(SEED)

# Variables LHS: (nombre, min, max)
variables = [
    ("alt_flaps_fin", 1800.0, 2300.0),
    ("alt_crucero_objetivo", 4000.0, 11000.0),
    ("CLIMB_THR", 0.80, 1.00),
    ("CRUISE_THR", 0.30, 0.50),
    ("PITCH_LEVEL_FLIGHT", 0.0, 20.0),
    ("tiempo_inicio_perturbacion", 50.0, 90.0),
    ("duracion_gust", 10.0, 30.0),
    ("duracion_pulso", 5.0, 30.0),
    ("pulso_elevator_mag", -1.0, 0.0),
    ("viento_magnitud", 20.0, 100.0),
    ("flaps_post_ascenso", 0.0, 1.0),
    ("roll_target_turn", 15.0, 60.0),
]

DIM = len(variables)

# ================= CREAR DIRECTORIO DE SALIDA =================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LHS =================
sampler = qmc.LatinHypercube(d=DIM, seed=SEED)
lhs_unit = sampler.random(n=N)  # valores en [0,1]

data = {}

for i, (name, vmin, vmax) in enumerate(variables):
    data[name] = vmin + lhs_unit[:, i] * (vmax - vmin)

# ================= BERNOULLI + SIDESLIP =================
stall_asimetrica = rng.binomial(1, 0.5, size=N)
sideslip_ref = np.zeros(N)

mask = stall_asimetrica == 1
sideslip_ref[mask] = rng.uniform(0.0, 15.0, size=mask.sum())

data["stall_asimetrica"] = stall_asimetrica
data["sideslip_ref"] = sideslip_ref

# ================= CONTROL DEL GOBERNADOR =================
data["DONE"] = [False] * N

# ================= EXPORTAR CSV y EXCEL =================
df = pd.DataFrame(data)

csv_path   = os.path.join(OUTPUT_DIR, "DOE_LHS_300_sims.csv")
excel_path = os.path.join(OUTPUT_DIR, "DOE_LHS_300_sims.xlsx")

df.to_csv(csv_path, index=False, sep=";", decimal=",")
df.to_excel(excel_path, index=False)

print("===================================================")
print("DOE LHS generado correctamente")
print(f"Semilla utilizada: {SEED}")
print("Archivos creados en:")
print(csv_path)
print(excel_path)
print("===================================================")
