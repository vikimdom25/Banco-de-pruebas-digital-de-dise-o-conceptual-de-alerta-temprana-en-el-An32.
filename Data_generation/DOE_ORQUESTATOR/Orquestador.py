import pandas as pd

# ================= CONFIGURACIÓN =================
CSV_PATH = r"C:\Users\santi\OneDrive\Documentos\Orquestador v1\DOE_LHS_300_sims.csv"
DECIMALES = 8 # redondeo para Nasal (ajústalo si quieres)

# ================= CARGAR TABLA =================
df = pd.read_csv(CSV_PATH, sep=";",decimal=",")

# ================= BUSCAR SIMULACIÓN PENDIENTE =================
pendientes = df.index[df["DONE"] == False]

if len(pendientes) == 0:
    print("Todas las simulaciones ya fueron ejecutadas.")
    exit()

i = pendientes[0]
row = df.loc[i]

# ================= IMPRIMIR BLOQUE NASAL =================
print("\n# ================= ALEATORIZACIONES (LHS SIM {:03d}) =================\n".format(i+1))

def f(x):
    return round(float(x), DECIMALES)

print(f"alt_flaps_fin          = {f(row.alt_flaps_fin)};")
print(f"alt_crucero_objetivo   = {f(row.alt_crucero_objetivo)};")
print(f"CLIMB_THR              = {f(row.CLIMB_THR)};")
print(f"CRUISE_THR             = {f(row.CRUISE_THR)};")
print(f"PITCH_LEVEL_FLIGHT     = {f(row.PITCH_LEVEL_FLIGHT)};")
print(f"tiempo_inicio_perturbacion = {f(row.tiempo_inicio_perturbacion)};")
print(f"duracion_gust          = {f(row.duracion_gust)};")
print(f"duracion_pulso         = {f(row.duracion_pulso)};")
print(f"pulso_elevator_mag     = {f(row.pulso_elevator_mag)};")
print(f"viento_magnitud        = {f(row.viento_magnitud)};")
print(f"flaps_post_ascenso     = {f(row.flaps_post_ascenso)};")
print(f"roll_target_turn       = {f(row.roll_target_turn)};")

print("\n# --- ASIMETRÍA ---")
print(f"stall_asimetrica = {int(row.stall_asimetrica)};")
print(f"sideslip_ref     = {f(row.sideslip_ref)};")

# ================= MARCAR COMO EJECUTADA =================
df.loc[i, "DONE"] = True
df.to_csv(CSV_PATH, index=False, sep=";")

print(f"\nSimulación {i+1} marcada como DONE.")
