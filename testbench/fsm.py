import numpy as np
import pandas as pd

# Umbrales FSM desde Herramienta de construccion
WARN_G = 1.4
CRIT_ROLL = 80.0
CRIT_DIVE_VZ = -40.0
CRIT_DIVE_GAMMA = -20.0
KEEP_DIVE_VZ = -15.0
KEEP_DIVE_GAMMA = -5.0
RESET_VZ = -5.0
RESET_PITCH = -5.0

def calcular_fsm_etiquetas(df):
    """
    Calcula el ground truth de las etiquetas de riesgo (0: Normal, 1: Advertencia, 2: Pérdida, 3: Picado).
    Las columnas esperadas son:
      - alpha-deg (o alpha)
      - nlf (o g_load)
      - vertical-speed-fps (o vz, ojo unidades, asumimos consistencia de entrada original en ft/s o m/s)
        (En FG vertical-speed-fps = vz, pitch-deg = pitch, roll-deg = roll)
      - pitch-deg
      - roll-deg
      - flap-pos-norm (o flaps_val)
    """

    # Manejar posibles alias
    alpha = df['alpha-deg'].to_numpy() if 'alpha-deg' in df else df['alpha'].to_numpy()
    g_load = df['nlf'].to_numpy() if 'nlf' in df else df['g_load'].to_numpy()
    vz = df['vertical-speed-fps'].to_numpy() if 'vertical-speed-fps' in df else df['vz'].to_numpy()
    pitch = df['pitch-deg'].to_numpy() if 'pitch-deg' in df else df['pitch'].to_numpy()
    roll = df['roll-deg'].to_numpy() if 'roll-deg' in df else df['roll'].to_numpy()
    flaps_val = df['flap-pos-norm'].to_numpy() if 'flap-pos-norm' in df else df['flaps_val'].to_numpy()

    n = len(alpha)
    states = np.zeros(n, dtype=np.int8)
    gamma = pitch - alpha

    if np.isscalar(flaps_val):
        stall_limit = np.full(n, 16.0 - 3.0 * flaps_val)
    else:
        stall_limit = 16.0 - (3.0 * flaps_val)

    event_active = False

    for i in range(n):
        lim = stall_limit[i]
        trig_red = (alpha[i] > lim) or (abs(roll[i]) > CRIT_ROLL)
        trig_dive = (vz[i] < CRIT_DIVE_VZ) or (gamma[i] < CRIT_DIVE_GAMMA)

        if trig_red:
            s, event_active = 2, True
        elif trig_dive:
            s, event_active = 3, True
        elif event_active:
            if (vz[i] < KEEP_DIVE_VZ) or (gamma[i] < KEEP_DIVE_GAMMA):
                s = 3
            elif (vz[i] > RESET_VZ) and (pitch[i] > RESET_PITCH) and (not trig_red):
                s, event_active = 0, False
            else:
                s = 1 if ((alpha[i] > lim-3) or (g_load[i] > WARN_G)) else 0
        else:
            s = 1 if ((alpha[i] > lim-3) or (g_load[i] > WARN_G)) else 0

        states[i] = s

    # Aplicar filtrado de mediana (ventana 5)
    states_series = pd.Series(states).rolling(5, center=True).median().fillna(0).astype(np.int8)
    return states_series.to_numpy()
