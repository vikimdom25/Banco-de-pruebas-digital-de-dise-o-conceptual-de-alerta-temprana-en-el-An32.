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

SAFE_ALTITUDE = 200.0
IMPACT_ALTITUDE = 50.0
STRUCTURAL_G = 9.0

class FSMRealtimeEvaluator:
    def __init__(self):
        self.event_active = False
        self.has_taken_off = False
        self.impacted = False
        self.recovered = False

    def evaluate(self, data: dict) -> int:
        """
        Devuelve el estado FSM actual:
        0: Normal
        1: Advertencia
        2: Pérdida (Stall)
        3: Picado (Dive)
        4: Impacto
        5: Recuperación
        """
        if self.impacted:
            return 4

        alt = data.get('altitude-ft', 0)
        alpha = data.get('alpha-deg', 0)
        g_load = data.get('nlf', 0)
        vz = data.get('vertical-speed-fps', 0)
        pitch = data.get('pitch-deg', 0)
        roll = data.get('roll-deg', 0)
        flaps_val = data.get('flap-pos-norm', 0)

        if alt > SAFE_ALTITUDE:
            self.has_taken_off = True

        if self.has_taken_off and alt < IMPACT_ALTITUDE:
            self.impacted = True
            return 4

        if g_load > STRUCTURAL_G:
            self.impacted = True
            return 4

        gamma = pitch - alpha
        stall_limit = 16.0 - (3.0 * flaps_val)

        trig_red = (alpha > stall_limit) or (abs(roll) > CRIT_ROLL)
        trig_dive = (vz < CRIT_DIVE_VZ) or (gamma < CRIT_DIVE_GAMMA)

        if trig_red:
            s, self.event_active = 2, True
            self.recovered = False
        elif trig_dive:
            s, self.event_active = 3, True
            self.recovered = False
        elif self.event_active:
            if (vz < KEEP_DIVE_VZ) or (gamma < KEEP_DIVE_GAMMA):
                s = 3
            elif (vz > RESET_VZ) and (pitch > RESET_PITCH) and (not trig_red):
                s, self.event_active = 5, False # Recuperación!
                self.recovered = True
            else:
                s = 1 if ((alpha > stall_limit-3) or (g_load > WARN_G)) else 0
        else:
            if self.recovered:
                s = 5
            else:
                s = 1 if ((alpha > stall_limit-3) or (g_load > WARN_G)) else 0

        return s

def calcular_fsm_etiquetas(df):
    """
    (Legacy batch processor para mantener compatibilidad con panel historico)
    """
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

    states_series = pd.Series(states).rolling(5, center=True).median().fillna(0).astype(np.int8)
    return states_series.to_numpy()
