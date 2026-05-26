# ================= DEBUG =================
debug_enabled = 1;
#================== CONVERSIONES ====================
var RAD2DEG = 57.2957795;
var D2R = (1/RAD2DEG);
# ================= VARIABLES DE ESTADO Y CONFIG =================
perdida_activada = 0;
rumbo_objetivo = nil;
g_max = 0.0;
flaps_retraidos = 0;
t_prev = systime();
tiempo_inicio_crucero = nil;
tiempo_en_perdida_activa = 0.0;
fin_crucero = 0;
prev_elev_cmd = 0.0;
roll_deseado = 0.0;     # nivelado por defecto
aoa_trim = nil;

# --- CRUCERO CONVERGENCIA Y ESTADO DE VIRAJE ---
CRUISE_CONV_ALT_TOL_PCT = 0.05; # 5% de alt_crucero_objetivo
CRUISE_CONV_VSPEED_MAX = 0.05; # |vspeed| <= 0.05 fps
CRUISE_CONV_TIME_REQ = 60.0; # 60 segundos
cruise_converged_timer = 0.0;
cruise_submode = "stabilizing"; # Nuevo sub-estado: stabilizing, turning
roll_target_turn = -45.0; # Roll deseado para el viraje (ejemplo: 25 grados)
turn_duration = 120.0; # Duración deseada del viraje en segundos
turn_start_time = nil;

# --- EVENTOS Y PERTURBACIONES DURANTE VIRAJE ---
var llamado1 = 0;
var llamado2 = 0;
var blitz = 0; 

# --- CONTROL PI DE SIDESLIP PARA VIRAJE ---
sideslip_err_sum = 0.0;
KI_SIDESLIP = 0.03; # Ganancia Integral (ajustable)
SIDESLIP_INT_LIMIT = 0.5; # Límite Anti-Windup

# MAQUINA DE ESTADOS
flight_mode = nil;

# --- Integradores / variables para roll & yaw PID ---
cruise_alt_err_sum = 0.0;
cruise_ias_err_sum = 0.0;
pitch_err_sum = 0.0;
ias_err_sum = 0.0;

# roll/yaw integrators & prev values
roll_err_sum = 0.0;
yaw_err_sum = 0.0;
roll_prev = nil;
heading_prev = nil;
heading_error_prev = 0; 

# --- Filtros ---
pitch_filt = nil;
pitch_prev = nil;
elev_cmd_filt = nil;

# --- DETECCION DE PERDIDA (parametros base, luego la función usa otros internos) ---
stall_timer = nil;
STALL_G_THRESHOLD = 1.5;        # g threshold orientativo (usado en heurísticas)
STALL_AOA_THRESHOLD = 14.0;     # deg
STALL_MIN_TIME = 0.5;           # tiempo mínimo para onset detectable (s)
aoa_prev = nil;
# --- ESTABILIDAD DE CRUCERO ---
vs_prev = nil;
vs_stable_time = 0.0;
VS_STABLE_THRESH = 2.0;
VS_ABS_THRESH = 5.0;
VS_STABLE_REQ_TIME = 3.0;
ALT_TOL = 100.0;
CRUISE_TRIM = 0.10;

# ================= PARAMETROS =================
# --- FASE 4: Ascenso ---
KP_PITCH = 0.060;
KI_PITCH = 0.005;
KD_PITCH = 0.01;
KP_IAS2PITCH = 0.0035;
KI_IAS2PITCH = 0.00015;

# --- FASE 5: Crucero (AJUSTADO) ---
KP_PITCH_CRUISE = 0.125;
KI_PITCH_CRUISE = 0.010;
KD_PITCH_CRUISE = 0.010;
KP_ALT2PITCH_CRUISE = 0.0250;
KI_ALT2PITCH_CRUISE = 0.0100;

AOA_TRIM_MIN = -0.3;
AOA_TRIM_MAX = 0.3;
pitch_trim_ref = nil;
PITCH_MAX = 40.0;
PITCH_MIN = -10.0;
ELEV_MIN = -0.8;
ELEV_MAX = 0.8;
ELEV_SLEW_MAX = 0.15;

# --- ROLL / YAW (NUEVAS GAINS calculadas desde KP base) ---
# Base KP: aileron KP = 0.10, rudder KP = 0.20
KP_ROLL = 0.06;
KI_ROLL = 0.005; 	 
KD_ROLL = 0.020;

KP_YAW	= 0.20;
KI_YAW	= 0.005; 	 
KD_YAW	= 0.025;

# Limitadores / anti-windup
ROLL_CMD_MIN = -0.5;
ROLL_CMD_MAX = 0.5;
YAW_CMD_MIN	 = -0.8;
YAW_CMD_MAX	 = 0.8;
ROLL_INT_LIMIT = 0.5; 	# límite al integrador roll
YAW_INT_LIMIT  = 1.0; 	# límite al integrador yaw

# ================= VARIABLES FIJAS y ACOPLADAS ==================
g_threshold = 2.5;
alt_rotacion = 55;
inicio_delay = 2 + 3 * rand();
alt_flaps_fin = 1800 + 500 * rand();
CLIMB_SPEED_KT = 100 + 40 * rand();

# ================= ALEATORIZACIONES =================
alt_crucero_objetivo = 4000 + 7000 * rand();

CLIMB_THR = 0.80 + 0.20 * rand();
CRUISE_THR = 0.65 + 0.35 * rand();
PITCH_LEVEL_FLIGHT = 0 + 20 * rand();
  

# --- ASIMETRÍA POR LAZO (aleatorizada) ---
# Si stall_asimetrica==1 => sideslip_ref random [0,5] deg; else 0
stall_asimetrica = (rand() > 0.5) ? 1 : 0;
sideslip_ref = 0.0;
if (stall_asimetrica == 1) {
    sideslip_ref = 5.0 * rand();
} else {
    sideslip_ref = 0.0;
}

# ================= FUNCIONES AUXILIARES =================

# ---- Pulse elevator: inyecta la diferencia entre elev_objetivo y elev_actual
var pulse_active = 0;
var pulse_end_time = 0;
var pulse_value = 0.0;
var pulse_owner_t = 0.0;

var pulse_elevator = func(elev_objetivo, duracion_pulso) {
    # elev_actual y elev_objetivo en la misma normalización que tu setprop
    pulse_value = elev_objetivo;
    pulse_end_time = systime() + duracion_pulso;
    pulse_active = 1;
    pulse_owner_t = systime();
    if (debug_enabled) print("[PULSE] Elevator pulse scheduled: delta=", sprintf("%.3f", pulse_value), " dur=", sprintf("%.2f", duracion_pulso));
};

# ---- Throttle ramp: degrada throttle (sym) desde valor actual hasta 0 en 'duracion' segundos
var thr_ramp_active = 0;
var thr_ramp_start = 0;
var thr_ramp_duration = 10.0;
var thr_initial = 0.0;

var iniciar_rampa_perdida = func(duracion) {
    thr_ramp_duration = duracion;
    thr_initial = getprop("/controls/engines/engine[0]/throttle");
    thr_ramp_start = systime();
    thr_ramp_active = 1;
    if (debug_enabled) print("[THR_RAMP] iniciada. init=", sprintf("%.3f", thr_initial), " dur=", sprintf("%.1f", duracion));
};

# background adjust function called from main loop (aplica ramp en cada tick)

# ---- Detect_stall: función robusta integrada en NASAL
# Devuelve: 0 = no stall, 1 = imminent, 2 = stall
var metadata_stall_state = 0;    # 0 none, 1 imminent, 2 stall
var metadata_stall_score = 0.0;
var metadata_stall_start = nil;
var metadata_stall_duration = 0.0;

var detect_stall = func(aoa, airspeed, nz, q_rate, d_aoa, beta, dt) {
    # parámetros ajustables
    var aoa_crit = 14.0; # deg
    var d_aoa_max = 25.0; # deg/s (normalización)
    # Estimar IAS_stall razonable: base en CLIMB_SPEED_KT (tiene sentido en tu script)
    var ias_stall = math.max(62.0, 0.75 * airspeed); 

    # Normalizaciones seguras (0..1)
    var s1 = aoa / aoa_crit;
    if (s1 < 0) s1 = 0;
    var s2 = 0.0;
    if (d_aoa > 0) s2 = d_aoa / d_aoa_max;
    s2 = math.min(1.0, s2);
    var s3 = 0.0;
    if (airspeed < ias_stall) s3 = 1.0 - (airspeed / ias_stall);
    s3 = math.max(0.0, math.min(1.0, s3));
    var beta_crit = 8.0;
    var s4 = math.min(1.0, math.abs(beta) / beta_crit);

    # Nz: normal_g risk (si nz muy alto, puede ser accelerated by load factor)
    var nz_norm = 0.0;
    if (nz > 1.0) nz_norm = math.min(1.0, (nz - 1.0) / 1.5); # escala para 1..2.5 g

    # q rate influence (pitch rate)
    var q_norm = math.min(1.0, math.max(0.0, math.abs(q_rate) / 20.0)); # 20 deg/s referencia

    # mezcla (pesos)
    var score = 0.40 * s1 + 0.20 * s2 + 0.15 * s3 + 0.10 * nz_norm + 0.10 * q_norm + 0.05 * s4;
    score = math.max(0.0, math.min(1.0, score));

    # estado segun score
    var new_state = 0;
    if (score >= 0.85) {
        new_state = 2; # stall
    } elsif (score >= 0.65) {
        new_state = 1; # imminent
    } else {
        new_state = 0; # none
    }

    # gestión temporal: requerimos persistencia mínima para confirmar stall
    if (new_state > 0) {
        if (metadata_stall_start == nil) metadata_stall_start = systime();
        metadata_stall_duration = systime() - metadata_stall_start;
    } else {
        metadata_stall_start = nil;
        metadata_stall_duration = 0.0;
    }

    # actualizar metadatos globales
    metadata_stall_state = new_state;
    metadata_stall_score = score;

    # devolver estado
    return new_state;
};

# ---- Recovery helper: rutina que orienta a recovery y resetea banderas
var recovery_start_time = nil;
var recovery_in_progress = 0;

var start_recovery = func() {
    # marca
    recovery_start_time = systime();
    recovery_in_progress = 1;
    perdida_activada = 1;
    flight_mode = "recovery";
    tiempo_en_perdida_activa = 0.0;
    cruise_submode = nil;
    if (debug_enabled) print("[RECOVERY] Iniciada a t=", sprintf("%.2f", recovery_start_time));
    # iniciar rampa de empuje (por ejemplo 8 s)
    iniciar_rampa_perdida(8.0);
};


# ======================= LÓGICA DE VIENTO (REESTRUCTURADA) =======================
# --- VARIABLES GLOBALES PARA CONTROLAR EL VIENTO ---
var viento_activo_manual = 0; # 0 = inactivo, 1 = activo
var v_body_ref = [0, 0, 0]; # Vector de fuerza del viento en BODY (m/s)
var viento_end_time = nil; # Tiempo en systime() cuando debe apagarse el viento

# Función para transformar de BODY a WORLD (NED)
var body_to_world_matrix_func = func(v) {

    var phi   = getprop("/orientation/roll-deg")   * D2R;
    var theta = getprop("/orientation/pitch-deg")  * D2R;
    var psi   = getprop("/orientation/heading-deg")* D2R;

    var cphi   = math.cos(phi);
    var sphi   = math.sin(phi);
    var ctheta = math.cos(theta);
    var stheta = math.sin(theta);
    var cpsi   = math.cos(psi);
    var spsi   = math.sin(psi);

    # Matriz de rotación Body -> World (NED)
    var R = [
        [ ctheta*cpsi,  sphi*stheta*cpsi - cphi*spsi,  cphi*stheta*cpsi + sphi*spsi ],
        [ ctheta*spsi,  sphi*stheta*spsi + cphi*cpsi,  cphi*stheta*spsi - sphi*cpsi ],
        [ -stheta,      sphi*ctheta,                   cphi*ctheta                  ]
    ];

    return [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2]
    ];
};

# =====================================================
#  FUNCION AUXILIAR: UPDATE WIND
#  Ejecutada en cada tick de flight_sequence
# =====================================================
# =====================================================
#  FUNCION AUXILIAR: UPDATE WIND (VERSION CORREGIDA)
# =====================================================
var update_wind = func() {

    if (viento_activo_manual != 1) return;

    if (systime() >= viento_end_time) {
        setprop("/fdm/jsbsim/atmosphere/gust-north-fps", 0.0);
        setprop("/fdm/jsbsim/atmosphere/gust-east-fps",  0.0);
        setprop("/fdm/jsbsim/atmosphere/gust-down-fps",  0.0);

        viento_activo_manual = 0;
        v_body_ref = [0,0,0];
        viento_end_time = nil;

        if (debug_enabled) print("[WIND] Gust finalizado.");
        return;
    }

    var v_world = body_to_world_matrix_func(v_body_ref);

    setprop("/fdm/jsbsim/atmosphere/gust-north-fps", v_world[0]);
    setprop("/fdm/jsbsim/atmosphere/gust-east-fps",  v_world[1]);
    setprop("/fdm/jsbsim/atmosphere/gust-down-fps",  v_world[2]);

    if (debug_enabled) {
        print("[GUST] N:", sprintf("%.1f", v_world[0]),
              " E:", sprintf("%.1f", v_world[1]),
              " D:", sprintf("%.1f", v_world[2]));
    }
};


# =====================================================
# Corriente de viento relativa al avión (BODY)
# (FUNCIÓN DE INICIALIZACIÓN SIN BUCLE INTERNO)
# =====================================================
var aplicarCorrienteViento = func(direccion, magnitud, duracion) {

    if (viento_activo_manual == 1) {
        print("[APLICAR VIENTO] Viento ya activo.");
        return;
    }

    viento_activo_manual = 1;
    viento_end_time = systime() + duracion;

    # 1. Dirección en BODY (de donde viene el viento)
    # Almacenamos el vector en la variable global v_body_ref
    var v = [0,0,0]; # X fwd, Y right, Z down (JSBSim standard)

    if (direccion == "adelante") v = [-magnitud, 0, 0];
    if (direccion == "atras")  v = [ magnitud, 0, 0];
    if (direccion == "derecha")  v = [0, -magnitud, 0];
    if (direccion == "izquierda") v = [0, magnitud, 0];
    if (direccion == "arriba") v = [0, 0, -magnitud];
    if (direccion == "abajo")  v = [0, 0, magnitud];

    v_body_ref = v;
    
    if (debug_enabled) print("[APLICAR VIENTO] Activado: Dir=", direccion, " Mag=", sprintf("%.1f", magnitud), "s Dur=", sprintf("%.1f", duracion), "s");
};


# ================= LOGICA DE VUELO =================
var flight_sequence = func {
    
    
    var airspeed = getprop("/velocities/airspeed-kt");
    var altitude = getprop("/position/altitude-ft");
    var heading = getprop("/orientation/heading-deg");
    var aoa = getprop("/orientation/alpha-deg");
    var vspeed = getprop("/velocities/vertical-speed-fps");
    var roll = getprop("/orientation/roll-deg");
    var pitch_deg = getprop("/orientation/pitch-deg");
    var gload = getprop("/accelerations/pilot-g");
    var sideslip = getprop("/orientation/side-slip-deg"); # Añadido: sideslip
    var q_rate =  getprop("/fdm/jsbsim/velocities/q-rad_sec") * RAD2DEG; # pitch rate

    if (gload > g_max) g_max = gload;

    var throttle_path_0 = "/controls/engines/engine[0]/throttle";
    var throttle_path_1 = "/controls/engines/engine[1]/throttle";
    var elevator_path = "/controls/flight/elevator";
    var rudder_path = "/controls/flight/rudder";
    var elevator_trim_path = "/controls/flight/elevator-trim";
    var aileron_path = "/controls/flight/aileron";
    var gear_path = "/controls/gear/gear-down";
    var flaps_path = "/controls/flight/flaps";

    # --- Actualizar viento ---
    update_wind();
    # --- Medir estabilidad VS ---
    var dt = systime() - t_prev;
    if (dt <= 0) dt = 0.05;

    if (vs_prev == nil) vs_prev = vspeed;
    if (roll_prev == nil) roll_prev = roll;
    if (heading_prev == nil) heading_prev = heading;
    if (aoa_prev == nil) aoa_prev = aoa;

    var vs_delta = math.abs(vspeed - vs_prev);
    if (math.abs(vspeed) <= VS_ABS_THRESH and vs_delta <= VS_STABLE_THRESH) {
        vs_stable_time += dt;
    } else {
        vs_stable_time = 0.0;
    }
    vs_prev = vspeed;

    # --------------------------
    # Manejo de rampa de throttle (si está activa)
    # --------------------------
    if (thr_ramp_active == 1) {
        var t = systime() - thr_ramp_start;
        if (t < thr_ramp_duration) {
            var k = 1 - (t / thr_ramp_duration);
            var thr_cmd = thr_initial * k;
            setprop(throttle_path_0, thr_cmd);
            setprop(throttle_path_1, thr_cmd);
        } else {
            thr_ramp_active = 0;
            setprop(throttle_path_0, 0.0);
            setprop(throttle_path_1, 0.0);
            if (debug_enabled) print("[THR_RAMP] finalizada, throttles a 0.");
        }
    }

    # --------------------------
    # Si estamos en recovery, ejecutar rutinas de recuperación
    # --------------------------
    if (flight_mode == "recovery") {
        # Procedimiento de recuperación suave (nose down + nivelado)
        var elev_recovery = -0.12; # nariz abajo suave (normalizado)
        setprop(elevator_path, math.clamp(elev_recovery, ELEV_MIN, ELEV_MAX));

        # Nivelar alas con aileron proporcional simple
        var ail_cmd = math.clamp(-0.03 * roll, ROLL_CMD_MIN, ROLL_CMD_MAX);
        setprop(aileron_path, ail_cmd);

        # Neutralizar rudder
        setprop(rudder_path, 0.0);

        # Dejar que la rampa de thrust (si está corriendo) haga su trabajo
        # Condición de salida de recovery: aire > 90 kt y pitch < 5deg y score < 0.6
        if (airspeed > 90 and pitch_deg < 5 and metadata_stall_score < 0.6) {
            # restablecer flags y volver a modo crucero estabilizing
            if (debug_enabled) print("[RECOVERY] Condiciones cumplidas. Saliendo de recovery.");
            flight_mode = "cruise";
            cruise_submode = "stabilizing";
            perdida_activada = 0;
            recovery_in_progress = 0;
            metadata_stall_state = 0;
            metadata_stall_score = 0.0;
            metadata_stall_start = nil;
            metadata_stall_duration = 0.0;
            # restablecer throttles a CRUISE_THR tras recuperar
            setprop(throttle_path_0, CRUISE_THR);
            setprop(throttle_path_1, CRUISE_THR);
        }

        t_prev = systime();
        settimer(flight_sequence, 0.02);
        return;
    }

    # --------------------------
    # Si no hay pérdida activada, operación normal (climb/cruise)
    # --------------------------
    if (!perdida_activada) {

        # --- RUMBO / ROLL control using PID (replacing simple P & removing perturb) ---
        if (rumbo_objetivo == nil and altitude >= alt_rotacion and altitude < alt_flaps_fin) {
            var offset = (rand() > 0.5 ? 45 + 15 * rand() : -45 - 15 * rand());
            rumbo_objetivo = heading + offset;
            if (rumbo_objetivo >= 360) rumbo_objetivo -= 360;
            if (rumbo_objetivo < 0) rumbo_objetivo += 360;
        }

        var rumbo_ref = (altitude < alt_rotacion or rumbo_objetivo == nil) ? 317 : rumbo_objetivo;
        var heading_error = heading - rumbo_ref;
        if (heading_error > 180) heading_error -= 360;
        if (heading_error < -180) heading_error += 360;

        # --- YAW PID (rudder) ---
        var rudder_cmd = 0.0;

        # Control PI de Sideslip durante el sub-modo "turning"
        if (flight_mode == "cruise" and cruise_submode == "turning") {

            sideslip_ref = 5;

            # Usa sideslip_ref (aleatorizado en inicio) => permite asimetría vía referencia no cero
            var sideslip_err = sideslip - sideslip_ref;

            # 1. Integración con Anti-Windup (Sideslip)
            var tentative_integrator_ss = sideslip_err_sum + sideslip_err * dt;

            # Cómputo del comando PI tentativo
            var rudder_raw =(KP_YAW * 0.75 * sideslip_err); 
            var rudder_cmd_with_tent = rudder_raw + (KI_SIDESLIP * tentative_integrator_ss);

            # 2. Clamp del comando
            var rudder_cmd_final = math.clamp(rudder_cmd_with_tent, YAW_CMD_MIN, YAW_CMD_MAX);

            # 3. Actualización de la integral (Anti-Windup)
            if (rudder_cmd_with_tent == rudder_cmd_final) {
                sideslip_err_sum = math.clamp(tentative_integrator_ss, -SIDESLIP_INT_LIMIT, SIDESLIP_INT_LIMIT);
            } else {
                sideslip_err_sum = sideslip_err_sum; 
            }

            # 4. Comando final
            rudder_cmd = rudder_cmd_final;

            setprop(rudder_path, rudder_cmd);
            
            # Asegura que el integrador de Rumbo esté a cero
            yaw_err_sum = 0.0;

            if (debug_enabled) print("[YAW/RUDDER] Control PI Sideslip. SS:", sprintf("%.2f", sideslip),
                                      " | SS_ref:", sprintf("%.2f", sideslip_ref),
                                      " | I_Sum:", sprintf("%.2f", sideslip_err_sum),
                                      " | Rudder:", sprintf("%.4f", rudder_cmd));

        } else {
            # Lógica PID de Yaw existente para mantener el rumbo fijo (en otros modos)
            # Deadband para evitar microcorrecciones
            var deadband = 0.02;         # grados

            var h_err = heading_error;
            if (abs(h_err) < deadband) {
                h_err = 0;
            }

            # Derivada del error (NO del heading)
            var yaw_deriv = (h_err - heading_error_prev) / dt;
            heading_error_prev = h_err;

            # PID estándar
            var rudder_raw = KP_YAW * (-h_err);

            # Integrador con anti-windup
            var tentative_integrator = yaw_err_sum + (-h_err) * dt;
            var out_with_tent = rudder_raw + (KI_YAW * tentative_integrator) - (KD_YAW * yaw_deriv);
            var out_clamped = math.clamp(out_with_tent, YAW_CMD_MIN, YAW_CMD_MAX);

            if (out_with_tent == out_clamped) {
                yaw_err_sum = math.clamp(tentative_integrator, -YAW_INT_LIMIT, YAW_INT_LIMIT);
            }

            var rudder_cmd = rudder_raw + (KI_YAW * yaw_err_sum) - (KD_YAW * yaw_deriv);
            rudder_cmd = math.clamp(rudder_cmd, YAW_CMD_MIN, YAW_CMD_MAX);

            # Rate limiter para evitar oscilación
            var max_rate = 0.4;      # unidades por segundo
            var max_delta = max_rate * dt;
            var rudder_prev = getprop(rudder_path);
            if (rudder_cmd > rudder_prev + max_delta) rudder_cmd = rudder_prev + max_delta;
            if (rudder_cmd < rudder_prev - max_delta) rudder_cmd = rudder_prev - max_delta;
            rudder_prev = rudder_cmd;

            # Aplicar comando
            setprop(rudder_path, rudder_cmd);

            # Reset del integral de sideslip al salir del modo viraje
            sideslip_err_sum = 0.0;

            heading_prev = heading;   # opcional ahora
        }

        # --- ROLL PID (ailerons) ---

        var roll_error = roll - roll_deseado; 	# positive if rolling right
        var roll_deriv = (roll - roll_prev) / dt;

        var deadband = 0.025;         # grados

        if (abs(roll_error) < deadband) {
            roll_error = 0;
        }

        var aileron_raw = ( - KP_ROLL * roll_error ); 	# negative sign: to counteract roll_error
        var tentative_roll_int = roll_err_sum + (-roll_error) * dt;
        var out_with_tent_roll = aileron_raw + (KI_ROLL * tentative_roll_int) - (KD_ROLL * roll_deriv);
        var out_clamped_roll = math.clamp(out_with_tent_roll, ROLL_CMD_MIN, ROLL_CMD_MAX);
        if (out_with_tent_roll == out_clamped_roll) {
            roll_err_sum = math.clamp(tentative_roll_int, -ROLL_INT_LIMIT, ROLL_INT_LIMIT);
        } else {
            # dont integrate if would saturate
            roll_err_sum = roll_err_sum;
        }
        var aileron_cmd = aileron_raw + (KI_ROLL * roll_err_sum) - (KD_ROLL * roll_deriv);
        aileron_cmd = math.clamp(aileron_cmd, ROLL_CMD_MIN, ROLL_CMD_MAX);
        setprop(aileron_path, aileron_cmd);

        roll_prev = roll;

            # --- FASES 1-3: DESPEGUE Y ASCENSO INICIAL ---
        if(flight_mode == nil){
            
            # NOTA: Python ya se encargó del autostart.
            # Aquí solo asumimos que el motor ya está encendido y despegamos.
            
            # Soltar freno de mano por si acaso
            setprop("/controls/gear/brake-parking", 0);

            if (airspeed < 100) {
                setprop(throttle_path_0, 1.0);
                setprop(throttle_path_1, 1.0);
                setprop(elevator_path, 0.0);
                setprop(elevator_trim_path, -0.1);
                setprop(flaps_path, 1.0);
            }

            if (airspeed >= 100 and altitude < alt_rotacion) {
                setprop(elevator_path, -0.42);
                setprop(elevator_trim_path, -0.20);
                setprop(flaps_path, 1.0);
            }

            if (altitude >= alt_rotacion and altitude < alt_flaps_fin) {
                setprop(elevator_path, -0.10);
                setprop(elevator_trim_path, -0.30);
                setprop(flaps_path, 1.0);
                setprop(gear_path, 0);
                setprop(throttle_path_0, CLIMB_THR);
                setprop(throttle_path_1, CLIMB_THR);
            }
        }

        # --- RETRAER FLAPS Y ENTRAR A MODO CLIMB ---
        if (altitude >= alt_flaps_fin and flaps_retraidos == 0) {
            var f = getprop(flaps_path);
            if (f > 0.90) {
                setprop(flaps_path, math.max(0.90, f - 0.01));
                setprop(throttle_path_0, CLIMB_THR);
                setprop(throttle_path_1, CLIMB_THR);
            } else {
                flaps_retraidos = 1;
                if (flight_mode == nil) {
                    flight_mode = "climb";
                    print("[TRANSICION] Flaps retraidos. Entrando a modo 'climb' (Fase 4).");
                }
            }
        }

        # --- MAQUINA DE ESTADOS ---
        if (flight_mode == "climb") {
            # --- FASE 4: ASCENSO (PITCH REF FIJA -> ELEVADOR) ---
            
            # La nueva referencia de Pitch es FIJA.
            var pitch_ref_climb = PITCH_LEVEL_FLIGHT; 
            
            if (pitch_trim_ref == nil) {
                # Inicializar variables para el PID de Pitch
                pitch_trim_ref = PITCH_LEVEL_FLIGHT;
                aoa_trim = aoa;
                pitch_filt = pitch_deg;
                pitch_prev = pitch_deg; # Inicializar la derivada
                pitch_err_sum = 0.0;
                
                # REINICIAR/ELIMINAR las integrales de IAS que ya NO se usan.
                ias_err_sum = 0.0; 
                if (debug_enabled) print("[CLIMB PITCH REF] Usando pitch_ref_climb fijo: ", sprintf("%.1f", PITCH_LEVEL_FLIGHT));
            }

            # ----------------------------------------------------
            # 1. PID PITCH (Lazo Interior Único)
            # ----------------------------------------------------

            if (pitch_filt == nil) pitch_filt = pitch_deg;
            pitch_filt = 0.90 * pitch_filt + 0.10 * pitch_deg; # Filtro suave
            
            # pitch_ref_climb ahora es PITCH_CLIMB_REF
            var pitch_err = pitch_ref_climb - pitch_deg;
            pitch_err_sum += pitch_err * dt;
            pitch_err_sum = math.clamp(pitch_err_sum, -50.0, 50.0);

            if (pitch_prev == nil) pitch_prev = pitch_filt;
            var pitch_deriv = (pitch_filt - pitch_prev) / dt;
            pitch_prev = pitch_filt;

            var elev_cmd_base = (-1)*((KP_PITCH * pitch_err) + (KI_PITCH * pitch_err_sum) - (KD_PITCH * pitch_deriv));

            # ----------------------------------------------------
            # 2. SLEW RATE y COMANDO FINAL
            # ----------------------------------------------------
            var elev_cmd_raw = elev_cmd_base;
            var delta = elev_cmd_raw - prev_elev_cmd;
            delta = math.clamp(delta, -ELEV_SLEW_MAX, ELEV_SLEW_MAX);
            var elev_cmd = prev_elev_cmd + delta;


            # Aplicación de pulse (se mantiene)
            if (pulse_active == 1) {
                if (systime() < pulse_end_time) {
                    elev_cmd = pulse_value;
                } else {
                    pulse_active = 0;
                    if (debug_enabled) print("[PULSE] finalizado tras ", sprintf("%.2f", systime() - pulse_owner_t), " s");
                }
            }

            setprop(elevator_path, math.clamp(elev_cmd, ELEV_MIN, ELEV_MAX));

            # Throttle fijo en modo climb
            setprop(throttle_path_0, CLIMB_THR);
            setprop(throttle_path_1, CLIMB_THR);

            # -----------------------------------------------------------------
            # 3. TRIM ADAPTATIVO BASADO EN ERROR DE PITCH (Tu lógica propuesta)
            #    Ahora utiliza pitch_ref_climb (PITCH_CLIMB_REF)
            # -----------------------------------------------------------------
            var aoa_trim_ref_calc = math.clamp(-1 * (pitch_ref_climb - pitch_deg) * 0.1, AOA_TRIM_MIN, AOA_TRIM_MAX);
            aoa_trim = 0.990 * aoa_trim + 0.010 * aoa_trim_ref_calc;
            aoa_trim = math.clamp(aoa_trim, AOA_TRIM_MIN, AOA_TRIM_MAX);
            var current_trim = getprop(elevator_trim_path);
            setprop(elevator_trim_path, math.clamp(current_trim + (aoa_trim - current_trim) * 0.01, -0.2, 0.2));

            prev_elev_cmd = elev_cmd;

            # --- Cambio a modo crucero ---
            if (altitude >= alt_crucero_objetivo - 200) {
                flight_mode = "cruise";
                cruise_submode = "stabilizing"; # Inicializa sub-modo
                tiempo_inicio_crucero = nil;
                if (debug_enabled) print("[TRANSICION] Alcanzando altitud de crucero. Cambiando a modo CRUISE.");
            }
        } elsif (flight_mode == "cruise") {

            # --- FASE 5: CRUCERO (ALT->PITCH + TRIM ADAPTATIVO) ---
            if (tiempo_inicio_crucero == nil) {
                tiempo_inicio_crucero = systime();
                if (debug_enabled) print("[FASE 5] Entrada a modo crucero. RESETEANDO VARIABLES.");
                pitch_err_sum = 0.0;
                cruise_alt_err_sum = 0.0;
                ias_err_sum = 0.0;
                cruise_ias_err_sum = 0.0;
                pitch_filt = pitch_deg;
                pitch_prev = pitch_deg;
                prev_elev_cmd = getprop(elevator_path);
                setprop(elevator_trim_path, CRUISE_TRIM);
                setprop(flaps_path, 1.0);
            }

            # --- THROTTLE FIJO (blend inicial) ---
            var tiempo_en_crucero = systime() - tiempo_inicio_crucero;
            var blend_duration = 5.0;
            if (tiempo_en_crucero < blend_duration) {
                var k_thr = tiempo_en_crucero / blend_duration;
                var throttle_cmd = (1.0 - k_thr) * CLIMB_THR + k_thr * CRUISE_THR;
                setprop(throttle_path_0, throttle_cmd);
                setprop(throttle_path_1, throttle_cmd);
            } else {
                # Si hay rampa de pérdida activa, la rampa la controla; si no, dejar CRUISE_THR
                if (!thr_ramp_active) {
                    setprop(throttle_path_0, CRUISE_THR);
                    setprop(throttle_path_1, CRUISE_THR);
                }
            }

            PITCH_MAX = 40.0;
            PITCH_MIN = -20.0;
            ELEV_MIN = -0.9;
            ELEV_MAX = 0.9;

            # Outer loop: ALT -> PITCH_REF (Control longitudinal base, se usa en todos los sub-modos)
            var alt_err = altitude - alt_crucero_objetivo;
            cruise_alt_err_sum += alt_err * dt;
            cruise_alt_err_sum = math.clamp(cruise_alt_err_sum, -500.0, 500.0);
            var pitch_ref_cruise = -(KP_ALT2PITCH_CRUISE * alt_err) - (KI_ALT2PITCH_CRUISE * cruise_alt_err_sum);
            pitch_ref_cruise = math.clamp(pitch_ref_cruise, PITCH_MIN, PITCH_MAX);

            # Inner loop: PITCH -> ELEVATOR
            if (pitch_filt == nil) pitch_filt = pitch_deg;
            pitch_filt = 0.80 * pitch_filt + 0.20 * pitch_deg;
            var pitch_err_cr = pitch_ref_cruise - pitch_filt;
            pitch_err_sum += pitch_err_cr * dt;
            pitch_err_sum = math.clamp(pitch_err_sum, -50.0, 50.0);
            var pitch_deriv_cr = (pitch_filt - pitch_prev) / dt;
            pitch_deriv_cr = math.clamp(pitch_deriv_cr, -5.0, 5.0);
            pitch_prev = pitch_filt;

            var elev_cmd_raw_cr = -(KP_PITCH_CRUISE * pitch_err_cr) -(KI_PITCH_CRUISE * pitch_err_sum) + (KD_PITCH_CRUISE * pitch_deriv_cr);
            var delta_cr = elev_cmd_raw_cr - prev_elev_cmd;
            delta_cr = math.clamp(delta_cr, -ELEV_SLEW_MAX, ELEV_SLEW_MAX);
            var elev_cmd = prev_elev_cmd + delta_cr;
            elev_cmd = elev_cmd;

            # Aplicación de pulse (si activo)
            if (pulse_active == 1) {
                if (systime() < pulse_end_time) {
                    elev_cmd = pulse_value;
                } else {
                    pulse_active = 0;
                    if (debug_enabled) print("[PULSE] finalizado tras ", sprintf("%.2f", systime() - pulse_owner_t), " s");
                }
            }

            setprop(elevator_path, math.clamp(elev_cmd, ELEV_MIN, ELEV_MAX));

            # --- TRIM ADAPTATIVO BASADO EN ALTITUD ---
            var aoa_trim_ref_calc = math.clamp(1 * (alt_err * 0.0020), AOA_TRIM_MIN, AOA_TRIM_MAX);
            aoa_trim = 0.995 * aoa_trim + 0.005 * aoa_trim_ref_calc;
            aoa_trim = math.clamp(aoa_trim, AOA_TRIM_MIN, AOA_TRIM_MAX);
            var current_trim = getprop(elevator_trim_path);
            setprop(elevator_trim_path, math.clamp(current_trim + (aoa_trim - current_trim) * 0.01, -0.5, 0.5));

            prev_elev_cmd = elev_cmd;

            # ----------------------------------------------------
            # --- LÓGICA DE SUB-MODOS DENTRO DE CRUCERO ---
            # ----------------------------------------------------
            if (cruise_submode == "stabilizing") {
                roll_deseado = 0.0; # Nivelado

                # --- Criterio de Convergencia ---
                var alt_err_abs = math.abs(alt_err);
                var alt_tol_abs = alt_crucero_objetivo * CRUISE_CONV_ALT_TOL_PCT;
                var vspeed_abs = math.abs(vspeed);

                var is_stable_alt = (alt_err_abs <= alt_tol_abs);
                var is_stable_vs = (vspeed_abs <= CRUISE_CONV_VSPEED_MAX);

                if (is_stable_alt and is_stable_vs) {
                    cruise_converged_timer += dt;
                    if (debug_enabled) print("[CRUCERO STABILIZING] Convergiendo... T: ", sprintf("%.1f", cruise_converged_timer), "s");
                } else {
                    cruise_converged_timer = 0.0;
                }

                if (cruise_converged_timer >= CRUISE_CONV_TIME_REQ) {
                    cruise_submode = "turning";
                    turn_start_time = systime();
                    # Define un nuevo roll deseado para el viraje
                    roll_deseado = (rand() > 0.5 ? roll_target_turn : -roll_target_turn); 
                    cruise_converged_timer = 0.0;
                    if (debug_enabled) print("[TRANSICION] CRUCERO CONVERGIDO. Iniciando VIRAJAE (Sub-modo: turning). Roll des: ", sprintf("%.1f", roll_deseado), " deg.");
                }

                if (debug_enabled) {
                    print("[FASE 5/STAB] Alt=", sprintf("%.1f", altitude), " | AltErr=", sprintf("%.1f", alt_err),
                          " | PitchRef=", sprintf("%.2f", pitch_ref_cruise), " | VSp=", sprintf("%.2f", vspeed));
                }

            

            } elsif (cruise_submode == "turning") {

                
                YAW_CMD_MIN	 = -1.0;
                YAW_CMD_MAX	 = 1.0;

                var turn_time = systime() - turn_start_time;

                if ((turn_time >= 0.05) and (llamado1 == 0)){

                    iniciar_rampa_perdida(turn_duration);
                    llamado1 = 1; 
                }
                
            
                if ((turn_time >= 60.0) and (llamado2 == 0)) {

                    if (debug_enabled) print("[EVENTO 60s] Ejecutando acciones de 60 segundos.");
                    aplicarCorrienteViento("abajo", 70.0, 30.0);
                    var elev = getprop(elevator_path);
                    # pulse_elevator(-0.4,5.0);

                    llamado2 = 1;  
                }

                if (turn_time >= turn_duration) {
                    # Finaliza el viraje, recupera el nivelado.
                    roll_deseado = 0.0; # Nivelado
                    cruise_submode = "stabilizing";

                    llamado1 = 0;
                    llamado2 = 0; 

                    if (debug_enabled) print("[TRANSICION] Viraje completado. Volviendo a CRUCERO STABILIZING.");
                }

                

                if (debug_enabled) {
                    print("[CRUCERO TURNING] Alt=", sprintf("%.1f", altitude), " | RollDes=", sprintf("%.1f", roll_deseado),
                          " | Roll=", sprintf("%.1f", roll), " | Sideslip=", sprintf("%.2f", sideslip),
                          " | Time=", sprintf("%.1f", turn_time), "LLamado2=",sprintf("%.1f", llamado2));
                }

                # --- DETECCION DE PERDIDA EN TIEMPO REAL (solo en turning) ---
                var d_aoa = (aoa - aoa_prev) / dt;
                aoa_prev = aoa;

                var stall_state_local = detect_stall(aoa, airspeed, gload, q_rate, d_aoa, sideslip, dt);
                # stall_state_local: 0 none,1 imminent,2 stall
                if (debug_enabled) {
                    print("[STALL_DET] state=", stall_state_local, " score=", sprintf("%.3f", metadata_stall_score), " dur(s)=", sprintf("%.2f", metadata_stall_duration));
                }

                # Si detecta stall o inminencia, iniciar acciones: marcar y lanzar rampa
                if (stall_state_local >= 1) {
                    # Si entra por primera vez en stall (o imminent->stall), iniciar rampa y logear
                    if (!perdida_activada) {
                        perdida_activada = 1;
                        if (debug_enabled) print("[EVENT] Perdida detectada (state=", stall_state_local, "). Activando flags y rampa de empuje.");
                        # inicio de rampa: desde el throttle actual hasta 0 en 8 s (ajustable)
                        iniciar_rampa_perdida(8.0);
                    }
                }

                # Si la condición de stall persiste y metadata_stall_duration > 40s -> forzar recovery
                if (metadata_stall_duration >= 40.0 and metadata_stall_state == 2 and recovery_in_progress == 0) {
                    if (debug_enabled) print("[EVENT] Stall persistente >40s. Forzando recovery.");
                    start_recovery();
                }
            }
        }

        t_prev = systime();

        # --- DETECCION DE PERDIDA LEGACY (se mantiene pero ya no bloquea) ---
        # Nota: este bloque se deja por compatibilidad pero la detección real es 'detect_stall'.
        if (gload < STALL_G_THRESHOLD and aoa > STALL_AOA_THRESHOLD and altitude > 20000) {
            if (stall_timer == nil) stall_timer = systime();
            var t_stall = systime() - stall_timer;
            if (t_stall >= STALL_MIN_TIME) {
                perdida_activada = 1;
            }
        } else {
            stall_timer = nil;
        }

    } else {
        # =============================================================
        # ZONA DE PÉRDIDA ACTIVA (CONTADOR FORZADO)
        # =============================================================

        # 1. Corrección del cálculo físico
        var d_aoa = (aoa - aoa_prev) / dt;
        aoa_prev = aoa;  # <--- CORREGIDO (Antes decía oa_prev)

        # 2. Seguimos ejecutando la detección solo para actualizar los logs/score,
        #    pero NO confiaremos en su duración interna porque se resetea con el ruido.
        var stall_state_local = detect_stall(aoa, airspeed, gload, q_rate, d_aoa, sideslip, dt);

        # 3. ACUMULADOR MANUAL:
        #    Como estamos en este 'else', significa que perdida_activada == 1.
        #    Sumamos el tiempo manualmente. Esto no se detiene aunque el sensor fluctúe.
        tiempo_en_perdida_activa += dt;

        # 4. CONDICIÓN DE RESCATE (40 SEGUNDOS):
        #    Usamos nuestra variable manual en vez de metadata_stall_duration
        if (tiempo_en_perdida_activa >= 40.0 and recovery_in_progress == 0) {
            if (debug_enabled) print("[EVENT] Límite de 40s alcanzado. Forzando RECOVERY.");
            start_recovery();
        }

        # 5. DEBUG VISUAL
        if (debug_enabled) {
            print("[PERDIDA ACTIVA] T_Acumulado=", sprintf("%.1f", tiempo_en_perdida_activa), 
                  "s / 40.0s | SensorState=", metadata_stall_state);
        }

        # Mantener logging
        if (debug_enabled) {
            print("[PERDIDA] bandera activa. metadata_state=", metadata_stall_state, " score=", sprintf("%.3f", metadata_stall_score),
                  " dur=", sprintf("%.2f", metadata_stall_duration));
        }
    }

    # Finalmente re-schedule
    t_prev = systime();
    settimer(flight_sequence, 0.02);
};

# =====================================================
# LISTENER DE ARRANQUE MONTECARLO (CONTROL EXTERNO)
# =====================================================

var montecarlo_started = 0;

var start_montecarlo = func {
    if (montecarlo_started) return;

    montecarlo_started = 1;

    print("[MONTECARLO] Señal de inicio recibida. Ejecutando flight_sequence.");

    # Inicializar tiempo base
    t_prev = systime();

    # Lanzar loop principal
    settimer(flight_sequence, 0.02);
};

setlistener("/sim/montecarlo/start", func(v) {
    if (v == 1) {
        start_montecarlo();
    }
});

# Seguridad: por si Python setea antes de que el script cargue
if (getprop("/sim/montecarlo/start") == 1) {
    start_montecarlo();
}

print("[MONTECARLO] Script cargado. Esperando /sim/montecarlo/start");