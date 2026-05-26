class ModuloAlertaEICAS_Hibrido:
    def __init__(self, umbral_peligro=0.80):
        self.umbral  = umbral_peligro
        self.z_score = 1.96

    def evaluar_batch(self, logits_fut_np, pred_mu_np, pred_log_var_np):
        logits_fut  = torch.from_numpy(logits_fut_np).float()
        pred_mu_t   = torch.from_numpy(pred_mu_np).float()
        pred_log_var_t = torch.from_numpy(pred_log_var_np).float()

        probs       = F.softmax(logits_fut, dim=1)
        p_clases    = probs[:, 1] + probs[:, 2] + probs[:, 3]
        riesgo_base = 1.0 - pred_mu_t
        sigma       = torch.exp(0.5 * pred_log_var_t)
        p_max       = torch.maximum(p_clases, riesgo_base)
        lcb         = torch.clamp(p_max - 1.96 * sigma, 0.0, 1.0)
        estados     = torch.zeros_like(p_max, dtype=torch.long)
        estados[lcb >= self.umbral] = 2
        return estados.numpy(), p_max.numpy(), sigma.numpy()


def generar_evidencia_amarilla_dashlink(p_max, sigma, flight_ids_vent,
                                        decay=0.98, threshold=2.5):
    """
    Acumulador de evidencia logística (alarma amarilla).
    Reinicia en cada frontera de instancia DASHlink (= frontera de vuelo).
    """
    evidencia   = 0.0
    pre_alertas = np.zeros(len(p_max), dtype=bool)

    for i in range(len(p_max)):
        if i > 0 and flight_ids_vent[i] != flight_ids_vent[i - 1]:
            evidencia = 0.0

        p        = np.clip(p_max[i], 1e-6, 1 - 1e-6)
        logit    = np.log(p / (1 - p))
        conf     = 1.0 / (1.0 + sigma[i])
        delta_p  = p_max[i] - p_max[i - 1] if i > 0 else 0.0
        momentum = max(delta_p, 0.0)

        inicio = i
        for k in range(i - 1, max(i - 9, -1), -1):
            if flight_ids_vent[k] == flight_ids_vent[i]:
                inicio = k
            else:
                break
        ventana      = p_max[inicio:i + 1]
        consistencia = np.mean(ventana)
        varianza_v   = np.var(ventana)

        score = logit * conf
        if consistencia > 0.55 and varianza_v < 0.04:
            evidencia = evidencia * decay + score * (1.0 + momentum)
        else:
            evidencia *= 0.92

        evidencia      = max(evidencia, 0.0)
        pre_alertas[i] = evidencia >= threshold

    return pre_alertas


print("\n" + "="*60)
print("  9. APLICANDO LÓGICA EICAS REAL")
print("="*60)

# Recuperar log_var desde sigma: log_var = 2 * log(sigma)
todas_log_var = 2.0 * np.log(np.clip(todas_sigma, 1e-6, None))

# Necesitamos logits_futuro completos — los recuperamos de p_fut via log
# NOTA: p_fut ya fue softmax-eado, así que usamos log(p) como proxy de logits
# para el módulo EICAS. No es exacto pero es equivalente para el ranking.
logits_fut_proxy = np.log(np.clip(todas_p_fut, 1e-9, 1.0))

eicas = ModuloAlertaEICAS_Hibrido(umbral_peligro=0.80)
estados_eicas, p_max_eicas, sigma_eicas = eicas.evaluar_batch(
    logits_fut_proxy, todas_mu, todas_log_var
)

# flight_ids por ventana = índice de instancia de origen
# (cada instancia DASHlink es un vuelo independiente)
# Reconstruir desde tamaño de ventanas por instancia
n_vent_por_inst = int((8000 - WINDOW_SIZE) / STRIDE) + 1
flight_ids_vent = np.repeat(np.arange(N_inst), n_vent_por_inst)[:len(todas_etiquetas)]

# Alerta Roja EICAS
alerta_roja_eicas = (estados_eicas == 2)

# Alerta Amarilla EICAS — configuración balanceada
alerta_amarilla_eicas = generar_evidencia_amarilla_dashlink(
    p_max_eicas, sigma_eicas, flight_ids_vent, decay=0.98, threshold=2.5
)
