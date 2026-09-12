import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import warnings
from collections import deque
from PyQt6.QtCore import QObject, pyqtSlot

from config import (RUTA_MODELO_V8, DIR_SCALERS, VENTANA_BUFFER,
                    VECTOR_ORDENADO, COLS_CINEMATICAS,
                    ALPHA_MIN_FISICO, ALPHA_MAX_FISICO)
from signals import event_bus

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ==========================================
# 1. ARQUITECTURA MODELO V8 (AntonovTransformerModelV6 de training.py)
# ==========================================
class AntonovTransformerModelV6(nn.Module):
    def __init__(self, num_features=19, sequence_length=500,
                 d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256,
                 num_clases_actual=4, num_clases_futuro=4):
        super(AntonovTransformerModelV6, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.input_projection = nn.Linear(num_features, d_model)

        pe = self._create_positional_encoding(sequence_length, d_model)
        self.register_buffer('positional_encoding', pe)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.head_cls_actual = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, num_clases_actual)
        )
        self.head_cls_futuro = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, num_clases_futuro)
        )

        # Tríada de Pooling (CLS + GAP + Last-Token) para el Índice de Riesgo
        self.head_riesgo = nn.Sequential(
            nn.Linear(d_model * 3, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 2) # Salida: [mu, log_var]
        )

        self.attention_maps = {}
        self.extract_attention = False

        for i, layer in enumerate(self.transformer_encoder.layers):
            layer.self_attn.register_forward_pre_hook(self._force_weights_pre_hook, with_kwargs=True)
            layer.self_attn.register_forward_hook(self._get_attention_hook(f'layer_{i}'))

    def _force_weights_pre_hook(self, module, args, kwargs):
        kwargs['need_weights'] = self.extract_attention
        return args, kwargs

    def _get_attention_hook(self, layer_name):
        def hook(module, input, output):
            if self.extract_attention and output[1] is not None:
                self.attention_maps[layer_name] = output[1].detach().cpu()
        return hook

    def forward(self, x, return_attention=False):
        self.extract_attention = return_attention

        batch_size = x.size(0)
        x = self.input_projection(x)
        x = x + self.positional_encoding

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_processed = self.transformer_encoder(x)

        # --- Pooling ---
        cls_representation = x_processed[:, 0, :]
        gap_representation = x_processed[:, 1:, :].mean(dim=1)
        last_token_representation = x_processed[:, -1, :]

        logits_actual = self.head_cls_actual(cls_representation)
        logits_futuro = self.head_cls_futuro(cls_representation)

        # Concatenación de la Tríada para Regresión de Salud/Riesgo
        triada_representation = torch.cat((cls_representation, gap_representation, last_token_representation), dim=1)
        salida_riesgo = self.head_riesgo(triada_representation)

        pred_mu = torch.sigmoid(salida_riesgo[:, 0])
        pred_log_var = salida_riesgo[:, 1]

        if return_attention:
            return logits_actual, logits_futuro, pred_mu, pred_log_var, self.attention_maps
        else:
            return logits_actual, logits_futuro, pred_mu, pred_log_var

    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


# ==========================================
# 2. LOGICA EICAS HIBRIDA
# ==========================================
class ModuloAlertaEICAS_Hibrido:
    def __init__(self, umbral_peligro=0.80, threshold_amarilla=2.5, decay=0.98):
        self.umbral  = umbral_peligro
        self.z_score = 1.96
        self.threshold_amarilla = threshold_amarilla
        self.decay = decay

        # Estado interno para la alarma amarilla (evidencia acumulada)
        self.evidencia_acumulada = 0.0
        self.historial_p_max = deque(maxlen=10) # Para consistencia y varianza
        self.last_p_max = 0.0

    def reset_evidencia(self):
        self.evidencia_acumulada = 0.0
        self.historial_p_max.clear()
        self.last_p_max = 0.0

    def evaluar_instancia(self, logits_fut_t, pred_mu_t, pred_log_var_t):
        probs = F.softmax(logits_fut_t, dim=1)
        p_clases = probs[:, 1] + probs[:, 2] + probs[:, 3]
        riesgo_base = 1.0 - pred_mu_t
        sigma = torch.exp(0.5 * pred_log_var_t)

        p_max_t = torch.maximum(p_clases, riesgo_base)
        lcb = torch.clamp(p_max_t - self.z_score * sigma, 0.0, 1.0)

        p_max = p_max_t.item()
        sigma_val = sigma.item()

        # 1. Alarma Roja (Inminente)
        alerta_roja = False
        if lcb.item() >= self.umbral:
            alerta_roja = True

        # 2. Alarma Amarilla (Acumulación de Evidencia)
        p_clip = np.clip(p_max, 1e-6, 1 - 1e-6)
        logit = np.log(p_clip / (1 - p_clip))
        conf = 1.0 / (1.0 + sigma_val)
        delta_p = p_max - self.last_p_max
        momentum = max(delta_p, 0.0)

        self.historial_p_max.append(p_max)

        alerta_amarilla = False
        if len(self.historial_p_max) > 2:
            consistencia = np.mean(self.historial_p_max)
            varianza_v = np.var(self.historial_p_max)

            score = logit * conf
            if consistencia > 0.55 and varianza_v < 0.04:
                self.evidencia_acumulada = self.evidencia_acumulada * self.decay + score * (1.0 + momentum)
            else:
                self.evidencia_acumulada *= 0.92

            self.evidencia_acumulada = max(self.evidencia_acumulada, 0.0)
            alerta_amarilla = self.evidencia_acumulada >= self.threshold_amarilla

        self.last_p_max = p_max

        return alerta_roja, alerta_amarilla, p_max, sigma_val, riesgo_base.item()


# ==========================================
# 3. WORKER DE INFERENCIA (QThread)
# ==========================================
class ModelWorker(QObject):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ModelWorker] Iniciando Motor V8 en: {self.device}")

        self._cargar_modelo_y_scalers()

        self.buffer = deque(maxlen=VENTANA_BUFFER)
        self.eicas = ModuloAlertaEICAS_Hibrido(umbral_peligro=0.80)

        # Conectar bus de eventos
        event_bus.telemetry_updated.connect(self.procesar_telemetria)
        event_bus.vuelo_cargado.connect(self._limpiar_buffer)

    def _limpiar_buffer(self, _):
        self.buffer.clear()
        self.eicas.reset_evidencia()

    def _cargar_modelo_y_scalers(self):
        try:
            # El archivo actual en el repo es un TorchScript (jit.save), no un state_dict
            self.modelo = torch.jit.load(RUTA_MODELO_V8, map_location=self.device)
            self.modelo.eval()

            self.std_scaler = joblib.load(os.path.join(DIR_SCALERS, 'standard_scaler_cinematica.pkl'))
            self.std_nz_scaler = joblib.load(os.path.join(DIR_SCALERS, 'standard_scaler_nz.pkl'))
            self.minmax_spd = joblib.load(os.path.join(DIR_SCALERS, 'minmax_speed_scaler.pkl'))
            self.minmax_alt = joblib.load(os.path.join(DIR_SCALERS, 'minmax_alt_scaler.pkl'))

            # Extraer parámetros de los scalers para realizar cálculos matemáticos directos (optimización Lightning fast Bolt)
            self.std_mean = self.std_scaler.mean_
            self.std_scale = self.std_scaler.scale_
            self.nz_mean = self.std_nz_scaler.mean_[0]
            self.nz_scale = self.std_nz_scaler.scale_[0]
            self.spd_min = self.minmax_spd.min_[0]
            self.spd_scale = self.minmax_spd.scale_[0]
            self.alt_min = self.minmax_alt.min_[0]
            self.alt_scale = self.minmax_alt.scale_[0]
        except Exception as e:
            print(f"[ModelWorker] ERROR CARGANDO MODELO/SCALERS: {e}")

    def _escalar(self, datos: dict) -> np.ndarray:
        # Preprocesamiento estricto sin data leakage
        alpha_val = datos.get('alpha-deg', 0.0)
        # Optimizar clip si ya está en rango común
        alpha_clipped = alpha_val if ALPHA_MIN_FISICO <= alpha_val <= ALPHA_MAX_FISICO else np.clip(alpha_val, ALPHA_MIN_FISICO, ALPHA_MAX_FISICO)
        datos['alpha-deg'] = 2.0 * ((alpha_clipped - ALPHA_MIN_FISICO) / (ALPHA_MAX_FISICO - ALPHA_MIN_FISICO)) - 1.0

        # Optimización: Reemplazar scikit-learn transform por NumPy math directo (~70x más rápido)
        cin_vals = [datos.get(c, 0.0) for c in COLS_CINEMATICAS]
        cin_scaled = np.tanh(((cin_vals - self.std_mean) / self.std_scale) / 3.0)
        for col, val in zip(COLS_CINEMATICAS, cin_scaled):
            datos[col] = val

        nz_val = datos.get('nlf', 1.0) - 1.0
        datos['nlf'] = np.tanh(((nz_val - self.nz_mean) / self.nz_scale) / 3.0)

        datos['airspeed-kt'] = datos.get('airspeed-kt', 0.0) * self.spd_scale + self.spd_min
        datos['altitude-ft'] = datos.get('altitude-ft', 0.0) * self.alt_scale + self.alt_min

        return np.array([datos.get(col, 0.0) for col in VECTOR_ORDENADO], dtype=np.float32)

    @pyqtSlot(dict)
    def procesar_telemetria(self, datos: dict):
        if not hasattr(self, 'modelo'): return

        if datos.get('es_salto', False):
            self.buffer.clear()
            ventana = datos.get('ventana_salto', [])
            for f in ventana:
                self.buffer.append(self._escalar(f.copy()))
        else:
            self.buffer.append(self._escalar(datos.copy()))

        if len(self.buffer) < VENTANA_BUFFER:
            return

        current_timestamp = datos.get('timestamp_ms', 0.0)

        # Inferencia
        tensor = torch.tensor(np.array(self.buffer)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_actual, logits_futuro, pred_mu, pred_log_var = self.modelo(tensor)

            alerta_roja, alerta_amarilla, prob_max, sigma, riesgo_salud = self.eicas.evaluar_instancia(
                logits_futuro, pred_mu, pred_log_var
            )

            probs_actual = torch.softmax(logits_actual, dim=1)[0]
            clase_actual = torch.argmax(probs_actual).item()

            # Emitir a la UI
            resultado = {
                'timestamp_ms': current_timestamp,
                'clase_actual': clase_actual,
                'alerta_roja_eicas': alerta_roja,
                'alerta_amarilla_eicas': alerta_amarilla,
                'prob_peligro_max': prob_max,
                'sigma': sigma,
                'riesgo_salud': riesgo_salud
            }
            event_bus.inference_updated.emit(resultado)
