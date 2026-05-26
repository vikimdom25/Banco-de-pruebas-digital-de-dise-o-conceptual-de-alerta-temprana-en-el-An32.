import torch
import numpy as np
import torch.nn as nn
import joblib
import warnings
import os
from collections import deque

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# --- 1. PEGAMOS TU ARQUITECTURA EXACTA AQUÍ ---
class AntonovTransformerModelV3(nn.Module):
    def __init__(self, num_features=19, sequence_length=500,
                 d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256,
                 num_clases_actual=4, num_clases_futuro=4):
        super(AntonovTransformerModelV3, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.input_projection = nn.Linear(num_features, d_model)
        self.positional_encoding = self._create_positional_encoding(sequence_length, d_model)
        
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
        self.head_tobit = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x)
        x = x + self.positional_encoding.to(x.device)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_processed = self.transformer_encoder(x)
        global_representation = x_processed[:, 0, :]

        logits_actual = self.head_cls_actual(global_representation)
        logits_futuro = self.head_cls_futuro(global_representation)
        salida_tobit = self.head_tobit(global_representation)
        return logits_actual, logits_futuro, salida_tobit[:, 0], salida_tobit[:, 1]

    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    

class MotorInferenciaStall:
    def __init__(self, ruta_modelo, directorio_scalers):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Iniciando Motor ML en: {self.device}")
        
        # 1. Instanciamos la arquitectura vacía (asegúrate de tener la clase AntonovTransformerModelV3 definida arriba)
        self.modelo = AntonovTransformerModelV3().to(self.device)
        
        # 2. Cargamos el diccionario (.pth) de forma segura
        checkpoint = torch.load(ruta_modelo, map_location=self.device)
        
        # 3. Inyectamos los pesos en la arquitectura
        self.modelo.load_state_dict(checkpoint['model_state_dict'])
        self.modelo.eval() # Modo evaluación
        
        # --- Carga de Scalers ---
        self.std_scaler = joblib.load(os.path.join(directorio_scalers, 'standard_scaler_cinematica.pkl'))
        self.std_nz_scaler = joblib.load(os.path.join(directorio_scalers, 'standard_scaler_nz.pkl'))
        self.minmax_spd = joblib.load(os.path.join(directorio_scalers, 'minmax_speed_scaler.pkl'))
        self.minmax_alt = joblib.load(os.path.join(directorio_scalers, 'minmax_alt_scaler.pkl'))
        
        self.window_size = 500
        self.buffer = deque(maxlen=self.window_size)
        
        self.vector_ordenado = [
            'throttle', 'flap-pos-norm', 'elevator-pos-norm', 'left-aileron-pos-norm', 
            'rudder-pos-norm', 'altitude-ft', 'pitch-deg', 'roll-deg', 'alpha-deg', 
            'side-slip-deg', 'airspeed-kt', 'vertical-speed-fps', 'q_rad_sec', 
            'p_rad_sec', 'r_rad_sec', 'nlf', 'airspeed-kt_dot', 'nlf_dot', 'alpha-deg_dot'
        ]
        self.cols_cinematicas = [
            'nlf_dot', 'alpha-deg_dot', 'airspeed-kt_dot', 
            'p_rad_sec', 'q_rad_sec', 'r_rad_sec', 
            'side-slip-deg', 'vertical-speed-fps', 'pitch-deg', 'roll-deg'
        ]

    def _escalar(self, dict_datos):
        datos = dict_datos.copy()
        
        # Alpha
        flap = datos.get('flap-pos-norm', 0)
        datos['alpha-deg'] = np.tanh(datos['alpha-deg'] / (16.0 - 3.0 * flap))
        
        # Cinemáticas
        cin_raw = np.array([[datos[c] for c in self.cols_cinematicas]])
        cin_scaled = np.tanh(self.std_scaler.transform(cin_raw) / 3.0)[0]
        for i, col in enumerate(self.cols_cinematicas): datos[col] = cin_scaled[i]
            
        # NLF
        nz_raw = np.array([[datos['nlf'] - 1.0]])
        datos['nlf'] = np.tanh(self.std_nz_scaler.transform(nz_raw) / 3.0)[0][0]
        
        # Speed & Alt
        datos['airspeed-kt'] = self.minmax_spd.transform([[datos['airspeed-kt']]])[0][0]
        datos['altitude-ft'] = self.minmax_alt.transform([[datos['altitude-ft']]])[0][0]
        
        return np.array([datos[col] for col in self.vector_ordenado], dtype=np.float32)

    def procesar_secuencia(self, lista_datos, es_salto=False):
        """
        Procesa una lista de diccionarios. 
        Si 'es_salto' es True, limpia la memoria y la reconstruye desde cero.
        """
        if es_salto:
            self.buffer.clear()
            
        for dict_datos in lista_datos:
            vector = self._escalar(dict_datos)
            self.buffer.append(vector)
        
        # Si aún no tenemos los 500 pasos (10 segundos a 50Hz), no predecimos
        if len(self.buffer) < self.window_size:
            return None, None, None, None, None, None
            
        tensor = torch.tensor(np.array(self.buffer)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_actual, logits_futuro, pred_tts, pred_log_var = self.modelo(tensor)
            
            probs_actual = torch.softmax(logits_actual, dim=1)[0]
            probs_futuro = torch.softmax(logits_futuro, dim=1)[0]
            
            clase_actual = torch.argmax(probs_actual).item()
            clase_futuro = torch.argmax(probs_futuro).item()
            
            t2s = pred_tts.item() * 30.0  
            
            # --- INCERTIDUMBRE (TOBIT) ---
            # La red saca el logaritmo de la varianza. La Desviación Estándar (sigma) es exp(0.5 * log_var)
            # Y la multiplicamos por 30.0 para revertir tu escalado original.
            sigma = torch.exp(0.5 * pred_log_var).item() * 30.0
            
        return clase_actual, probs_actual.cpu().numpy(), clase_futuro, probs_futuro.cpu().numpy(), t2s, sigma