import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# ==============================================================================
# --- 1. DEFINICIÓN DEL MODELO V8 ---
# ==============================================================================
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

        # ACOTACIÓN FÍSICA: La media (Índice de Salud) se acota entre 0 (Stall) y 1 (Seguro)
        pred_mu = torch.sigmoid(salida_riesgo[:, 0])
        pred_log_var = salida_riesgo[:, 1] # La varianza queda libre

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


# ==============================================================================
# --- 2. LA NUEVA FUNCIÓN DE PÉRDIDA DESACOPLADA ---
# ==============================================================================
class DecoupledHealthLoss(nn.Module):
    def __init__(self, weight_var=0.1):
        super(DecoupledHealthLoss, self).__init__()
        self.weight_var = weight_var

    def forward(self, pred_mu, pred_log_var, y_true):
        # 1. Pérdida Principal (MSE Clásico).
        loss_mu = F.mse_loss(pred_mu, y_true)

        # 2. Pérdida de Varianza (NLL) Desacoplada
        pred_log_var_seguro = torch.clamp(pred_log_var, min=-6.0, max=6.0)
        precision = torch.exp(-pred_log_var_seguro)

        # EL DETACH: Evita que la varianza afecte a la predicción principal
        error_cuadratico = (y_true - pred_mu.detach())**2

        loss_var = (precision * error_cuadratico) + pred_log_var_seguro
        loss_var = loss_var.mean()

        return loss_mu + (self.weight_var * loss_var)


# ==============================================================================
# --- 3. CONFIGURACIÓN DEL ENTRENAMIENTO ---
# ==============================================================================
# Asegúrate de que esta ruta exista en tu Google Drive
RUTA_CHECKPOINTS = '/content/drive/MyDrive/Proyecto Base de datos beta/checkpoints'
os.makedirs(RUTA_CHECKPOINTS, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AntonovTransformerModelV6().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Pesos para las clases (balanceo)
pesos_clases = torch.tensor([1.68, 3.94, 9.38, 21.23], dtype=torch.float32).to(device)
criterion_cls_actual = nn.CrossEntropyLoss(weight=pesos_clases)
criterion_cls_futuro = nn.CrossEntropyLoss(weight=pesos_clases)

criterion_reg_riesgo = DecoupledHealthLoss(weight_var=0.1)

LAMBDA_ACTUAL = 1.0
LAMBDA_FUTURO = 0.5
LAMBDA_RIESGO = 0.5

TAU_CRITICO = 5.0 # Ventana de advertencia en segundos

EPOCHS_MAXIMAS = 80
EPOCAS_WARMUP = 30
PACIENCIA_EARLY_STOPPING = 15

mejor_loss_validacion = float('inf')
epocas_sin_mejora = 0

scheduler_fase1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=EPOCAS_WARMUP)
scheduler_fase2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(EPOCHS_MAXIMAS - EPOCAS_WARMUP))
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[scheduler_fase1, scheduler_fase2], milestones=[EPOCAS_WARMUP]
)

# ==============================================================================
# --- 4. BUCLE PRINCIPAL (TWO-STAGE) ---
# ==============================================================================
print(f"\n Iniciando Entrenamiento V6 (Riesgo Continuo + Varianza) en {device}...")

for epoch in range(EPOCHS_MAXIMAS):
    model.train()
    loss_train_epoch = 0.0

    loop_train = tqdm(train_loader, desc=f"Entrenando Epoch {epoch+1}/{EPOCHS_MAXIMAS}")
    for x_batch, y_actual_batch, y_futuro_batch, y_tts_batch in loop_train:
        x_batch = x_batch.to(device)
        y_actual_batch = y_actual_batch.to(device)
        y_futuro_batch = y_futuro_batch.to(device)
        y_tts_batch = y_tts_batch.to(device)

        #  TRANSFORMACIÓN DEL TARGET EN TIEMPO REAL 
        y_riesgo_batch = 1.0 - torch.exp(-(30.0 * y_tts_batch) / TAU_CRITICO)

        optimizer.zero_grad()
        logits_actual, logits_futuro, pred_mu, pred_log_var = model(x_batch)

        # Fase 1: Varianza congelada
        if epoch < EPOCAS_WARMUP:
            pred_log_var_usada = torch.zeros_like(pred_log_var)
        else:
            pred_log_var_usada = pred_log_var

        loss_actual = criterion_cls_actual(logits_actual, y_actual_batch)
        loss_futuro = criterion_cls_futuro(logits_futuro, y_futuro_batch)
        loss_riesgo = criterion_reg_riesgo(pred_mu, pred_log_var_usada, y_riesgo_batch)

        loss_total = (LAMBDA_ACTUAL * loss_actual) + (LAMBDA_FUTURO * loss_futuro) + (LAMBDA_RIESGO * loss_riesgo)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_train_epoch += loss_total.item()
        loop_train.set_postfix(Loss=loss_total.item())

    promedio_loss_train = loss_train_epoch / len(train_loader)

    # --- VALIDACIÓN ---
    model.eval()
    loss_val_epoch = 0.0
    correctos_actual = 0
    error_absoluto_riesgo = 0.0
    total_muestras = 0

    with torch.no_grad():
        for x_val, y_actual_val, y_futuro_val, y_tts_val in val_loader:
            x_val = x_val.to(device)
            y_actual_val = y_actual_val.to(device)
            y_futuro_val = y_futuro_val.to(device)
            y_tts_val = y_tts_val.to(device)

            # Transformación en validación
            y_riesgo_val = 1.0 - torch.exp(-(30.0 * y_tts_val) / TAU_CRITICO)

            logits_actual, logits_futuro, pred_mu, pred_log_var = model(x_val)

            if epoch < EPOCAS_WARMUP:
                pred_log_var_usada = torch.zeros_like(pred_log_var)
            else:
                pred_log_var_usada = pred_log_var

            loss_actual = criterion_cls_actual(logits_actual, y_actual_val)
            loss_futuro = criterion_cls_futuro(logits_futuro, y_futuro_val)
            loss_riesgo = criterion_reg_riesgo(pred_mu, pred_log_var_usada, y_riesgo_val)

            loss_val = (LAMBDA_ACTUAL * loss_actual) + (LAMBDA_FUTURO * loss_futuro) + (LAMBDA_RIESGO * loss_riesgo)
            loss_val_epoch += loss_val.item()

            _, preds_actual = torch.max(logits_actual, 1)
            correctos_actual += torch.sum(preds_actual == y_actual_val).item()

            error_absoluto_riesgo += torch.sum(torch.abs(pred_mu - y_riesgo_val)).item()
            total_muestras += y_actual_val.size(0)

    promedio_loss_val = loss_val_epoch / len(val_loader)
    acc_actual = (correctos_actual / total_muestras) * 100
    mae_riesgo = error_absoluto_riesgo / total_muestras

    scheduler.step()
    lr_actual = optimizer.param_groups[0]['lr']

    fase_actual = "Fase 1 (Warm-up)" if epoch < EPOCAS_WARMUP else "Fase 2 (Incertidumbre Activa)"
    print(f" Epoch {epoch+1} [{fase_actual}] | Train Loss: {promedio_loss_train:.4f} | Val Loss: {promedio_loss_val:.4f} | LR: {lr_actual:.2e}")
    print(f" Val Acc Actual: {acc_actual:.2f}% | ⏱️ Val MAE Salud (0-1): {mae_riesgo:.4f}")

    if epoch >= EPOCAS_WARMUP:
        if promedio_loss_val < mejor_loss_validacion:
            mejor_loss_validacion = promedio_loss_val
            epocas_sin_mejora = 0
            ruta_guardado = os.path.join(RUTA_CHECKPOINTS, 'mejor_modelo_v8_riesgo.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_val': promedio_loss_val,
            }, ruta_guardado)
            print(f" ¡Récord! Modelo V6 guardado exitosamente.")
        else:
            epocas_sin_mejora += 1
            print(f"  Sin mejora: {epocas_sin_mejora}/{PACIENCIA_EARLY_STOPPING}")
            if epocas_sin_mejora >= PACIENCIA_EARLY_STOPPING:
                print(f"\n 🛑 EARLY STOPPING: El modelo se detuvo.")
                break
    else:
        print(f" ⏳ Récords bloqueados hasta la Época {EPOCAS_WARMUP+1}...")
