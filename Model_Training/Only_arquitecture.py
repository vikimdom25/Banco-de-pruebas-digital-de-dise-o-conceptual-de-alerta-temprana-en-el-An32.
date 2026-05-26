import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import os
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# --- Arquitectura separada ---
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# original AntonovTransformerModelV6
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

        self.head_riesgo = nn.Sequential(
            nn.Linear(d_model * 3, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 2)
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

        cls_representation = x_processed[:, 0, :]
        gap_representation = x_processed[:, 1:, :].mean(dim=1)
        last_token_representation = x_processed[:, -1, :]

        logits_actual = self.head_cls_actual(cls_representation)
        logits_futuro = self.head_cls_futuro(cls_representation)

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
