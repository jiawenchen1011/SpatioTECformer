import torch
import torch.nn as nn
from temporal_encoding import LearnableTemporalEncoding, TemporalEncoding
from enhanced_cnn import EnhancedCNN2D
from adaptive_fusion import AdaptiveFusion

# Prediction model
class CNN_Encoder(nn.Module):
    def __init__(self, configs, cnn_channels, cnn_kernel_size, num_latitudes, num_longitudes, input_feature_dim, use_learnable_temporal=True):
        super(CNN_Encoder, self).__init__()
        self.num_latitudes = num_latitudes
        self.num_longitudes = num_longitudes
        self.pred_len = configs.pred_len
        self.input_feature_dim = input_feature_dim

        # CNN feature extraction: Reshapes (batch_size, seq_len, channels, lat, lon) to (batch_size * seq_len, channels, lat, lon) to extract spatial features for each time step
        self.cnn_stack = nn.Sequential(
            EnhancedCNN2D(in_channels=5, out_channels=cnn_channels, kernel_sizes=[3, 5, 7]),
            EnhancedCNN2D(in_channels=cnn_channels, out_channels=cnn_channels, kernel_sizes=[3, 5, 7]),
            EnhancedCNN2D(in_channels=cnn_channels, out_channels=cnn_channels, kernel_sizes=[3, 5, 7])
        )
        # Projects flattened CNN features to the d_model dimension
        self.proj = nn.Linear(cnn_channels * num_latitudes * num_longitudes, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # Maps external features (temporal features + auxiliary features)
        self.phys_proj = nn.Linear(input_feature_dim, configs.d_model)

        # Adaptive feature fusion
        self.adaptive_fusion = AdaptiveFusion(configs.d_model)

        # Temporal encoding: Supports learnable or fixed positional encodings
        if use_learnable_temporal:
            self.temporal_encoding = LearnableTemporalEncoding(configs.d_model, max_len=configs.seq_len)
        else:
            self.temporal_encoding = TemporalEncoding(configs.d_model, max_len=configs.seq_len)

        # Encoder component: Employs an 8-layer Transformer encoder to capture spatial and temporal dependencies in the time series
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Output projection layer: Maps to predicted TEC values for each grid point
        self.output_projection = nn.Linear(configs.d_model, num_latitudes * num_longitudes)

    def forward(self, x_dec, x_mark_dec):
        # x_dec shape: (batch_size, seq_len, 5, lat, lon) - historical data only
        # x_mark_dec shape: (batch_size, seq_len, input_feature_dim) - temporal and auxiliary features
        batch_size, seq_len, channels, lat, lon = x_dec.shape

        # CNN feature extraction
        x = x_dec.view(batch_size * seq_len, channels, lat, lon)
        x = self.cnn_stack(x)
        x = x.view(batch_size, seq_len, -1)
        x_tec = self.proj(x)  # (batch_size, seq_len, d_model)
        x_tec = self.dropout(x_tec)

        # External feature processing
        x_phys = self.phys_proj(x_mark_dec)  # (batch_size, seq_len, d_model)

        # Adaptive feature fusion
        x = self.adaptive_fusion(x_tec, x_phys)  # (batch_size, seq_len, d_model)

        # Add temporal encoding
        x = self.temporal_encoding(x)  # (batch_size, seq_len, d_model)

        # Transformer processing (no causal mask required)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        encoded = self.encoder(x)
        encoded = encoded.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # Use the encoding of the last time step as the input for prediction, assuming it encapsulates sufficient historical information
        pred_repr = encoded[:, -1:, :]  # (batch_size, 1, d_model)
        pred_repr = self.layer_norm(pred_repr)
        out = self.output_projection(pred_repr)
        out = out.view(batch_size, self.pred_len, self.num_latitudes, self.num_longitudes)
        return out