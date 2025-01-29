import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=512, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, lengths):
        # Pack sequence
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process through GRU
        outputs, _ = self.gru(packed)
        
        # Unpack sequence
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        return outputs

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.norm(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class BrainTranslator(nn.Module):
    def __init__(self, bart_model, input_dim=192, hidden_dim=512, embedding_dim=1024):
        super().__init__()
        
        # Neural encoder components
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.transformer_encoder = TransformerEncoder(
            d_model=embedding_dim,
            nhead=8,
            num_layers=6
        )
        
        # BART components
        self.bart = bart_model
        
    def freeze_bart(self):
        """Freeze BART parameters"""
        for param in self.bart.parameters():
            param.requires_grad = False
            
    def freeze_neural_encoder(self):
        """Freeze neural encoder parameters"""
        for param in self.temporal_encoder.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

    def forward(self, neural_data, neural_mask, input_ids=None, attention_mask=None):
        # Get sequence lengths from mask
        lengths = neural_mask.sum(dim=1).cpu()
        
        # Process through temporal encoder
        temporal_features = self.temporal_encoder(neural_data, lengths)
        
        # Project to BART dimension
        projected = self.projection(temporal_features)
        
        # Process through transformer encoder
        encoded = self.transformer_encoder(
            projected,
            mask=(1 - neural_mask).bool()
        )
        
        # Generate text with BART
        if self.training:
            outputs = self.bart(
                inputs_embeds=encoded,
                attention_mask=neural_mask,
                labels=input_ids,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            outputs = self.bart.generate(
                inputs_embeds=encoded,
                attention_mask=neural_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            return outputs
