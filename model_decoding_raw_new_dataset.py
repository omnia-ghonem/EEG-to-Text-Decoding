import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Feature Embedding Model
class FeatureEmbedded(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=512, num_layers=2, is_bidirectional=True):
        super(FeatureEmbedded, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional
        
        self.lstm = nn.GRU(
            input_size=self.input_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True, 
            dropout=0.2,
            bidirectional=self.is_bidirectional
        )
                
    def forward(self, x, lengths, device):
        if len(lengths.shape) == 0:
            lengths = lengths.unsqueeze(0)

        # Ensure correct feature dimension
        x = x[..., :self.input_dim]  # Ensure input matches GRU expectation
        
        # Pack sequence
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Get last hidden state
        if self.is_bidirectional:
            sentence_embedding = lstm_out[:, -1, :]
        else:
            idx = (lengths - 1).view(-1, 1).expand(lengths.size(0), self.hidden_dim)
            sentence_embedding = lstm_out.gather(0, idx.unsqueeze(0)).squeeze()
        
        return sentence_embedding

# Brain Translator Model
class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=192, decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super(BrainTranslator, self).__init__()

        self.hidden_dim = 512
        self.feature_embedded = FeatureEmbedded(input_dim=in_feature, hidden_dim=self.hidden_dim)
        self.fc = nn.Linear(in_feature, in_feature)

        # **Fix Conv1d: Ensure correct input dimension**
        self.conv1d_point = nn.Conv1d(in_feature, in_feature, kernel_size=1, stride=1)  # FIXED

        # Transformer Encoder
        self.pos_embedding = nn.Parameter(torch.randn(1, 201, in_feature))  # Ensure it matches in_feature=192
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature,  
            nhead=additional_encoder_nhead,  
            dim_feedforward=additional_encoder_dim_feedforward, 
            dropout=0.1, 
            activation="gelu", 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=12)
        self.layernorm_embedding = nn.LayerNorm(in_feature, eps=1e-05)

        self.brain_projection = nn.Linear(in_feature, 1024)  # FIXED: Projection to correct dim
        
        self.bart = bart
        
    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch, lengths_batch, word_contents_batch, word_contents_attn_batch, stepone, subject_batch, device):
        if len(lengths_batch.shape) == 0:
            lengths_batch = lengths_batch.unsqueeze(0)
            
        if len(input_embeddings_batch.shape) == 2:
            input_embeddings_batch = input_embeddings_batch.unsqueeze(0)
            
        # Ensure feature dimension is **192**
        input_embeddings_batch = input_embeddings_batch[..., :192]  # Fix input shape

        # Feature embedding
        feature_embedding = self.feature_embedded(input_embeddings_batch, lengths_batch, device)
        feature_embedding = self.fc(feature_embedding)

        # **Fix Conv1d Processing**
        feature_embedding = feature_embedding.unsqueeze(-1)  # Shape: [batch, 192, 1]
        feature_embedding = self.conv1d_point(feature_embedding)  # Shape remains [batch, 192, 1]
        feature_embedding = feature_embedding.squeeze(-1)  # Remove last dimension if necessary

        # Fix Positional Embedding
        brain_embedding = feature_embedding.unsqueeze(1) + self.pos_embedding[:, :feature_embedding.size(1), :192]

        # Transformer Encoding
        brain_embedding = self.encoder(brain_embedding, src_key_padding_mask=input_masks_invert)
        brain_embedding = self.layernorm_embedding(brain_embedding)
        brain_embedding = self.brain_projection(brain_embedding)

        if stepone:
            if word_contents_batch is not None:
                words_embedding = self.bart.model.encoder.embed_tokens(word_contents_batch)
                return nn.MSELoss()(brain_embedding, words_embedding)
            return brain_embedding
        else:
            out = self.bart(
                inputs_embeds=brain_embedding, 
                attention_mask=input_masks_batch, 
                labels=target_ids_batch,
                return_dict=True
            )
            return out.logits
