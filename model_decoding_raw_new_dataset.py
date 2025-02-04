import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=1024, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)

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
        
        # Initialize parameters
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
                
    def forward(self, x, lengths, device):
        batch_size = x.size(0)
        
        # Handle input reshaping
        if x.dim() == 2:
            # If input is [batch_size, features*seq_len]
            if isinstance(lengths, list):
                seq_len = lengths[0]
            else:
                seq_len = lengths.item()
            x = x.view(batch_size, seq_len, -1)
        elif x.dim() == 3:
            # Input is already [batch_size, seq_len, features]
            seq_len = x.size(1)
            
        # Validate dimensions
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(-1)}")
        
        # Convert lengths to list if needed
        if torch.is_tensor(lengths):
            lengths = lengths.cpu().tolist()
        elif isinstance(lengths, int):
            lengths = [lengths]
        
        # Ensure lengths are proper size
        if len(lengths) == 1 and batch_size > 1:
            lengths = lengths * batch_size
            
        # Pack sequence
        packed_input = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process through GRU
        lstm_out, _ = self.lstm(packed_input)
        
        # Unpack sequence
        unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Get final states for each sequence in batch
        batch_final_states = []
        for i in range(batch_size):
            if self.is_bidirectional:
                # For bidirectional, concatenate both directions
                final_forward = unpacked_output[i, lengths[i]-1, :self.hidden_dim]
                final_backward = unpacked_output[i, 0, self.hidden_dim:]
                final_state = torch.cat([final_forward, final_backward], dim=0)
            else:
                final_state = unpacked_output[i, lengths[i]-1, :]
            batch_final_states.append(final_state)
            
        # Stack all final states
        output = torch.stack(batch_final_states, dim=0)
        return output.to(device)

class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=192, decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super(BrainTranslator, self).__init__()
        self.hidden_dim = 512
        self.in_feature = in_feature
        
        # Feature embedding layers
        self.feature_embedded = FeatureEmbedded(input_dim=in_feature, hidden_dim=self.hidden_dim)
        self.fc = ProjectionHead(embedding_dim=in_feature, projection_dim=in_feature, dropout=0.1)

        # Convolutional layer
        self.conv1d_point = nn.Conv1d(1, 64, 1, stride=1)
        
        # Transformer encoder layers
        self.pos_embedding = nn.Parameter(torch.randn(1, 201, in_feature))
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
        
        # Brain projection layer
        self.brain_projection = ProjectionHead(embedding_dim=in_feature, projection_dim=1024, dropout=0.2)
        
        # BART model
        self.bart = bart
        
    def freeze_pretrained_bart(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
            if ('bart' in name):
                param.requires_grad = False

    def freeze_pretrained_brain(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if ('bart' in name):
                param.requires_grad = True

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch, 
               lengths_batch, word_contents_batch, word_contents_attn_batch, stepone, subject_batch, device):
        # Process input embeddings
        batch_size = input_embeddings_batch.size(0)
        
        # Calculate proper sequence length and feature dimension
        if input_embeddings_batch.dim() == 2:
            # For 2D input, calculate proper dimensions
            total_size = input_embeddings_batch.size(1)
            feature_dim = self.in_feature
            seq_len = total_size // feature_dim
            input_embeddings_batch = input_embeddings_batch.view(batch_size, seq_len, feature_dim)
        elif input_embeddings_batch.dim() == 3:
            # If input is already [batch_size, seq_len, features], verify dimensions
            seq_len = input_embeddings_batch.size(1)
            feature_dim = input_embeddings_batch.size(2)
            if feature_dim != self.in_feature:
                raise ValueError(f"Expected feature dimension {self.in_feature}, got {feature_dim}")
        else:
            raise ValueError(f"Unexpected input dimension: {input_embeddings_batch.dim()}")
            
        # Get feature embeddings
        feature_embedding = self.feature_embedded(input_embeddings_batch, lengths_batch, device)
        if len(feature_embedding.shape) == 2:
            feature_embedding = feature_embedding.unsqueeze(0)
        
        # Project features
        encoded_embedding = self.fc(feature_embedding)
        
        # Apply convolutional layer
        tmp = encoded_embedding.unsqueeze(1)
        tmp = self.conv1d_point(tmp)
        tmp = tmp.transpose(1, 2)
        
        # Add positional embeddings
        brain_embedding = tmp + self.pos_embedding[:, :tmp.size(1), :]
        
        # Apply transformer encoder
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
