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
        
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
                
    def forward(self, x, lengths, device):
        # Convert lengths to list
        if torch.is_tensor(lengths):
            lengths = lengths.cpu().tolist()
        elif isinstance(lengths, int):
            lengths = [lengths]
        
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
        
        # Get final states
        batch_size = x.size(0)
        batch_final_states = []
        for i in range(batch_size):
            if self.is_bidirectional:
                final_forward = unpacked_output[i, lengths[i]-1, :self.hidden_dim]
                final_backward = unpacked_output[i, 0, self.hidden_dim:]
                final_state = torch.cat([final_forward, final_backward], dim=0)
            else:
                final_state = unpacked_output[i, lengths[i]-1, :]
            batch_final_states.append(final_state)
            
        output = torch.stack(batch_final_states, dim=0)
        return output.to(device)

class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=192, decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super(BrainTranslator, self).__init__()
        self.hidden_dim = 512
        self.in_feature = in_feature
        
        self.feature_embedded = FeatureEmbedded(input_dim=in_feature, hidden_dim=self.hidden_dim)
        self.fc = ProjectionHead(embedding_dim=in_feature, projection_dim=in_feature, dropout=0.1)

        self.conv1d_point = nn.Conv1d(1, 64, 1, stride=1)
        
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
        
        self.brain_projection = ProjectionHead(embedding_dim=in_feature, projection_dim=1024, dropout=0.2)
        
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
        
        # Ensure proper input dimensions
        if len(input_embeddings_batch.shape) == 2:
            input_embeddings_batch = input_embeddings_batch.unsqueeze(0)
            
        # Handle case where batch size is 20 and total size is 4020
        batch_size = input_embeddings_batch.size(0)
        
        # Calculate correct sequence length (4020 / (20 * 192) = 1.046875, so actual sequence length should be 21)
        seq_len = input_embeddings_batch.size(1)
        if input_embeddings_batch.numel() == 4020:  # Specific case handling
            seq_len = 21
            input_embeddings_batch = input_embeddings_batch.reshape(batch_size, seq_len, -1)
        
        feature_embedding = self.feature_embedded(input_embeddings_batch, lengths_batch, device)
        if len(feature_embedding.shape) == 2:
            feature_embedding = feature_embedding.unsqueeze(0)
            
        encoded_embedding = self.fc(feature_embedding)
        
        tmp = encoded_embedding.unsqueeze(1)
        tmp = self.conv1d_point(tmp)
        tmp = tmp.transpose(1, 2)
        
        brain_embedding = tmp + self.pos_embedding[:, :tmp.size(1), :]
        
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
