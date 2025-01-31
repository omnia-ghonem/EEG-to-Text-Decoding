import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=1024,
        dropout=0.1
    ):
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
        batch_embeddings = []
        
        # Ensure lengths are in the right format
        if len(lengths.shape) == 0:
            lengths = [lengths.item()]
        elif torch.is_tensor(lengths):
            lengths = lengths.cpu().tolist()
            
        for i, (x_sentence, length) in enumerate(zip(x, lengths)):
            # Reshape input if needed
            if len(x_sentence.shape) == 1:
                x_sentence = x_sentence.view(1, -1, self.input_dim)
            elif len(x_sentence.shape) == 2:
                if x_sentence.shape[1] != self.input_dim:
                    x_sentence = x_sentence.transpose(0, 1)
                x_sentence = x_sentence.unsqueeze(0)
            elif len(x_sentence.shape) == 3:
                if x_sentence.shape[2] != self.input_dim:
                    x_sentence = x_sentence.transpose(1, 2)
            
            # Ensure the correct sequence length is used
            x_sentence = x_sentence[:, :length, :]
            
            # Pack sequence
            lengths_tensor = torch.tensor([length], dtype=torch.int64)
            packed_input = pack_padded_sequence(
                x_sentence, 
                lengths_tensor,
                batch_first=True, 
                enforce_sorted=False
            )
            
            # Process through GRU
            lstm_out, _ = self.lstm(packed_input)
            
            # Unpack sequence
            unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            
            # Get final hidden state
            if self.is_bidirectional:
                # For bidirectional, concatenate both directions
                last_forward = unpacked_output[0, -1, :self.hidden_dim]
                last_backward = unpacked_output[0, 0, self.hidden_dim:]
                sentence_embedding = torch.cat([last_forward, last_backward], dim=0)
            else:
                sentence_embedding = unpacked_output[0, -1, :]
            
            batch_embeddings.append(sentence_embedding)
        
        return torch.stack(batch_embeddings, 0).to(device)

class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=192, decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super(BrainTranslator, self).__init__()
        
        self.hidden_dim = 512
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
            if 'bart' in name:
                param.requires_grad = False

    def freeze_pretrained_brain(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if 'bart' in name:
                param.requires_grad = True

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch, 
               lengths_batch, word_contents_batch, word_contents_attn_batch, stepone, subject_batch, device):
        
        # Ensure input embeddings have correct shape [batch_size, seq_len, features]
        if len(input_embeddings_batch.shape) == 2:
            input_embeddings_batch = input_embeddings_batch.unsqueeze(0)
            
        # Reshape if needed to match expected dimensions
        if input_embeddings_batch.shape[-1] != 192:
            input_embeddings_batch = input_embeddings_batch.reshape(input_embeddings_batch.shape[0], -1, 192)
            
        feature_embedding = self.feature_embedded(input_embeddings_batch, lengths_batch, device)
        
        if len(feature_embedding.shape) == 2:
            feature_embedding = feature_embedding.unsqueeze(0)
            
        encoded_embedding = self.fc(feature_embedding)
        
        # Process with conv1d
        tmp = encoded_embedding.unsqueeze(1)
        tmp = self.conv1d_point(tmp)
        tmp = tmp.transpose(1, 2)
        
        # Add positional embeddings
        brain_embedding = tmp + self.pos_embedding[:, :tmp.size(1), :]
        
        # Transformer processing
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
