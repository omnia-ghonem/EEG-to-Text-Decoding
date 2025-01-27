import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=1024, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x

class EnhancedEEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead, dim_feedforward, dropout=0.1):
        super(EnhancedEEGEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 56, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        x = x + self.pos_encoder
        x = self.dropout(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
    
    def forward(self, eeg_features, text_features):
        attn_output, _ = self.attention(eeg_features, text_features, text_features)
        return attn_output

class BrainTranslator(nn.Module):
    def __init__(self, t5_model, in_feature=840, hidden_dim=512, num_layers=6, 
                 nhead=8, dim_feedforward=2048, decoder_embedding_size=None):
        # Fix the class name in super().__init__()
        super(BrainTranslator, self).__init__()
        
        self.eeg_encoder = EnhancedEEGEncoder(in_feature, hidden_dim, num_layers, nhead, dim_feedforward)
        self.attention_layer = AttentionLayer(hidden_dim)
        self.t5_model = t5_model
        
        # Add handling for decoder_embedding_size if needed
        self.decoder_embedding_size = decoder_embedding_size
        if decoder_embedding_size is not None:
            self.decoder_projection = nn.Linear(hidden_dim, decoder_embedding_size)
        
    def freeze_pretrained_t5(self):
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False

    def freeze_pretrained_brain(self):
        for name, param in self.named_parameters():
            if 't5_model' not in name:
                param.requires_grad = False

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, 
                target_ids_batch_converted, lenghts_words, word_contents_batch, 
                word_contents_attn_batch, stepone, subject_batch, device, features=False):
        eeg_features = self.eeg_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)
        
        if stepone:
            words_embedding = self.t5_model.encoder.embed_tokens(word_contents_batch)
            loss = nn.MSELoss()
            return loss(eeg_features, words_embedding)
        else:
            attn_output = self.attention_layer(eeg_features, word_contents_batch)
            
            # Project to decoder embedding size if specified
            if self.decoder_embedding_size is not None:
                attn_output = self.decoder_projection(attn_output)
                
            out = self.t5_model(inputs_embeds=attn_output, 
                              attention_mask=input_masks_batch, 
                              return_dict=True, 
                              labels=target_ids_batch_converted)
            
            if features:
                return out.logits, eeg_features
            return out.logits

class FeatureEmbedded(nn.Module):
    def __init__(self, input_dim=105, hidden_dim=512, num_layers=2, is_bidirectional=True):
        super(FeatureEmbedded, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional
        
        self.lstm = nn.GRU(input_size=self.input_dim, 
                           hidden_size=self.hidden_dim, 
                           num_layers=self.num_layers, 
                           batch_first=True, 
                           dropout=0.2,
                           bidirectional=self.is_bidirectional)
        
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
                
    def forward(self, x, lenghts, device):
        sentence_embedding_batch = []
        for x_sentence, lenghts_sentence in zip(x, lenghts):
            lstm_input = pack_padded_sequence(x_sentence, lenghts_sentence.cpu().numpy(), batch_first=True, enforce_sorted=False)
            lstm_outs, hidden = self.lstm(lstm_input)
            lstm_outs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outs)  

            if not self.is_bidirectional:
                sentence_embedding = lstm_outs[-1]
            else:
                sentence_embedding = torch.cat((lstm_outs[-1, :, :self.hidden_dim], lstm_outs[0, :, self.hidden_dim:]), dim=1)

            sentence_embedding_batch.append(sentence_embedding)

        return torch.squeeze(torch.stack(sentence_embedding_batch, 0)).to(device)
