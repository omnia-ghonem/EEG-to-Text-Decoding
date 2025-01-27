import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, V)
        
        # Combine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_linear(out)

class EnhancedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention
        attention_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

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

class FeatureEmbedded(nn.Module):
    def __init__(self, input_dim=105, hidden_dim=512, num_layers=2, is_bidirectional=True):
        super().__init__()
        
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
        
        # Add attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2 if is_bidirectional else hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
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
        sentence_embedding_batch = []
        
        for x_sentence, lengths_sentence in zip(x, lengths):
            # Pack sequence for LSTM
            lstm_input = pack_padded_sequence(
                x_sentence, 
                lengths_sentence.cpu().numpy(),
                batch_first=True,
                enforce_sorted=False
            )
            
            # Process through LSTM
            lstm_outs, _ = self.lstm(lstm_input)
            lstm_outs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outs, batch_first=True)
            
            # Apply attention
            attn_output, _ = self.attention(lstm_outs, lstm_outs, lstm_outs)
            
            # Gather final representations
            sentence_embedding = []
            for i in range(lengths_sentence.shape[0]):
                sentence_embedding.append(attn_output[i, int(lengths_sentence[i]-1), :])
            sentence_embedding = torch.stack(sentence_embedding, 0)
            
            sentence_embedding_batch.append(sentence_embedding)
            
        return torch.squeeze(torch.stack(sentence_embedding_batch, 0)).to(device)

class BrainTranslator(nn.Module):
    def __init__(self, t5, in_feature=840, decoder_embedding_size=1024, additional_encoder_nhead=8, 
                 additional_encoder_dim_feedforward=2048):
        super(BrainTranslator, self).__init__()
        
        # Embedded EEG raw features
        self.hidden_dim = 512
        self.feature_embedded = FeatureEmbedded(input_dim=104, hidden_dim=self.hidden_dim)
        self.fc = ProjectionHead(embedding_dim=in_feature, projection_dim=in_feature, dropout=0.1)
        
        # Enhanced encoder with attention
        self.enhanced_encoder = EnhancedEncoder(
            input_dim=in_feature,
            hidden_dim=in_feature,
            num_heads=additional_encoder_nhead
        )

        # conv1d
        self.conv1d_point = nn.Conv1d(1, 64, 1, stride=1)

        # Subject handling
        SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 
                   'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS', 
                   'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']
        self.subjects_map_id = {subj: i for i, subj in enumerate(SUBJECTS)}
        
        # Learnable subject matrices
        self.subject_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(64, 1)) for _ in range(len(SUBJECTS))
        ])
        
        # Brain transformer encoder components
        self.pos_embedding = nn.Parameter(torch.randn(1, 56, in_feature))
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
        
        # Cross-attention between brain features and language model
        self.cross_attention = MultiHeadAttention(
            d_model=in_feature,
            num_heads=additional_encoder_nhead,
            dropout=0.1
        )

        self.brain_projection = ProjectionHead(embedding_dim=in_feature, projection_dim=1024, dropout=0.2)
        
        # T5 model
        self.t5 = t5
        
    def freeze_pretrained_t5(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
            if ('t5' in name):
                param.requires_grad = False

    def freeze_pretrained_brain(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if ('t5' in name):
                param.requires_grad = True

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, 
                target_ids_batch_converted, lenghts_words, word_contents_batch, 
                word_contents_attn_batch, stepone, subject_batch, device, features=False):
        
        # Feature embedding with attention
        feature_embedding = self.feature_embedded(input_embeddings_batch, lenghts_words, device)
        if len(feature_embedding.shape) == 2:
            feature_embedding = torch.unsqueeze(feature_embedding, 0)
            
        # Project features
        encoded_embedding = self.fc(feature_embedding)
        
        # Subject-specific processing
        encoded_embedding_subject = []
        for i in range(encoded_embedding.shape[0]):
            tmp = torch.unsqueeze(encoded_embedding[i,:,:], 1)
            tmp = self.conv1d_point(tmp)
            tmp = torch.swapaxes(tmp, 1, 2)
            mat_subject = self.subject_matrices[self.subjects_map_id[subject_batch[i]]].to(device)
            tmp = torch.matmul(tmp, mat_subject)
            tmp = torch.squeeze(tmp)
            encoded_embedding_subject.append(tmp)
            
        if len(encoded_embedding_subject) == 1:
            encoded_embedding_subject = torch.unsqueeze(encoded_embedding_subject[0], 0)
        else:
            encoded_embedding_subject = torch.stack(encoded_embedding_subject, 0).to(device)
        
        # Add positional embeddings and apply enhanced encoder
        brain_embedding = encoded_embedding_subject + self.pos_embedding
        brain_embedding = self.enhanced_encoder(brain_embedding, mask=~input_masks_invert)
        brain_embedding = self.layernorm_embedding(brain_embedding)
        
        # Apply transformer encoder
        brain_embedding = self.encoder(brain_embedding, src_key_padding_mask=input_masks_invert)
        
        # Project to final embedding space
        brain_embedding = self.brain_projection(brain_embedding)

        if stepone:
            # Training step one: align with word embeddings
            words_embedding = self.t5.shared(word_contents_batch)
            loss = nn.MSELoss()
            return loss(brain_embedding, words_embedding)
        else:
            # Training step two: generate text
            out = self.t5(
                inputs_embeds=brain_embedding,
                attention_mask=input_masks_batch,
                labels=target_ids_batch_converted,
                return_dict=True
            )
            
            if features:
                return out.logits, brain_embedding
            return out.logits
