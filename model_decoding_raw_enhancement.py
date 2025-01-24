import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence

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

class TemporalAlignmentModule(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.alignment_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
    
    def forward(self, x):
        return self.alignment_net(x)

class EnhancedRNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True
        )
        self.output_projection = nn.Linear(hidden_size*2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, lengths=None):
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.gru(x)
        if isinstance(outputs, torch.nn.utils.rnn.PackedSequence):
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return self.layer_norm(self.output_projection(outputs))

class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=1024, decoder_embedding_size=1024, 
                 additional_encoder_nhead=8, additional_encoder_dim_feedforward=4096):
        super(BrainTranslator, self).__init__()
        
        # EEG Feature Extraction
        self.hidden_dim = 512
        self.feature_embedded = FeatureEmbedded(input_dim=104, hidden_dim=self.hidden_dim)
        
        # Projection and Temporal Alignment
        self.fc = ProjectionHead(embedding_dim=in_feature, projection_dim=in_feature)
        self.temporal_align = TemporalAlignmentModule(input_dim=in_feature)
        
        # Subject-specific processing
        self.conv1d_point = nn.Conv1d(1, 64, 1, stride=1)
        
        # Subject mapping
        SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 
                   'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS', 
                   'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']
        self.subjects_map_id = {subj: idx for idx, subj in enumerate(SUBJECTS)}
        
        # Subject matrices
        self.subject_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(64, 1)) for _ in range(len(SUBJECTS))
        ])
        
        # Enhanced RNN Decoder
        self.rnn_decoder = EnhancedRNNDecoder(
            input_size=in_feature,
            hidden_size=decoder_embedding_size
        )
        
        # Transformer components
        self.pos_embedding = nn.Parameter(torch.randn(1, 56, in_feature))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, 
            nhead=additional_encoder_nhead,
            dim_feedforward=additional_encoder_dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.layernorm_embedding = nn.LayerNorm(in_feature)
        self.brain_projection = ProjectionHead(embedding_dim=in_feature, projection_dim=1024, dropout=0.2)
        
        # BART
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

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, 
                target_ids_batch_converted, lenghts_words, word_contents_batch,
                word_contents_attn_batch, stepone, subject_batch, device, features=False):
        
        # Extract features
        feature_embedding = self.feature_embedded(input_embeddings_batch, lenghts_words, device)
        if len(feature_embedding.shape) == 2:
            feature_embedding = torch.unsqueeze(feature_embedding, 0)
            
        # Project features
        encoded_embedding = self.fc(feature_embedding)
        
        # Apply temporal alignment
        encoded_embedding = self.temporal_align(encoded_embedding)
        
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
        
        # Add positional embeddings and apply transformer encoder
        brain_embedding = encoded_embedding_subject + self.pos_embedding
        brain_embedding = self.encoder(brain_embedding, src_key_padding_mask=input_masks_invert)
        brain_embedding = self.layernorm_embedding(brain_embedding)
        brain_embedding = self.brain_projection(brain_embedding)
        
        if stepone:
            # Alignment phase
            words_embedding = self.bart.model.encoder.embed_tokens(word_contents_batch)
            return nn.MSELoss()(brain_embedding, words_embedding)
        else:
            # Decoding phase
            decoded_features = self.rnn_decoder(brain_embedding)
            out = self.bart(
                inputs_embeds=decoded_features,
                attention_mask=input_masks_batch,
                labels=target_ids_batch_converted,
                return_dict=True
            )
            
            if features:
                return out.logits, brain_embedding
            return out.logits

class FeatureEmbedded(nn.Module):
    def __init__(self, input_dim=105, hidden_dim=512, num_layers=2, is_bidirectional=True):
        super(FeatureEmbedded, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional
        
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=self.is_bidirectional
        )
        
        # Initialize parameters
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
                
    def forward(self, x, lengths, device):
        sentence_embedding_batch = []
        
        for x_sentence, lengths_sentence in zip(x, lengths):
            lstm_input = pack_padded_sequence(
                x_sentence, 
                lengths_sentence.cpu().numpy(), 
                batch_first=True, 
                enforce_sorted=False
            )
            
            lstm_outs, _ = self.gru(lstm_input)
            lstm_outs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outs)
            
            sentence_embedding = []
            for i in range(lengths_sentence.shape[0]):
                sentence_embedding.append(lstm_outs[int(lengths_sentence[i]-1), i, :])
            sentence_embedding = torch.stack(sentence_embedding, 0)
            sentence_embedding_batch.append(sentence_embedding)
            
        return torch.squeeze(torch.stack(sentence_embedding_batch, 0)).to(device)
