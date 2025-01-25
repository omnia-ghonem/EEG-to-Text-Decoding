import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence



class EnhancedTemporalAlignment(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.alignment_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
    def forward(self, x):
        aligned = self.alignment_net(x)
        return aligned + x  # Residual connection

class EnhancedLSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(
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
        outputs, (hidden, cell) = self.lstm(x)
        if isinstance(outputs, torch.nn.utils.rnn.PackedSequence):
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return self.layer_norm(self.output_projection(outputs))

class FeatureEmbedded(nn.Module):
    def __init__(self, input_dim=105, hidden_dim=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
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
        sentence_embeddings = []
        
        for x_sentence, lengths_sentence in zip(x, lengths):
            lstm_input = pack_padded_sequence(
                x_sentence, 
                lengths_sentence.cpu().numpy(), 
                batch_first=True, 
                enforce_sorted=False
            )
            
            outputs, (hidden, cell) = self.lstm(lstm_input)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            
            sentence_embedding = []
            for i in range(lengths_sentence.shape[0]):
                sentence_embedding.append(outputs[int(lengths_sentence[i]-1), i, :])
            sentence_embedding = torch.stack(sentence_embedding, 0)
            sentence_embeddings.append(sentence_embedding)
            
        return torch.squeeze(torch.stack(sentence_embeddings, 0)).to(device)

class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=1024, decoder_embedding_size=1024,
                 additional_encoder_nhead=8, additional_encoder_dim_feedforward=4096):
        super().__init__()
        
        # EEG Feature Extraction with LSTM
        self.hidden_dim = 512
        self.feature_embedded = FeatureEmbedded(input_dim=104, hidden_dim=self.hidden_dim)
        
        # Enhanced temporal alignment
        self.temporal_align = EnhancedTemporalAlignment(input_dim=in_feature)
        
        # Subject processing
        self.conv1d_point = nn.Conv1d(1, 64, 1, stride=1)
        
        # Subject mapping
        SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH',
                   'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS',
                   'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']
        self.subjects_map = {subj: idx for idx, subj in enumerate(SUBJECTS)}
        
        # Learnable subject matrices
        self.subject_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(64, 1)) for _ in range(len(SUBJECTS))
        ])
        
        # Enhanced LSTM decoder
        self.lstm_decoder = EnhancedLSTMDecoder(
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
        
        # BART
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
        
    def forward(self, input_embeddings, input_masks, input_masks_invert,
                target_ids, lengths_words, word_contents,
                word_contents_attn, stepone, subjects, device, features=False):
                
        # Extract features
        features = self.feature_embedded(input_embeddings, lengths_words, device)
        if len(features.shape) == 2:
            features = features.unsqueeze(0)
            
        # Apply temporal alignment
        aligned_features = self.temporal_align(features)
        
        # Subject-specific processing
        subject_features = []
        for i, subject in enumerate(subjects):
            tmp = aligned_features[i].unsqueeze(1)
            tmp = self.conv1d_point(tmp)
            tmp = torch.swapaxes(tmp, 1, 2)
            subject_matrix = self.subject_matrices[self.subjects_map[subject]].to(device)
            tmp = torch.matmul(tmp, subject_matrix).squeeze()
            subject_features.append(tmp)
            
        subject_features = torch.stack(subject_features, 0).to(device) if len(subject_features) > 1 else subject_features[0].unsqueeze(0)
        
        # Add position embeddings and apply transformer
        brain_embedding = subject_features + self.pos_embedding
        brain_embedding = self.encoder(brain_embedding, src_key_padding_mask=input_masks_invert)
        brain_embedding = self.layernorm_embedding(brain_embedding)
        
        if stepone:
            # Alignment phase
            word_embeddings = self.bart.model.encoder.embed_tokens(word_contents)
            return F.mse_loss(brain_embedding, word_embeddings)
        else:
            # Decoding phase  
            decoded_features = self.lstm_decoder(brain_embedding)
            out = self.bart(
                inputs_embeds=decoded_features,
                attention_mask=input_masks,
                labels=target_ids,
                return_dict=True
            )
            
            return (out.logits, brain_embedding) if features else out.logits
