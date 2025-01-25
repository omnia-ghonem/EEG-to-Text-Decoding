import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class Generator(nn.Module):
    def __init__(self, input_dim=104, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
    def forward(self, z, condition=None):
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
        output, _ = self.lstm(z)
        return self.decoder(output)

class Discriminator(nn.Module):
    def __init__(self, input_dim=104, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, condition=None):
        if condition is not None:
            x = torch.cat([x, condition], dim=-1)
        features, _ = self.lstm(x)
        return self.classifier(features)

class HybridEncoder(nn.Module):
    def __init__(self, input_dim=104, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths = lengths.cpu()
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x)
        if isinstance(lstm_out, torch.nn.utils.rnn.PackedSequence):
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return self.norm(lstm_out)

class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=1024, decoder_embedding_size=1024,
                 additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super().__init__()
        
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.hybrid_encoder = HybridEncoder()
        
        self.hidden_dim = 256
        self.feature_embedded = FeatureEmbedded(input_dim=104, hidden_dim=self.hidden_dim)
        self.temporal_align = EnhancedTemporalAlignment(input_dim=self.hidden_dim)
        self.conv1d_point = nn.Conv1d(1, 32, 1, stride=1)
        
        SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH',
                   'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS',
                   'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']
        self.subjects_map = {subj: idx for idx, subj in enumerate(SUBJECTS)}
        self.subject_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(32, 1)) for _ in range(len(SUBJECTS))
        ])
        
        self.lstm_decoder = EnhancedLSTMDecoder(input_size=in_feature, hidden_size=decoder_embedding_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 56, in_feature))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature,
            nhead=additional_encoder_nhead,
            dim_feedforward=additional_encoder_dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.layernorm_embedding = nn.LayerNorm(in_feature)
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

    def generate_synthetic_data(self, batch_size, seq_length, latent_dim=64):
        z = torch.randn(batch_size, seq_length, latent_dim).to(next(self.parameters()).device)
        return self.generator(z)

    def gan_loss(self, real_eeg, batch_size, seq_length):
        synthetic_eeg = self.generate_synthetic_data(batch_size, seq_length)
        
        real_labels = torch.ones(batch_size, seq_length, 1).to(real_eeg.device)
        fake_labels = torch.zeros(batch_size, seq_length, 1).to(real_eeg.device)
        
        d_real = self.discriminator(real_eeg)
        d_fake = self.discriminator(synthetic_eeg.detach())
        
        d_loss = F.binary_cross_entropy_with_logits(d_real, real_labels) + \
                 F.binary_cross_entropy_with_logits(d_fake, fake_labels)
        
        g_fake = self.discriminator(synthetic_eeg)
        g_loss = F.binary_cross_entropy_with_logits(g_fake, real_labels)
        
        return d_loss, g_loss

    def forward(self, input_embeddings, input_masks, input_masks_invert,
                target_ids, lengths_words, word_contents,
                word_contents_attn, stepone, subjects, device, features=False):
        
        batch_size = input_embeddings[0].size(0)
        seq_length = input_embeddings[0].size(1)
        
        torch.cuda.empty_cache()
        
        d_loss, g_loss = self.gan_loss(input_embeddings[0], batch_size, seq_length)
        encoded_features = self.hybrid_encoder(input_embeddings[0], lengths_words[0])
        
        features = self.feature_embedded(input_embeddings, lengths_words, device)
        if len(features.shape) == 2:
            features = features.unsqueeze(0)
        aligned_features = self.temporal_align(features)
        
        subject_features = []
        for i, subject in enumerate(subjects):
            tmp = aligned_features[i].unsqueeze(1)
            tmp = self.conv1d_point(tmp)
            tmp = torch.swapaxes(tmp, 1, 2)
            subject_matrix = self.subject_matrices[self.subjects_map[subject]].to(device)
            tmp = torch.matmul(tmp, subject_matrix).squeeze()
            subject_features.append(tmp)
            
        subject_features = torch.stack(subject_features, 0).to(device) if len(subject_features) > 1 else subject_features[0].unsqueeze(0)
        
        brain_embedding = subject_features + self.pos_embedding
        brain_embedding = self.encoder(brain_embedding, src_key_padding_mask=input_masks_invert)
        brain_embedding = self.layernorm_embedding(brain_embedding)
        
        if stepone:
            word_embeddings = self.bart.model.encoder.embed_tokens(word_contents)
            alignment_loss = F.mse_loss(brain_embedding, word_embeddings)
            return alignment_loss + 0.1 * (d_loss + g_loss)
        else:
            decoded_features = self.lstm_decoder(brain_embedding)
            out = self.bart(
                inputs_embeds=decoded_features,
                attention_mask=input_masks,
                labels=target_ids,
                return_dict=True
            )
            
            return (out.logits, brain_embedding) if features else out.logits

class EnhancedTemporalAlignment(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.alignment_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )
        
    def forward(self, x):
        aligned = self.alignment_net(x)
        return aligned + torch.zeros_like(aligned)  # Skip connection with zero padding

class EnhancedLSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True
        )
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths = lengths.cpu()
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(x)
        if isinstance(outputs, torch.nn.utils.rnn.PackedSequence):
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return self.layer_norm(self.output_projection(outputs))

class FeatureEmbedded(nn.Module):
    def __init__(self, input_dim=105, hidden_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
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
