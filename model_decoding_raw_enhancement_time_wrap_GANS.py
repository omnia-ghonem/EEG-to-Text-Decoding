import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

class TimeWarpingLayer(nn.Module):
    def __init__(self, n_components=5, smooth_sigma=3.0):
        super().__init__()
        self.n_components = n_components
        self.smooth_sigma = smooth_sigma
        self.warping_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def smoothing(self, x):
        """Apply Gaussian smoothing to the input sequences"""
        return torch.from_numpy(
            gaussian_filter1d(x.detach().cpu().numpy(), self.smooth_sigma, axis=1)
        ).to(x.device)
    
    def generate_warping_functions(self, seq_len, batch_size):
        """Generate monotonic warping functions"""
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1).to(next(self.parameters()).device)
        warps = self.warping_net(t.unsqueeze(-1))
        warps = torch.cumsum(F.softplus(warps.squeeze(-1)), dim=1)
        warps = warps / warps[:, -1:]  # Normalize to [0, 1]
        return warps
    
    def apply_warping(self, x, warps):
        """Apply warping functions to the sequences"""
        batch_size, seq_len, n_features = x.shape
        device = x.device
        t_orig = torch.linspace(0, 1, seq_len).to(device)
        
        x_np = x.detach().cpu().numpy()
        warps_np = warps.detach().cpu().numpy()
        t_orig_np = t_orig.cpu().numpy()
        
        warped_sequences = []
        for i in range(batch_size):
            sequence = x_np[i]
            warp = warps_np[i]
            warped_sequence = []
            
            for j in range(n_features):
                f = interp1d(t_orig_np, sequence[:, j], kind='linear', fill_value='extrapolate')
                warped_sequence.append(f(warp))
            warped_sequences.append(np.stack(warped_sequence, axis=1))
        
        warped_tensor = torch.tensor(np.stack(warped_sequences, axis=0), 
                                   device=device, 
                                   dtype=x.dtype,
                                   requires_grad=x.requires_grad)
        return warped_tensor
    
    def forward(self, x):
        x_smooth = self.smoothing(x)
        warps = self.generate_warping_functions(x.size(1), x.size(0))
        warped_data = self.apply_warping(x_smooth, warps)
        warp_diff = torch.diff(warps, dim=1)
        smoothness_loss = torch.mean(torch.square(torch.diff(warp_diff, dim=1)))
        return warped_data, smoothness_loss

class Generator(nn.Module):
    def __init__(self, input_dim=104, latent_dim=64, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=1, batch_first=True, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
    def forward(self, z, condition=None):
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
        output, _ = self.lstm(z)
        return self.decoder(output)

class Discriminator(nn.Module):
    def __init__(self, input_dim=104, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, condition=None):
        if condition is not None:
            x = torch.cat([x, condition], dim=-1)
        features, _ = self.lstm(x)
        return self.classifier(features)

class HybridEncoder(nn.Module):
    def __init__(self, input_dim=104, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths = lengths.cpu()
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x)
        if isinstance(lstm_out, torch.nn.utils.rnn.PackedSequence):
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return self.dropout(self.norm(lstm_out))

class EnhancedTemporalAlignment(nn.Module):
    def __init__(self, input_dim=256, dropout=0.2):
        super().__init__()
        self.alignment_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.Dropout(dropout),
            nn.LayerNorm(1024)
        )
        
    def forward(self, x):
        aligned = self.alignment_net(x)
        return aligned + torch.zeros_like(aligned)

class EnhancedLSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths = lengths.cpu()
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(x)
        if isinstance(outputs, torch.nn.utils.rnn.PackedSequence):
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return self.layer_norm(self.dropout(self.output_projection(outputs)))

class FeatureEmbedded(nn.Module):
    def __init__(self, input_dim=105, hidden_dim=256, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
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
            outputs = self.dropout(outputs)
            
            sentence_embedding = []
            for i in range(lengths_sentence.shape[0]):
                sentence_embedding.append(outputs[int(lengths_sentence[i]-1), i, :])
            sentence_embedding = torch.stack(sentence_embedding, 0)
            sentence_embeddings.append(sentence_embedding)
            
        return torch.squeeze(torch.stack(sentence_embeddings, 0)).to(device)

class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=1024, decoder_embedding_size=1024,
                 additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048,
                 dropout_rate=0.2):
        super().__init__()
        self.time_warping = TimeWarpingLayer(n_components=5)    
        
        self.generator = Generator(dropout=dropout_rate)
        self.discriminator = Discriminator(dropout=dropout_rate)
        self.hybrid_encoder = HybridEncoder(dropout=dropout_rate)
        
        self.hidden_dim = 256
        self.feature_embedded = FeatureEmbedded(
            input_dim=104, 
            hidden_dim=self.hidden_dim, 
            dropout=dropout_rate
        )
        
        self.temporal_align = EnhancedTemporalAlignment(
            input_dim=self.hidden_dim, 
            dropout=dropout_rate
        )
        
        self.conv1d_point = nn.Conv1d(1, 32, 1, stride=1)
        self.dropout = nn.Dropout(dropout_rate)
        
        SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH',
                   'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS',
                   'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']
        self.subjects_map = {subj: idx for idx, subj in enumerate(SUBJECTS)}
        
        self.subject_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(32, 1) * 0.02)
            for _ in range(len(SUBJECTS))
        ])
        
        self.lstm_decoder = EnhancedLSTMDecoder(
            input_size=in_feature, 
            hidden_size=decoder_embedding_size,
            dropout=dropout_rate
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 56, in_feature) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature,
            nhead=additional_encoder_nhead,
            dim_feedforward=additional_encoder_dim_feedforward,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.layernorm_embedding = nn.LayerNorm(in_feature)
        self.bart = bart
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def augment_data(self, input_embeddings):
        # Add random noise
        noise = torch.randn_like(input_embeddings) * 0.01
        augmented = input_embeddings + noise
        
        # Random scaling
        scale = torch.randn(input_embeddings.size(0), 1, 1).to(input_embeddings.device) * 0.1 + 1.0
        augmented = augmented * scale
        
        return augmented
    
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
                word_contents_attn, stepone, subjects, device, return_features=False):
        
        batch_size = input_embeddings[0].size(0)
        seq_length = input_embeddings[0].size(1)
        
        # Apply data augmentation during training
        if self.training:
            input_embeddings = [self.augment_data(emb) for emb in input_embeddings]
        
        torch.cuda.empty_cache()
        warped_embeddings = []
        total_warp_loss = 0
        for emb in input_embeddings:
            warped_emb, warp_loss = self.time_warping(emb)
            warped_emb = self.dropout(warped_emb)
            warped_embeddings.append(warped_emb)
            total_warp_loss += warp_loss
            
        d_loss, g_loss = self.gan_loss(warped_embeddings[0], batch_size, seq_length)
        encoded_features = self.hybrid_encoder(warped_embeddings[0], lengths_words[0])
        
        embedded_features = self.feature_embedded(warped_embeddings, lengths_words, device)
        if len(embedded_features.shape) == 2:
            embedded_features = embedded_features.unsqueeze(0)
        
        embedded_features = self.dropout(embedded_features)
        aligned_features = self.temporal_align(embedded_features)
        
        subject_features = []
        for i, subject in enumerate(subjects):
            tmp = aligned_features[i].unsqueeze(1)
            tmp = self.conv1d_point(tmp)
            tmp = torch.swapaxes(tmp, 1, 2)
            subject_matrix = self.subject_matrices[self.subjects_map[subject]].to(device)
            tmp = torch.matmul(tmp, subject_matrix).squeeze()
            tmp = self.dropout(tmp)
            subject_features.append(tmp)
            
        subject_features = torch.stack(subject_features, 0).to(device) if len(subject_features) > 1 else subject_features[0].unsqueeze(0)
        
        brain_embedding = subject_features + self.pos_embedding
        brain_embedding = self.encoder(brain_embedding, src_key_padding_mask=input_masks_invert)
        brain_embedding = self.layernorm_embedding(brain_embedding)
        brain_embedding = self.dropout(brain_embedding)
        
        if stepone:
            word_embeddings = self.bart.model.encoder.embed_tokens(word_contents)
            # Add L2 regularization to alignment loss
            alignment_loss = F.mse_loss(brain_embedding, word_embeddings)
            l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
            return alignment_loss + 0.1 * (d_loss + g_loss) + 0.01 * l2_reg
        else:
            decoded_features = self.lstm_decoder(brain_embedding)
            out = self.bart(
                inputs_embeds=decoded_features,
                attention_mask=input_masks,
                labels=target_ids,
                return_dict=True
            )
            
            return (out.logits, brain_embedding) if return_features else out.logits
