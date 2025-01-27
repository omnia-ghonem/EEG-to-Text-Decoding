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
        
        # Detach tensors before converting to numpy
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
        
        # Convert back to tensor and maintain gradients if needed
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
    def __init__(self, input_dim=104, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, lengths=None):
        if lengths is not None:
            # Pack the sequence for LSTM processing
            packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed_x)
            # Unpack the sequence
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
            
        # Self-attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.norm(lstm_out + attn_out)


class EnhancedTemporalAlignment(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.input_dim = input_dim
        # Adjust the network to maintain input dimensions
        self.alignment_net = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 4),  # *2 because input is bidirectional
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 4, input_dim * 2),  # Match input dimension
            nn.LayerNorm(input_dim * 2)
        )
        
    def forward(self, x):
        # Now both tensors will have matching dimensions for the residual connection
        aligned = self.alignment_net(x)
        return aligned + x  # Residual connection will work as dimensions match


class EnhancedLSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, lengths=None):
        if lengths is not None:
            packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            outputs, _ = self.lstm(packed_x)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, _ = self.lstm(x)
            
        return self.layer_norm(self.output_projection(outputs))


class FeatureEmbedded(nn.Module):
    def __init__(self, input_dim=105, hidden_dim=256, num_layers=2):
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
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            
            # Use last hidden states from both directions
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            sentence_embeddings.append(final_hidden)
            
        return torch.stack(sentence_embeddings, 0).to(device)


class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=1024, decoder_embedding_size=1024,
                 additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super().__init__()
        
        # Core components
        self.time_warping = TimeWarpingLayer(n_components=5)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.hybrid_encoder = HybridEncoder()
        
        # Feature processing
        self.hidden_dim = 256  # Base hidden dimension
        self.feature_embedded = FeatureEmbedded(input_dim=104, hidden_dim=self.hidden_dim)
        # Pass the base hidden_dim - the class will handle bidirectional doubling
        self.temporal_align = EnhancedTemporalAlignment(input_dim=self.hidden_dim)
        self.conv1d_point = nn.Conv1d(1, 32, 1, stride=1)
        
        # Subject-specific processing
        SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH',
                   'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS',
                   'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']
        self.subjects_map = {subj: idx for idx, subj in enumerate(SUBJECTS)}
        
        # Initialize subject matrices with proper scaling
        self.subject_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(32, 1) / np.sqrt(32)) for _ in range(len(SUBJECTS))
        ])
        
        # Add layer normalization for subject features
        self.subject_norm = nn.LayerNorm(1)
        
        # Decoder components
        self.lstm_decoder = EnhancedLSTMDecoder(input_size=in_feature, hidden_size=decoder_embedding_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 56, in_feature))
        
        # Transformer encoder
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
        
        # BART model
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

    def generate_synthetic_data(self, batch_size, seq_length, latent_dim=64):
        z = torch.randn(batch_size, seq_length, latent_dim).to(next(self.parameters()).device)
        return self.generator(z)

    def gan_loss(self, real_eeg, batch_size, seq_length):
        synthetic_eeg = self.generate_synthetic_data(batch_size, seq_length)
        
        real_labels = torch.ones(batch_size, seq_length, 1).to(real_eeg.device)
        fake_labels = torch.zeros(batch_size, seq_length, 1).to(real_eeg.device)
        
        # Discriminator loss
        d_real = self.discriminator(real_eeg)
        d_fake = self.discriminator(synthetic_eeg.detach())
        d_loss = F.binary_cross_entropy_with_logits(d_real, real_labels) + \
                 F.binary_cross_entropy_with_logits(d_fake, fake_labels)
        
        # Generator loss
        g_fake = self.discriminator(synthetic_eeg)
        g_loss = F.binary_cross_entropy_with_logits(g_fake, real_labels)
        
        return d_loss, g_loss

    def forward(self, input_embeddings, input_masks, input_masks_invert,
                    target_ids, lengths_words, word_contents,
                    word_contents_attn, stepone, subjects, device, return_features=False):
            
            batch_size = input_embeddings[0].size(0)
            seq_length = input_embeddings[0].size(1)
            
            torch.cuda.empty_cache()
            
            # Apply time warping and process through hybrid encoder
            warped_embeddings = []
            total_warp_loss = 0
            for emb in input_embeddings:
                warped_emb, warp_loss = self.time_warping(emb)
                warped_embeddings.append(warped_emb)
                total_warp_loss += warp_loss
            
            # Calculate GAN losses
            d_loss, g_loss = self.gan_loss(warped_embeddings[0], batch_size, seq_length)
            
            # Process through hybrid encoder
            encoded_features = self.hybrid_encoder(warped_embeddings[0], lengths_words[0])
            
            # Feature embedding and temporal alignment
            embedded_features = self.feature_embedded(warped_embeddings, lengths_words, device)
            if len(embedded_features.shape) == 2:
                embedded_features = embedded_features.unsqueeze(0)
            aligned_features = self.temporal_align(embedded_features)
            
            # Process subject-specific features
            subject_features = []
            for i, subject in enumerate(subjects):
                # Reshape aligned features to work with conv1d
                tmp = aligned_features[i].unsqueeze(1)  # Add channel dimension
                
                # Get subject-specific matrix
                subject_idx = self.subjects_map[subject]
                subject_matrix = self.subject_matrices[subject_idx]  # Shape: [32, 1]
                
                # Apply 1D convolution
                conv_output = self.conv1d_point(tmp)  # Shape: [batch, 32, sequence_length]
                
                # Reshape conv_output to handle the 3D tensor properly
                # Permute to get shape [sequence_length, batch, 32]
                conv_output = conv_output.permute(2, 0, 1)
                
                # Matrix multiplication with broadcasting
                # [sequence_length, batch, 32] x [32, 1] -> [sequence_length, batch, 1]
                subject_output = torch.matmul(conv_output, subject_matrix)
                
                # Squeeze extra dimensions
                subject_output = subject_output.squeeze(-1).squeeze(-1)
                
                # Apply layer normalization
                subject_output = self.subject_norm(subject_output)
                
                subject_features.append(subject_output)
            
            # Stack all subject features
            subject_features = torch.stack(subject_features, dim=0)
            
            if stepone:
                if return_features:
                    return subject_features
                return F.mse_loss(subject_features, target_ids)
            
            # Add positional embeddings
            features_with_pos = encoded_features + self.pos_embedding[:, :encoded_features.size(1), :]
            features_with_pos = self.layernorm_embedding(features_with_pos)
            
            # Apply transformer encoder
            if input_masks is not None:
                encoded_features = self.encoder(features_with_pos, src_key_padding_mask=input_masks[0])
            else:
                encoded_features = self.encoder(features_with_pos)
            
            # Process through BART
            attention_mask = None if input_masks is None else input_masks[0]
            outputs = self.bart(
                inputs_embeds=encoded_features,
                attention_mask=attention_mask,
                labels=target_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            if return_features:
                return outputs.logits, subject_features, total_warp_loss
            return outputs.logits
