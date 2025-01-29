import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import json
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from scipy.signal import resample
from scipy.interpolate import interp1d

class EEGAugmenter:
    def __init__(self, p_augment=0.5):
        self.p_augment = p_augment
        
    def gaussian_noise(self, eeg_data, std_factor=0.1):
        """Add Gaussian noise to the EEG signal"""
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        noise = torch.randn_like(eeg_data) * (torch.std(eeg_data) * std_factor)
        return eeg_data + noise
    
    def scaling(self, eeg_data, scaling_factor_range=(0.8, 1.2)):
        """Scale the amplitude of the EEG signal"""
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        scaling_factor = np.random.uniform(*scaling_factor_range)
        return eeg_data * scaling_factor
    
    def time_shift(self, eeg_data, max_shift_ratio=0.1):
        """Apply random time shift to the EEG signal"""
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
            
        # Calculate max_shift based on sequence length
        if len(eeg_data.shape) == 3:  # (batch, time, channels)
            time_length = eeg_data.shape[1]
        else:  # (time, channels)
            time_length = eeg_data.shape[0]
            
        max_shift = max(1, int(time_length * max_shift_ratio))
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        if len(eeg_data.shape) == 3:
            if shift > 0:
                return torch.cat([eeg_data[:, shift:, :], torch.zeros_like(eeg_data[:, :shift, :])], dim=1)
            else:
                return torch.cat([torch.zeros_like(eeg_data[:, :abs(shift), :]), eeg_data[:, :shift, :]], dim=1)
        else:
            if shift > 0:
                return torch.cat([eeg_data[shift:, :], torch.zeros_like(eeg_data[:shift, :])], dim=0)
            else:
                return torch.cat([torch.zeros_like(eeg_data[:abs(shift), :]), eeg_data[:shift, :]], dim=0)

    def temporal_cutout(self, eeg_data, n_holes=1, hole_length_ratio=0.1):
        """Apply temporal cutout by zeroing random time segments"""
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
            
        augmented = eeg_data.clone()
        time_length = eeg_data.shape[-2] if len(eeg_data.shape) == 3 else eeg_data.shape[0]
        
        # Calculate hole_length based on sequence length
        hole_length = max(1, int(time_length * hole_length_ratio))
        
        # Ensure hole_length is not larger than the sequence
        hole_length = min(hole_length, time_length - 1)
        
        for _ in range(n_holes):
            if time_length <= hole_length:
                continue
                
            start = np.random.randint(0, time_length - hole_length)
            if len(eeg_data.shape) == 3:
                augmented[:, start:start+hole_length, :] = 0
            else:
                augmented[start:start+hole_length, :] = 0
                
        return augmented
    
    def channel_dropout(self, eeg_data, p_dropout=0.1):
        """Randomly drop out EEG channels"""
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        mask = torch.bernoulli(torch.ones_like(eeg_data) * (1 - p_dropout))
        return eeg_data * mask
    
    def frequency_shift(self, eeg_data, max_shift_ratio=0.1):
        """Apply random frequency shift using FFT"""
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
            
        # Calculate max_shift based on sequence length
        if len(eeg_data.shape) == 3:
            time_length = eeg_data.shape[1]
        else:
            time_length = eeg_data.shape[0]
            
        max_shift = max(1, int(time_length * max_shift_ratio))
        
        if len(eeg_data.shape) == 3:
            fft = torch.fft.fft(eeg_data, dim=1)
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                fft[:, shift:, :] = fft[:, :-shift, :]
                fft[:, :shift, :] = 0
            else:
                fft[:, :shift, :] = fft[:, -shift:, :]
                fft[:, shift:, :] = 0
            return torch.real(torch.fft.ifft(fft, dim=1))
        else:
            fft = torch.fft.fft(eeg_data, dim=0)
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                fft[shift:, :] = fft[:-shift, :]
                fft[:shift, :] = 0
            else:
                fft[:shift, :] = fft[-shift:, :]
                fft[shift:, :] = 0
            return torch.real(torch.fft.ifft(fft, dim=0))

    def augment(self, eeg_data):
        """Apply random augmentations to the EEG data"""
        if np.random.random() > self.p_augment:
            return eeg_data
            
        # Convert to tensor if necessary
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
            
        # List of augmentation functions with their probability of being applied
        augmentations = [
            (self.gaussian_noise, 0.3),
            (self.scaling, 0.3),
            (self.time_shift, 0.2),
            (self.temporal_cutout, 0.2),
            (self.channel_dropout, 0.2),
            (self.frequency_shift, 0.2)
        ]
        
        augmented_data = eeg_data.clone()
        
        # Apply random augmentations based on their probabilities
        for aug_func, prob in augmentations:
            if np.random.random() < prob:
                try:
                    augmented_data = aug_func(augmented_data)
                except Exception as e:
                    print(f"Warning: {aug_func.__name__} failed with error: {str(e)}")
                    continue
                
        return augmented_data

# [Previous normalize_1d and get_input_sample functions remain the same]

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', eeg_type='GD', 
                 bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], raweeg=False, 
                 setting='unique_sent', is_add_CLS_token=False, use_augmentation=True):
        self.tokenizer = setup_tokenizer(tokenizer)
        self.inputs = []
        self.phase = phase
        self.augmenter = EEGAugmenter(p_augment=0.5) if use_augmentation else None
        
        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
            
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')
        
        # [Rest of the initialization code remains the same]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx].copy()  # Create a copy to avoid modifying original data
        
        # Apply augmentation only during training if augmenter exists
        if self.phase == 'train' and self.augmenter is not None:
            try:
                # Augment input embeddings
                if isinstance(input_sample['input_embeddings'], torch.Tensor):
                    input_sample['input_embeddings'] = self.augmenter.augment(input_sample['input_embeddings'])
                
                # Handle raw EEG data
                if 'input_raw_embeddings' in input_sample:
                    if isinstance(input_sample['input_raw_embeddings'], list):
                        augmented_raw = []
                        for raw_emb in input_sample['input_raw_embeddings']:
                            try:
                                aug_emb = self.augmenter.augment(raw_emb)
                                augmented_raw.append(aug_emb)
                            except Exception as e:
                                print(f"Warning: Raw EEG augmentation failed: {str(e)}")
                                augmented_raw.append(raw_emb)  # Use original if augmentation fails
                        input_sample['input_raw_embeddings'] = augmented_raw
                    elif isinstance(input_sample['input_raw_embeddings'], torch.Tensor):
                        input_sample['input_raw_embeddings'] = self.augmenter.augment(input_sample['input_raw_embeddings'])
            except Exception as e:
                print(f"Warning: Augmentation failed: {str(e)}")
        
        return (
            input_sample['input_embeddings'],
            input_sample['seq_len'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'], 
            input_sample['target_ids'],
            input_sample['target_mask'],
            input_sample['sentiment_label'],
            input_sample['sent_level_EEG'],
            input_sample['input_raw_embeddings'],
            input_sample['word_contents'],
            input_sample['word_contents_attn'],
            input_sample['subject']
        )
