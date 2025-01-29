import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import json
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings('ignore')

class EEGAugmenter:
    def __init__(self, p_augment=0.5):
        self.p_augment = p_augment
        
    def _ensure_float32(self, eeg_data):
        """Ensure data is float32 tensor"""
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.tensor(eeg_data)
        if eeg_data.dtype != torch.float32:
            eeg_data = eeg_data.to(torch.float32)
        return eeg_data
        
    def gaussian_noise(self, eeg_data, std_factor=0.1):
        """Add Gaussian noise to the EEG signal"""
        eeg_data = self._ensure_float32(eeg_data)
        noise = torch.randn_like(eeg_data) * (torch.std(eeg_data) * std_factor)
        return eeg_data + noise
    
    def scaling(self, eeg_data, scaling_factor_range=(0.8, 1.2)):
        """Scale the amplitude of the EEG signal"""
        eeg_data = self._ensure_float32(eeg_data)
        scaling_factor = np.random.uniform(*scaling_factor_range)
        return eeg_data * scaling_factor
    
    def time_shift(self, eeg_data, max_shift_ratio=0.1):
        """Apply random time shift to the EEG signal"""
        eeg_data = self._ensure_float32(eeg_data)
            
        # Get time dimension
        if len(eeg_data.shape) == 3:  # (batch, time, channels)
            time_length = eeg_data.shape[1]
        else:  # (time, channels)
            time_length = eeg_data.shape[0]
            
        if time_length <= 1:
            return eeg_data
            
        max_shift = max(1, min(int(time_length * max_shift_ratio), time_length - 1))
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        if shift == 0:
            return eeg_data
            
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
        eeg_data = self._ensure_float32(eeg_data)
        
        time_length = eeg_data.shape[-2] if len(eeg_data.shape) == 3 else eeg_data.shape[0]
        if time_length <= 1:
            return eeg_data
            
        augmented = eeg_data.clone()
        hole_length = max(1, min(int(time_length * hole_length_ratio), time_length - 1))
        
        for _ in range(n_holes):
            start = np.random.randint(0, time_length - hole_length + 1)
            if len(eeg_data.shape) == 3:
                augmented[:, start:start+hole_length, :] = 0
            else:
                augmented[start:start+hole_length, :] = 0
                
        return augmented
    
    def channel_dropout(self, eeg_data, p_dropout=0.1):
        """Randomly drop out EEG channels"""
        eeg_data = self._ensure_float32(eeg_data)
        mask = torch.bernoulli(torch.ones_like(eeg_data) * (1 - p_dropout))
        return eeg_data * mask
    
    def frequency_shift(self, eeg_data, max_shift_ratio=0.1):
        """Apply random frequency shift using FFT"""
        eeg_data = self._ensure_float32(eeg_data)
        
        # Get time dimension
        if len(eeg_data.shape) == 3:
            time_length = eeg_data.shape[1]
        else:
            time_length = eeg_data.shape[0]
            
        if time_length <= 1:
            return eeg_data
            
        max_shift = max(1, min(int(time_length * max_shift_ratio), time_length - 1))
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        if shift == 0:
            return eeg_data
            
        try:
            if len(eeg_data.shape) == 3:
                fft = torch.fft.fft(eeg_data.float(), dim=1)
                if shift > 0:
                    fft[:, shift:, :] = fft[:, :-shift, :]
                    fft[:, :shift, :] = 0
                else:
                    fft[:, :shift, :] = fft[:, -shift:, :]
                    fft[:, shift:, :] = 0
                return torch.real(torch.fft.ifft(fft, dim=1))
            else:
                fft = torch.fft.fft(eeg_data.float(), dim=0)
                if shift > 0:
                    fft[shift:, :] = fft[:-shift, :]
                    fft[:shift, :] = 0
                else:
                    fft[:shift, :] = fft[-shift:, :]
                    fft[shift:, :] = 0
                return torch.real(torch.fft.ifft(fft, dim=0))
        except Exception as e:
            print(f"Warning: FFT operation failed: {str(e)}")
            return eeg_data

    def augment(self, eeg_data):
        """Apply random augmentations to the EEG data"""
        if np.random.random() > self.p_augment:
            return eeg_data
            
        eeg_data = self._ensure_float32(eeg_data)
            
        augmentations = [
            (self.gaussian_noise, 0.3),
            (self.scaling, 0.3),
            (self.time_shift, 0.2),
            (self.temporal_cutout, 0.2),
            (self.channel_dropout, 0.2),
            (self.frequency_shift, 0.2)
        ]
        
        augmented_data = eeg_data.clone()
        
        for aug_func, prob in augmentations:
            if np.random.random() < prob:
                try:
                    temp_data = aug_func(augmented_data)
                    if not torch.isnan(temp_data).any() and not torch.isinf(temp_data).any():
                        augmented_data = temp_data
                except Exception as e:
                    continue
                
        return augmented_data

def get_input_sample(sent_obj, tokenizer, eeg_type='GD', bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                    max_len=56, add_CLS_token=False, subj='unspecified', raw_eeg=False):
    """Process a single input sample"""
    if sent_obj is None or not sent_obj.get('word', []):
        return None

    try:
        input_sample = {}
        
        # Get target string and tokenize
        target_string = sent_obj['content']
        target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, 
                                   truncation=True, return_tensors='pt')
        input_sample['target_ids'] = target_tokenized['input_ids'][0]
        
        # Process word-level data
        word_embeddings = []
        word_raw_embeddings = []
        word_contents = []
        
        if add_CLS_token:
            word_embeddings.append(torch.ones(105*len(bands), dtype=torch.float32))

        valid_words = 0
        for word in sent_obj['word']:
            if not word:
                continue
                
            # Get EEG features for the word
            frequency_features = []
            try:
                for band in bands:
                    frequency_features.append(word['word_level_EEG'][eeg_type][eeg_type+band])
            except (KeyError, TypeError):
                continue
                
            word_eeg_embedding = np.concatenate(frequency_features)
            if len(word_eeg_embedding) != 105*len(bands):
                continue
                
            word_level_eeg_tensor = torch.tensor(word_eeg_embedding, dtype=torch.float32)
            
            # Get raw EEG if needed
            if raw_eeg:
                try:
                    word_raw_eeg = word['rawEEG'][0][:,:104]
                    word_raw_embeddings.append(torch.tensor(word_raw_eeg, dtype=torch.float32))
                except (KeyError, IndexError):
                    return None
            
            word_contents.append(word['content'])
            word_embeddings.append(word_level_eeg_tensor)
            valid_words += 1

        if valid_words < 1:
            return None

        # Pad sequences
        while len(word_embeddings) < max_len:
            word_embeddings.append(torch.zeros(105*len(bands), dtype=torch.float32))
            if raw_eeg:
                word_raw_embeddings.append(torch.zeros((1,104), dtype=torch.float32))

        # Create final tensors
        input_sample['word_embeddings'] = torch.stack(word_embeddings)
        if raw_eeg:
            input_sample['input_raw_embeddings'] = word_raw_embeddings
            
        # Create attention masks
        input_sample['input_attn_mask'] = torch.zeros(max_len)
        input_sample['input_attn_mask'][:valid_words + (1 if add_CLS_token else 0)] = 1
        
        input_sample['input_attn_mask_invert'] = torch.ones(max_len)
        input_sample['input_attn_mask_invert'][:valid_words + (1 if add_CLS_token else 0)] = 0
        
        input_sample['seq_len'] = torch.tensor(valid_words, dtype=torch.long)
        input_sample['subject'] = subj
        
        return input_sample

    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        return None

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', eeg_type='GD', 
                 bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], raweeg=False, 
                 setting='unique_sent', is_add_CLS_token=False, use_augmentation=True):
        """Initialize dataset with robust error handling"""
        
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inputs = []
        self.phase = phase
        self.augmenter = EEGAugmenter(p_augment=0.5) if use_augmentation else None
        
        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
            
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')
        
        try:
            for input_dataset_dict in input_dataset_dicts:
                if subject == 'ALL':
                    subjects = list(input_dataset_dict.keys())
                    print('[INFO]using subjects: ', subjects)
                else:
                    subjects = [subject]
                
                total_num_sentence = len(input_dataset_dict[subjects[0]])
                train_divider = int(0.8*total_num_sentence)
                dev_divider = train_divider + int(0.1*total_num_sentence)
                
                print(f'train divider = {train_divider}')
                print(f'dev divider = {dev_divider}')

                if setting == 'unique_sent':
                    for key in subjects:
                        start_idx = 0
                        end_idx = total_num_sentence
                        
                        if phase == 'train':
                            end_idx = train_divider
                        elif phase == 'dev':
                            start_idx = train_divider
                            end_idx = dev_divider
                        elif phase == 'test':
                            start_idx = dev_divider
                            
                        for i in range(start_idx, end_idx):
                            try:
                                sample = input_dataset_dict[key][i]
                                if not sample or not sample.get('word', []):
                                    continue
                                    
                                input_sample = get_input_sample(
                                    sample, self.tokenizer, eeg_type, 
                                    bands=bands, add_CLS_token=is_add_CLS_token,
                                    subj=key, raw_eeg=raweeg
                                )
                                
                                if input_sample is not None and input_sample['seq_len'] > 0:
                                    self.inputs.append(input_sample)
                            except Exception as e:
                                print(f"Error processing sample {i} for subject {key}: {str(e)}")
                                continue

                print('++ adding task to dataset, now we have:', len(self.inputs))
                
        except Exception as e:
            print(f"Error during dataset initialization: {str(e)}")
            raise

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """Get a sample from the dataset with proper error handling"""
        try:
            input_sample = self.inputs[idx].copy()
            
            # Apply augmentation only during training if augmenter exists
            if self.phase == 'train' and self.augmenter is not None:
                try:
                    # Ensure input embeddings are float32
                    input_sample['word_embeddings'] = input_sample['word_embeddings'].to(torch.float32)
                    input_sample['word_embeddings'] = self.augmenter.augment(input_sample['word_embeddings'])
                    
                    # Handle raw EEG data if present
                    if 'input_raw_embeddings' in input_sample:
                        if isinstance(input_sample['input_raw_embeddings'], list):
                            augmented_raw = []
                            for raw_emb in input_sample['input_raw_embeddings']:
                                try:
                                    raw_emb = raw_emb.to(torch.float32)
                                    aug_emb = self.augmenter.augment(raw_emb)
                                    augmented_raw.append(aug_emb)
                                except Exception as e:
                                    augmented_raw.append(raw_emb)
                            input_sample['input_raw_embeddings'] = augmented_raw
                        elif isinstance(input_sample['input_raw_embeddings'], torch.Tensor):
                            input_sample['input_raw_embeddings'] = input_sample['input_raw_embeddings'].to(torch.float32)
                            input_sample['input_raw_embeddings'] = self.augmenter.augment(input_sample['input_raw_embeddings'])
                except Exception as e:
                    print(f"Warning: Augmentation failed: {str(e)}")
            
            # Ensure all tensors are of correct type
            seq_len = input_sample['seq_len']
            if isinstance(seq_len, torch.Tensor):
                seq_len = seq_len.item()
            
            if seq_len <= 0:
                raise ValueError("Sequence length must be positive")
            
            return (
                input_sample['word_embeddings'],
                torch.tensor(seq_len, dtype=torch.long),
                input_sample['input_attn_mask'],
                input_sample['input_attn_mask_invert'],
                input_sample['target_ids'],
                input_sample.get('target_mask', torch.ones_like(input_sample['target_ids'])),
                input_sample.get('sentiment_label', torch.tensor(-100)),
                input_sample.get('sent_level_EEG', torch.zeros(105*len(input_sample['word_embeddings']))),
                input_sample.get('input_raw_embeddings', []),
                input_sample.get('word_contents', torch.zeros_like(input_sample['target_ids'])),
                input_sample.get('word_contents_attn', torch.ones_like(input_sample['target_ids'])),
                input_sample['subject']
            )
  
