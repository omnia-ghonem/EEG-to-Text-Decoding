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
        """
        Initialize the EEG data augmenter
        Args:
            p_augment (float): Probability of applying augmentation to a sample
        """
        self.p_augment = p_augment
        
    def gaussian_noise(self, eeg_data, std_factor=0.1):
        """Add Gaussian noise to the EEG signal"""
        noise = torch.randn_like(eeg_data) * (torch.std(eeg_data) * std_factor)
        return eeg_data + noise
    
    def scaling(self, eeg_data, scaling_factor_range=(0.8, 1.2)):
        """Scale the amplitude of the EEG signal"""
        scaling_factor = np.random.uniform(*scaling_factor_range)
        return eeg_data * scaling_factor
    
    def time_shift(self, eeg_data, max_shift=10):
        """Apply random time shift to the EEG signal"""
        if len(eeg_data.shape) == 3:  # (batch, time, channels)
            shift = np.random.randint(-max_shift, max_shift)
            if shift > 0:
                return torch.cat([eeg_data[:, shift:, :], torch.zeros_like(eeg_data[:, :shift, :])], dim=1)
            else:
                return torch.cat([torch.zeros_like(eeg_data[:, :abs(shift), :]), eeg_data[:, :shift, :]], dim=1)
        else:  # (time, channels)
            shift = np.random.randint(-max_shift, max_shift)
            if shift > 0:
                return torch.cat([eeg_data[shift:, :], torch.zeros_like(eeg_data[:shift, :])], dim=0)
            else:
                return torch.cat([torch.zeros_like(eeg_data[:abs(shift), :]), eeg_data[:shift, :]], dim=0)

    def temporal_cutout(self, eeg_data, n_holes=1, hole_length=20):
        """Apply temporal cutout by zeroing random time segments"""
        augmented = eeg_data.clone()
        time_length = eeg_data.shape[-2] if len(eeg_data.shape) == 3 else eeg_data.shape[0]
        
        for _ in range(n_holes):
            start = np.random.randint(0, time_length - hole_length)
            if len(eeg_data.shape) == 3:
                augmented[:, start:start+hole_length, :] = 0
            else:
                augmented[start:start+hole_length, :] = 0
        return augmented
    
    def channel_dropout(self, eeg_data, p_dropout=0.1):
        """Randomly drop out EEG channels"""
        mask = torch.bernoulli(torch.ones_like(eeg_data) * (1 - p_dropout))
        return eeg_data * mask
    
    def frequency_shift(self, eeg_data, max_shift=2):
        """Apply random frequency shift using FFT"""
        # Convert to frequency domain
        if len(eeg_data.shape) == 3:
            fft = torch.fft.fft(eeg_data, dim=1)
            shift = np.random.randint(-max_shift, max_shift+1)
            # Shift frequency components
            if shift > 0:
                fft[:, shift:, :] = fft[:, :-shift, :]
                fft[:, :shift, :] = 0
            else:
                fft[:, :shift, :] = fft[:, -shift:, :]
                fft[:, shift:, :] = 0
            # Convert back to time domain
            return torch.real(torch.fft.ifft(fft, dim=1))
        else:
            fft = torch.fft.fft(eeg_data, dim=0)
            shift = np.random.randint(-max_shift, max_shift+1)
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
                augmented_data = aug_func(augmented_data)
                
        return augmented_data

def normalize_1d(input_tensor):
    """Normalize a 1D tensor"""
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 

def get_input_sample(sent_obj, tokenizer, eeg_type='GD', bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                    max_len=56, add_CLS_token=False, subj='unspecified', raw_eeg=False):
    """Process a single input sample"""
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        content = word_obj['content']
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            print(f'expect word: {content} of subj: {subj} eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)
    
    def get_word_raweeg_tensor(word_obj):
        word_raw_eeg = word_obj['rawEEG'][0] #1000
        return_tensor = torch.from_numpy(word_raw_eeg)
        return return_tensor

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        return None

    input_sample = {}
    
    # Get target string and tokenize with GPT2 tokenizer
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    # Get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # Handle sentiment labels
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    ZUCO_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
    if target_string in ZUCO_SENTIMENT_LABELS:
        input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1)
    else:
        input_sample['sentiment_label'] = torch.tensor(-100)

    # Get input embeddings
    word_embeddings = []
    word_raw_embeddings = []
    word_contents = []

    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))

    for word in sent_obj['word']:
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands)
        if raw_eeg:
            try:
                word_level_raw_eeg_tensor = get_word_raweeg_tensor(word)
            except:
                print('error in raw eeg')
                print(word['content'])
                print(sent_obj['content'])
                print()
                return None
                
        if word_level_eeg_tensor is None:
            return None
            
        if torch.isnan(word_level_eeg_tensor).any():
            return None
            
        word_contents.append(word['content'])
        word_embeddings.append(word_level_eeg_tensor)

        if raw_eeg:
            word_level_raw_eeg_tensor = word_level_raw_eeg_tensor[:,:104]
            word_raw_embeddings.append(word_level_raw_eeg_tensor)

    if len(word_embeddings) < 1:
        return None

    # Pad sequences to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))
        if raw_eeg:
            word_raw_embeddings.append(torch.zeros(1,104))

    # Tokenize word contents
    word_contents_tokenized = tokenizer(' '.join(word_contents), padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    
    input_sample['word_contents'] = word_contents_tokenized['input_ids'][0]
    input_sample['word_contents_attn'] = word_contents_tokenized['attention_mask'][0]
    input_sample['input_embeddings'] = torch.stack(word_embeddings)
    
    if raw_eeg:
        input_sample['input_raw_embeddings'] = word_raw_embeddings

    # Create attention masks
    input_sample['input_attn_mask'] = torch.zeros(max_len)
    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1)
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word']))

    input_sample['input_attn_mask_invert'] = torch.ones(max_len)
    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1)
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word']))

    # Set target mask and sequence length
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    input_sample['subject'] = subj
    return input_sample

def setup_tokenizer(tokenizer):
    """Set up tokenizer with padding token"""
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', eeg_type='GD', 
                 bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], raweeg=False, 
                 setting='unique_sent', is_add_CLS_token=False, use_augmentation=True):
        """
        Initialize the ZuCo dataset
        Args:
            use_augmentation (bool): Whether to use data augmentation
            Other args remain the same as original implementation
        """
        self.tokenizer = setup_tokenizer(tokenizer)
        self.inputs = []
        self.phase = phase
        self.augmenter = EEGAugmenter(p_augment=0.5) if use_augmentation else None
        
        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
            
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')
        
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
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, 
                                                         eeg_type, bands=bands, add_CLS_token=is_add_CLS_token, 
                                                         subj=key, raw_eeg=raweeg)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                                
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider, dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                         eeg_type, bands=bands, add_CLS_token=is_add_CLS_token,
                                                         subj=key, raw_eeg=raweeg)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                                
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider, total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                         eeg_type, bands=bands, add_CLS_token=is_add_CLS_token,
                                                         subj=key, raw_eeg=raweeg)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                                
                elif phase == 'all':
                    print('[INFO]initializing all dataset...')
                    for key in subjects:
                        for i in range(total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                         eeg_type, bands=bands, add_CLS_token=is_add_CLS_token,
                                                         subj=key, raw_eeg=raweeg)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                                
            elif setting == 'unique_subj':
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                         eeg_type, bands=bands, add_CLS_token=is_add_CLS_token,
                                                         subj=key, raw_eeg=raweeg)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                                
                elif phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                         eeg_type, bands=bands, add_CLS_token=is_add_CLS_token,
                                                         subj=key, raw_eeg=raweeg)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                                
                elif phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                         eeg_type, bands=bands, add_CLS_token=is_add_CLS_token,
                                                         subj=key, raw_eeg=raweeg)
                            if input_sample is not None:
                                self.inputs.append(input_sample)

            print('++ adding task to dataset, now we have:', len(self.inputs))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        
        # Apply augmentation only during training if augmenter exists
        if self.phase == 'train' and self.augmenter is not None:
            # Augment input embeddings
            input_sample['input_embeddings'] = self.augmenter.augment(input_sample['input_embeddings'])
            
            # Also augment raw EEG if present
            if isinstance(input_sample['input_raw_embeddings'], list):
                augmented_raw = []
                for raw_emb in input_sample['input_raw_embeddings']:
                    augmented_raw.append(self.augmenter.augment(raw_emb))
                input_sample['input_raw_embeddings'] = augmented_raw
            elif isinstance(input_sample['input_raw_embeddings'], torch.Tensor):
                input_sample['input_raw_embeddings'] = self.augmenter.augment(input_sample['input_raw_embeddings'])
        
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
