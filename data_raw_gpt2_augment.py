import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import json
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from scipy.signal import resample
import random
from scipy.interpolate import interp1d

class EEGAugmentor:
    def __init__(self, p=0.5):
        self.p = p
        
    def gaussian_noise(self, data, std=0.1):
        """Add Gaussian noise to the signal"""
        if random.random() < self.p:
            noise = np.random.normal(0, std, data.shape)
            return data + noise
        return data
    
    def scaling(self, data, factor_range=(0.8, 1.2)):
        """Scale the amplitude of the signal"""
        if random.random() < self.p:
            factor = random.uniform(*factor_range)
            return data * factor
        return data
    
    def time_shift(self, data, max_shift=10):
        """Shift the signal in time"""
        if random.random() < self.p:
            shift = random.randint(-max_shift, max_shift)
            return np.roll(data, shift, axis=-1)
        return data
    
    def temporal_crop(self, data, crop_factor=0.9):
        """Randomly crop and resize back to original length"""
        if random.random() < self.p:
            length = data.shape[-1]
            crop_length = int(length * crop_factor)
            start = random.randint(0, length - crop_length)
            cropped = data[..., start:start + crop_length]
            # Resize back to original length
            x_original = np.linspace(0, 1, length)
            x_cropped = np.linspace(0, 1, crop_length)
            interpolator = interp1d(x_cropped, cropped, axis=-1, kind='linear')
            return interpolator(x_original)
        return data
    
    def masking(self, data, mask_factor=0.1):
        """Randomly mask portions of the signal"""
        if random.random() < self.p:
            mask = np.random.rand(*data.shape) > mask_factor
            return data * mask
        return data
    
    def frequency_shift(self, data, sampling_rate=1000, max_shift=2):
        """Shift the frequency components"""
        if random.random() < self.p:
            # Compute FFT
            fft_data = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
            
            # Apply frequency shift
            shift = random.uniform(-max_shift, max_shift)
            shifted_freqs = freqs + shift
            
            # Interpolate to get shifted spectrum
            interpolator = interp1d(freqs, fft_data, bounds_error=False, fill_value=0)
            shifted_fft = interpolator(shifted_freqs)
            
            # Inverse FFT
            return np.real(np.fft.ifft(shifted_fft))
        return data
    
    def augment(self, data):
        """Apply a random combination of augmentations"""
        augmentations = [
            self.gaussian_noise,
            self.scaling,
            self.time_shift,
            self.temporal_crop,
            self.masking,
            self.frequency_shift
        ]
        
        # Apply 2-3 random augmentations
        num_augments = random.randint(2, 3)
        chosen_augments = random.sample(augmentations, num_augments)
        
        augmented_data = data.copy()
        for aug in chosen_augments:
            augmented_data = aug(augmented_data)
            
        return augmented_data

ZUCO_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
SST_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/stanfordsentiment/stanfordSentimentTreebank/ternary_dataset.json'))

def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 

def get_input_sample(sent_obj, tokenizer, eeg_type='GD', bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len=56, add_CLS_token=False, subj='unspecified', raw_eeg=False):
    
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
    
    def get_word_raweeg_tensor(word_obj, augmentor=None):
        word_raw_eeg = word_obj['rawEEG'][0] #1000
        
        # Apply augmentation if specified
        if augmentor is not None:
            word_raw_eeg = augmentor.augment(word_raw_eeg)
            
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
    
    # get target string and tokenize with GPT2 tokenizer
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # get sentiment label
    # handle special cases
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    if target_string in ZUCO_SENTIMENT_LABELS:
        input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1)
    else:
        input_sample['sentiment_label'] = torch.tensor(-100)

    # get input embeddings
    word_embeddings = []
    word_raw_embeddings = []
    word_contents = []

    if add_CLS_token:
        word_embeddings.append(torch.ones(104*len(bands)))

    for word in sent_obj['word']:
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands)
        if raw_eeg:
            try:
                word_level_raw_eeg_tensor = get_word_raweeg_tensor(word, self.augmentor if self.augment else None)
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

    # pad sequences to max_len
    n_eeg_representations = len(word_embeddings)
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))
        if raw_eeg:
            word_raw_embeddings.append(torch.zeros(1,104))

    # tokenize word contents with GPT2 tokenizer
    word_contents_tokenized = tokenizer(' '.join(word_contents), padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    
    input_sample['word_contents'] = word_contents_tokenized['input_ids'][0]
    input_sample['word_contents_attn'] = word_contents_tokenized['attention_mask'][0]

    input_sample['input_embeddings'] = torch.stack(word_embeddings)
    
    if raw_eeg:
        input_sample['input_raw_embeddings'] = word_raw_embeddings

    # create attention masks
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

    # set target mask
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    input_sample['subject'] = subj

    return input_sample

def setup_tokenizer(tokenizer):
    # Set up padding token
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', eeg_type='GD', 
                 bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], raweeg=False, 
                 setting='unique_sent', is_add_CLS_token=False, augment=False):
        # Setup tokenizer with padding token
        self.tokenizer = setup_tokenizer(tokenizer)
        self.augment = augment
        if self.augment:
            self.augmentor = EEGAugmentor(p=0.5)
        self.inputs = []
        # Tokenizer is now set up in the setup_tokenizer function call above

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
                                                         subj=key)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                                
                elif phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                         eeg_type, bands=bands, add_CLS_token=is_add_CLS_token,
                                                         subj=key)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                                
                elif phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                         eeg_type, bands=bands, add_CLS_token=is_add_CLS_token,
                                                         subj=key)
                            if input_sample is not None:
                                self.inputs.append(input_sample)

            print('++ adding task to dataset, now we have:', len(self.inputs))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
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
