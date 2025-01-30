import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import random
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from scipy.signal import butter, lfilter, resample
from scipy.interpolate import interp1d
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from bert_score import score
import warnings
warnings.filterwarnings('ignore')

# Load sentiment labels
ZUCO_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
SST_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/stanfordsentiment/stanfordSentimentTreebank/ternary_dataset.json'))

def butter_bandpass_filter(signal, lowcut, highcut, fs=500, order=5):
    """Apply bandpass filter to the signal"""
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return torch.Tensor(y).float()

def normalize_1d(input_tensor):
    """Normalize 1D tensor"""
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor

class EEGAugmentor:
    def __init__(self, noise_level=0.1, scaling_factor=0.2, shift_limit=0.1, 
                 dropout_rate=0.1, permutation_segments=5, mask_size=0.1,
                 freq_shift_range=0.1, amplitude_scale_range=0.2):
        self.noise_level = noise_level
        self.scaling_factor = scaling_factor
        self.shift_limit = shift_limit
        self.dropout_rate = dropout_rate
        self.permutation_segments = permutation_segments
        self.mask_size = mask_size
        self.freq_shift_range = freq_shift_range
        self.amplitude_scale_range = amplitude_scale_range
    
    def add_gaussian_noise(self, eeg_data):
        """Add Gaussian noise to EEG data"""
        noise = np.random.normal(0, self.noise_level, eeg_data.shape)
        return eeg_data + noise

    def scaling(self, eeg_data):
        """Scale EEG data by a random factor"""
        scale_factor = 1.0 + random.uniform(-self.scaling_factor, self.scaling_factor)
        return eeg_data * scale_factor
    
    def time_shift(self, eeg_data):
        """Shift EEG data in time domain"""
        shift_points = int(len(eeg_data) * random.uniform(-self.shift_limit, self.shift_limit))
        return np.roll(eeg_data, shift_points, axis=0)
    
    def channel_dropout(self, eeg_data):
        """Randomly drop out channels"""
        mask = np.random.binomial(1, 1-self.dropout_rate, eeg_data.shape)
        return eeg_data * mask
    
    def temporal_permutation(self, eeg_data):
        """Permute temporal segments of EEG data"""
        if len(eeg_data) == 0:
            return eeg_data
            
        segment_length = len(eeg_data) // self.permutation_segments
        if segment_length == 0:
            return eeg_data
            
        segments = [eeg_data[i:i+segment_length] for i in range(0, len(eeg_data), segment_length)]
        if len(segments[-1]) < segment_length:
            segments = segments[:-1]
        if segments:
            random.shuffle(segments)
            return np.concatenate(segments)
        return eeg_data
    
    def masking(self, eeg_data):
        """Mask segments of EEG data"""
        if len(eeg_data) > 0:
            mask_length = max(1, int(len(eeg_data) * self.mask_size))
            start_idx = random.randint(0, len(eeg_data) - mask_length)
            masked_data = eeg_data.copy()
            masked_data[start_idx:start_idx+mask_length] = 0
            return masked_data
        return eeg_data

    def frequency_shift(self, eeg_data):
        """Shift frequency components of EEG data"""
        try:
            if len(eeg_data) == 0:
                return eeg_data
                
            # Apply FFT
            fft_data = np.fft.fft(eeg_data)
            freqs = np.fft.fftfreq(len(eeg_data))
            
            # Random frequency shift
            shift = random.uniform(-self.freq_shift_range, self.freq_shift_range)
            shifted_freqs = freqs + shift
            
            # Interpolate to get shifted spectrum
            interpolator = interp1d(freqs, fft_data, bounds_error=False, fill_value=0)
            shifted_fft = interpolator(shifted_freqs)
            
            # Inverse FFT
            shifted_data = np.fft.ifft(shifted_fft).real
            return shifted_data
            
        except Exception as e:
            print(f"Warning: Error in frequency shift: {e}")
            return eeg_data

    def amplitude_scale(self, eeg_data):
        """Scale amplitude of EEG data non-uniformly"""
        try:
            if len(eeg_data) == 0:
                return eeg_data
                
            scale = 1.0 + random.uniform(-self.amplitude_scale_range, self.amplitude_scale_range)
            scaled_data = np.power(np.abs(eeg_data), scale) * np.sign(eeg_data)
            return scaled_data
            
        except Exception as e:
            print(f"Warning: Error in amplitude scaling: {e}")
            return eeg_data

def augment_zuco_dataset(dataset_dict, augmentor, num_augmentations=1):
    """Augment the ZuCo dataset with various EEG transformations"""
    augmented_dict = {}
    
    for subject, sentences in dataset_dict.items():
        try:
            augmented_dict[subject] = []
            augmented_dict[subject].extend(sentences)
            
            for sentence in sentences:
                for aug_idx in range(num_augmentations):
                    try:
                        aug_sentence = dict(sentence)
                        
                        for word in aug_sentence['word']:
                            # Augment word-level EEG
                            if 'word_level_EEG' in word:
                                for eeg_type in word['word_level_EEG']:
                                    for band in word['word_level_EEG'][eeg_type]:
                                        try:
                                            eeg_data = np.array(word['word_level_EEG'][eeg_type][band])
                                            
                                            # Apply random augmentations with probabilities
                                            augmentations = [
                                                (0.5, augmentor.add_gaussian_noise),
                                                (0.5, augmentor.scaling),
                                                (0.3, augmentor.time_shift),
                                                (0.3, augmentor.channel_dropout),
                                                (0.2, augmentor.temporal_permutation),
                                                (0.2, augmentor.masking),
                                                (0.2, augmentor.frequency_shift),
                                                (0.2, augmentor.amplitude_scale)
                                            ]
                                            
                                            for prob, aug_func in augmentations:
                                                if random.random() < prob:
                                                    eeg_data = aug_func(eeg_data)
                                            
                                            word['word_level_EEG'][eeg_type][band] = eeg_data.tolist()
                                        except Exception as e:
                                            print(f"Warning: Could not augment word_level_EEG for band {band}: {e}")
                                            continue
                            
                            # Augment raw EEG if present
                            try:
                                if 'rawEEG' in word and word['rawEEG'] and len(word['rawEEG']) > 0:
                                    raw_eeg = np.array(word['rawEEG'][0])
                                    
                                    # Apply augmentations to raw EEG
                                    if random.random() < 0.5:
                                        raw_eeg = augmentor.add_gaussian_noise(raw_eeg)
                                    if random.random() < 0.4:
                                        raw_eeg = augmentor.channel_dropout(raw_eeg)
                                    if random.random() < 0.3:
                                        raw_eeg = augmentor.temporal_permutation(raw_eeg)
                                    if random.random() < 0.2:
                                        raw_eeg = augmentor.frequency_shift(raw_eeg)
                                    if random.random() < 0.2:
                                        raw_eeg = augmentor.amplitude_scale(raw_eeg)
                                    
                                    word['rawEEG'][0] = raw_eeg.tolist()
                            except Exception as e:
                                print(f"Warning: Could not augment rawEEG: {e}")
                                continue
                        
                        augmented_dict[subject].append(aug_sentence)
                    except Exception as e:
                        print(f"Warning: Could not augment sentence: {e}")
                        continue
                        
        except Exception as e:
            print(f"Warning: Could not process subject {subject}: {e}")
            continue
    
    return augmented_dict

def get_input_sample(sent_obj, tokenizer, eeg_type='GD', 
                    bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                    max_len=56, add_CLS_token=False, subj='unspecified', raw_eeg=False):
    """Process input sample for the model"""
    if sent_obj is None:
        return None

    try:
        def get_word_raweeg_tensor(word_obj):
            if 'rawEEG' not in word_obj or not word_obj['rawEEG'] or len(word_obj['rawEEG']) == 0:
                raise ValueError("No raw EEG data available")
            try:
                word_raw_eeg = word_obj['rawEEG'][0]
                return torch.from_numpy(np.array(word_raw_eeg))
            except (IndexError, ValueError) as e:
                raise ValueError(f"Error processing raw EEG: {e}")

        def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
            frequency_features = []
            content = word_obj['content']
            for band in bands:
                if eeg_type not in word_obj.get('word_level_EEG', {}):
                    raise ValueError(f"Missing EEG type {eeg_type}")
                band_key = eeg_type + band
                if band_key not in word_obj['word_level_EEG'][eeg_type]:
                    raise ValueError(f"Missing band {band_key}")
                frequency_features.append(word_obj['word_level_EEG'][eeg_type][band_key])
            
            word_eeg_embedding = np.concatenate(frequency_features)
            if len(word_eeg_embedding) != 105*len(bands):
                raise ValueError(f"Unexpected EEG embedding dimension")
            
            return normalize_1d(torch.from_numpy(word_eeg_embedding))

        # Process input sample
        input_sample = {}
        target_string = sent_obj['content']
        
        # Handle target tokenization
        target_tokenized = tokenizer(target_string, padding='max_length', 
                                   max_length=max_len, truncation=True, 
                                   return_tensors='pt', return_attention_mask=True)
        input_sample['target_ids'] = target_tokenized['input_ids'][0]
        
        # Process word-level data
        word_embeddings = []
        word_raw_embeddings = []
        word_contents = []

        if add_CLS_token:
            word_embeddings.append(torch.ones(105*len(bands)))

        for word in sent_obj['word']:
            try:
                word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands)
                if raw_eeg:
                    word_level_raw_eeg_tensor = get_word_raweeg_tensor(word)
            except Exception as e:
                print(f"Warning: Error processing word {word.get('content', 'unknown')}: {e}")
                continue

            if word_level_eeg_tensor is None or torch.isnan(word_level_eeg_tensor).any():
                continue
                
            word_contents.append(word.get('content', ''))
            word_embeddings.append(word_level_eeg_tensor)

            if raw_eeg:
                try:
                    word_level_raw_eeg_tensor = word_level_raw_eeg_tensor[:,:105]
                    word_raw_embeddings.append(word_level_raw_eeg_tensor)
                except Exception as e:
                    print(f"Warning: Error processing raw EEG tensor: {e}")
                    continue

        if len(word_embeddings) < 1:
            return None

        # Pad sequences to max_len
        while len(word_embeddings) < max_len:
            word_embeddings.append(torch.zeros(105*len(bands)))
            if raw_eeg:
                word_raw_embeddings.append(torch.zeros(1,105))

        # Process word contents
        word_contents_tokenized = tokenizer(' '.join(word_contents), 
                                          padding='max_length',
                                          max_length=max_len,
                                          truncation=True,
                                          return_tensors='pt',
                                          return_attention_mask=True)
       
        # Populate input sample dictionary
        input_sample['word_contents'] = word_contents_tokenized['input_ids'][0]
        input_sample['word_contents_attn'] = word_contents_tokenized['attention_mask'][0]
        input_sample['input_embeddings'] = torch.stack(word_embeddings)
        
        if raw_eeg:
            input_sample['input_raw_embeddings'] = word_raw_embeddings

        # Create attention masks
        input_sample['input_attn_mask'] = torch.zeros(max_len)
        input_sample['input_attn_mask_invert'] = torch.ones(max_len)

        seq_len = len(sent_obj['word'])
        if add_CLS_token:
            input_sample['input_attn_mask'][:seq_len+1] = torch.ones(seq_len+1)
            input_sample['input_attn_mask_invert'][:seq_len+1] = torch.zeros(seq_len+1)
        else:
            input_sample['input_attn_mask'][:seq_len] = torch.ones(seq_len)
            input_sample['input_attn_mask_invert'][:seq_len] = torch.zeros(seq_len)

        # Add additional metadata
        input_sample['target_mask'] = target_tokenized['attention_mask'][0]
        input_sample['seq_len'] = seq_len
        input_sample['subject'] = subj

        if seq_len == 0:
            print(f'Warning: Discarding zero-length instance: {target_string}')
            return None

        # Handle sentiment labels
        if 'emp11111ty' in target_string:
            target_string = target_string.replace('emp11111ty','empty')
        if 'film.1' in target_string:
            target_string = target_string.replace('film.1','film.')
        
        input_sample['sentiment_label'] = torch.tensor(
            ZUCO_SENTIMENT_LABELS.get(target_string, -100) + 1
        )

        return input_sample

    except Exception as e:
        print(f"Error processing input sample: {e}")
        return None

class ZuCo_dataset(Dataset):
    """ZuCo dataset class for EEG data"""
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL',
                 eeg_type='GD', bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'],
                 raweeg=False, setting='unique_sent', is_add_CLS_token=False,
                 use_augmentation=False, num_augmentations=1):
        
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
            
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')
        
        for input_dataset_dict in input_dataset_dicts:
            try:
                # Apply augmentation if requested and in training phase
                if use_augmentation and phase == 'train':
                    try:
                        augmentor = EEGAugmentor()
                        print('[INFO]Applying data augmentation...')
                        input_dataset_dict = augment_zuco_dataset(
                            input_dataset_dict, 
                            augmentor,
                            num_augmentations
                        )
                        print('[INFO]Augmentation completed')
                    except Exception as e:
                        print(f'[WARNING]Error during augmentation: {e}')
                        print('[INFO]Proceeding with original data')
                
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
                        print('[INFO]initializing train set...')
                        self._process_phase(input_dataset_dict, subjects, 0, train_divider,
                                         tokenizer, eeg_type, bands, is_add_CLS_token, raweeg)
                    elif phase == 'dev':
                        print('[INFO]initializing dev set...')
                        self._process_phase(input_dataset_dict, subjects, train_divider, dev_divider,
                                         tokenizer, eeg_type, bands, is_add_CLS_token, raweeg)
                    elif phase == 'test':
                        print('[INFO]initializing test set...')
                        self._process_phase(input_dataset_dict, subjects, dev_divider, total_num_sentence,
                                         tokenizer, eeg_type, bands, is_add_CLS_token, raweeg)
                elif setting == 'unique_subj':
                    self._process_unique_subj(input_dataset_dict, phase, total_num_sentence,
                                          tokenizer, eeg_type, bands, is_add_CLS_token, raweeg)
                    
            except Exception as e:
                print(f'[WARNING]Error processing dataset: {e}')
                continue
            
            print('++ adding task to dataset, now we have:', len(self.inputs))

    def _process_phase(self, input_dataset_dict, subjects, start_idx, end_idx,
                      tokenizer, eeg_type, bands, is_add_CLS_token, raweeg):
        """Process data for a specific phase (train/dev/test)"""
        for key in subjects:
            for i in range(start_idx, end_idx):
                try:
                    input_sample = get_input_sample(
                        input_dataset_dict[key][i],
                        tokenizer,
                        eeg_type,
                        bands=bands,
                        add_CLS_token=is_add_CLS_token,
                        subj=key,
                        raw_eeg=raweeg
                    )
                    if input_sample is not None:
                        self.inputs.append(input_sample)
                except Exception as e:
                    print(f'[WARNING]Error processing sample {i} for subject {key}: {e}')
                    continue

    def _process_unique_subj(self, input_dataset_dict, phase, total_num_sentence,
                           tokenizer, eeg_type, bands, is_add_CLS_token, raweeg):
        """Process data using unique subject setting"""
        subjects_map = {
            'train': ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'],
            'dev': ['ZMG'],
            'test': ['ZPH']
        }
        
        if phase not in subjects_map:
            raise ValueError(f"Invalid phase: {phase}")
            
        subjects = subjects_map[phase]
        print(f'[INFO]initializing {phase} set using unique_subj setting...')
        
        for i in range(total_num_sentence):
            for key in subjects:
                try:
                    input_sample = get_input_sample(
                        input_dataset_dict[key][i],
                        tokenizer,
                        eeg_type,
                        bands=bands,
                        add_CLS_token=is_add_CLS_token,
                        subj=key,
                        raw_eeg=raweeg
                    )
                    if input_sample is not None:
                        self.inputs.append(input_sample)
                except Exception as e:
                    print(f'[WARNING]Error processing sample {i} for subject {key}: {e}')
                    continue

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'], 
            input_sample['seq_len'],
            input_sample['input_attn_mask'], 
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'], 
            input_sample['target_mask'], 
            input_sample['sentiment_label'],
            input_sample.get('sent_level_EEG', None),
            input_sample.get('input_raw_embeddings', None),
            input_sample['word_contents'],
            input_sample['word_contents_attn'],
            input_sample['subject']
        )
