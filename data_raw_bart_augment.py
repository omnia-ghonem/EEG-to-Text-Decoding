import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from scipy.signal import butter, lfilter, resample
from scipy.interpolate import interp1d
import random

# EEG Augmentation Module
class EEGAugmentor:
    def __init__(self, noise_level=0.1, scaling_factor=0.2, shift_limit=0.1, 
                 dropout_rate=0.1, permutation_segments=5, mask_size=0.1):
        self.noise_level = noise_level
        self.scaling_factor = scaling_factor
        self.shift_limit = shift_limit
        self.dropout_rate = dropout_rate
        self.permutation_segments = permutation_segments
        self.mask_size = mask_size
    
    def add_gaussian_noise(self, eeg_data):
        noise = np.random.normal(0, self.noise_level, eeg_data.shape)
        return eeg_data + noise

    def scaling(self, eeg_data):
        scale_factor = 1.0 + random.uniform(-self.scaling_factor, self.scaling_factor)
        return eeg_data * scale_factor
    
    def time_shift(self, eeg_data):
        shift_points = int(len(eeg_data) * random.uniform(-self.shift_limit, self.shift_limit))
        return np.roll(eeg_data, shift_points, axis=0)
    
    def channel_dropout(self, eeg_data):
        mask = np.random.binomial(1, 1-self.dropout_rate, eeg_data.shape)
        return eeg_data * mask
    
    def temporal_permutation(self, eeg_data):
        segment_length = len(eeg_data) // self.permutation_segments
        segments = [eeg_data[i:i+segment_length] for i in range(0, len(eeg_data), segment_length)]
        if len(segments[-1]) < segment_length:
            segments = segments[:-1]
        random.shuffle(segments)
        return np.concatenate(segments)
    
    def masking(self, eeg_data):
        mask_length = int(len(eeg_data) * self.mask_size)
        start_idx = random.randint(0, len(eeg_data) - mask_length)
        eeg_data[start_idx:start_idx+mask_length] = 0
        return eeg_data
    
    def frequency_shift(self, eeg_data, sampling_rate=500):
        fft_data = np.fft.fft(eeg_data)
        freq_shift = random.randint(-5, 5)
        freq_points = len(eeg_data)
        shift_points = int(freq_shift * freq_points / sampling_rate)
        shifted_fft = np.roll(fft_data, shift_points)
        return np.real(np.fft.ifft(shifted_fft))
    
    def amplitude_scale(self, eeg_data):
        scale = 1.0 + random.uniform(-0.5, 0.5)
        mean_val = np.mean(eeg_data)
        centered_data = eeg_data - mean_val
        scaled_data = centered_data * scale
        return scaled_data + mean_val

def augment_zuco_dataset(dataset_dict, augmentor, num_augmentations=1):
    augmented_dict = {}
    
    for subject, sentences in dataset_dict.items():
        augmented_dict[subject] = []
        augmented_dict[subject].extend(sentences)
        
        for sentence in sentences:
            for aug_idx in range(num_augmentations):
                aug_sentence = dict(sentence)
                
                for word in aug_sentence['word']:
                    if 'word_level_EEG' in word:
                        for eeg_type in word['word_level_EEG']:
                            for band in word['word_level_EEG'][eeg_type]:
                                eeg_data = np.array(word['word_level_EEG'][eeg_type][band])
                                
                                if random.random() < 0.5:
                                    eeg_data = augmentor.add_gaussian_noise(eeg_data)
                                if random.random() < 0.5:
                                    eeg_data = augmentor.scaling(eeg_data)
                                if random.random() < 0.3:
                                    eeg_data = augmentor.time_shift(eeg_data)
                                if random.random() < 0.3:
                                    eeg_data = augmentor.frequency_shift(eeg_data)
                                if random.random() < 0.2:
                                    eeg_data = augmentor.amplitude_scale(eeg_data)
                                
                                word['word_level_EEG'][eeg_type][band] = eeg_data.tolist()
                    
                    if 'rawEEG' in word:
                        raw_eeg = np.array(word['rawEEG'][0])
                        
                        if random.random() < 0.5:
                            raw_eeg = augmentor.add_gaussian_noise(raw_eeg)
                        if random.random() < 0.4:
                            raw_eeg = augmentor.channel_dropout(raw_eeg)
                        if random.random() < 0.3:
                            raw_eeg = augmentor.temporal_permutation(raw_eeg)
                        
                        word['rawEEG'][0] = raw_eeg.tolist()
                
                augmented_dict[subject].append(aug_sentence)
    
    return augmented_dict

# Load sentiment labels
ZUCO_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
SST_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/stanfordsentiment/stanfordSentimentTreebank/ternary_dataset.json'))

def butter_bandpass_filter(signal, lowcut, highcut, fs=500, order=5):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return torch.Tensor(y).float()

def normalize_1d(input_tensor):
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 

def get_input_sample(sent_obj, tokenizer, eeg_type='GD', 
                    bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                    max_len=56, add_CLS_token=False, subj='unspecified', raw_eeg=False):
    
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
        word_raw_eeg = word_obj['rawEEG'][0]
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
    target_string = sent_obj['content']
    
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, 
                               truncation=True, return_tensors='pt', return_attention_mask=True)
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    if target_string in ZUCO_SENTIMENT_LABELS:
        input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1)
    else:
        input_sample['sentiment_label'] = torch.tensor(-100)

    word_embeddings = []
    word_raw_embeddings = []
    word_contents = []

    if add_CLS_token:
        word_embeddings.append(torch.ones(104*len(bands)))

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

    n_eeg_representations = len(word_embeddings)
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))
        if raw_eeg:
            word_raw_embeddings.append(torch.zeros(1,104))

    word_contents_tokenized = tokenizer(' '.join(word_contents), padding='max_length', 
                                      max_length=max_len, truncation=True, 
                                      return_tensors='pt', return_attention_mask=True)
   
    input_sample['word_contents'] = word_contents_tokenized['input_ids'][0]
    input_sample['word_contents_attn'] = word_contents_tokenized['attention_mask'][0]
    input_sample['input_embeddings'] = torch.stack(word_embeddings)
    
    if raw_eeg:
        input_sample['input_raw_embeddings'] = word_raw_embeddings

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

    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    input_sample['subject'] = subj

    return input_sample

class ZuCo_dataset(Dataset):
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
            # Apply augmentation if requested
            if use_augmentation and phase == 'train':
                augmentor = EEGAugmentor()
                input_dataset_dict = augment_zuco_dataset(
                    input_dataset_dict, 
                    augmentor,
                    num_augmentations
                )
            
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
                            input_sample = get_input_sample(input_dataset_dict[key][i], 
                                                          self.tokenizer, 
                                                          eeg_type,
                                                          bands=bands, 
                                                          add_CLS_token=is_add_CLS_token, 
                                                          subj=key,
                                                          raw_eeg=raweeg)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider, dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],
                                                          self.tokenizer,
                                                          eeg_type,
                                                          bands=bands, 
                                                          add_CLS_token=is_add_CLS_token, 
                                                          subj=key,
                                                          raw_eeg=raweeg)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                elif phase == 'all':
                    print('[INFO]initializing all dataset...')
                    for key in subjects:
                        for i in range(total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],
                                                          self.tokenizer,
                                                          eeg_type,
                                                          bands=bands, 
                                                          add_CLS_token=is_add_CLS_token, 
                                                          subj=key,
                                                          raw_eeg=raweeg)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider, total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],
                                                          self.tokenizer,
                                                          eeg_type,
                                                          bands=bands, 
                                                          add_CLS_token=is_add_CLS_token, 
                                                          subj=key,
                                                          raw_eeg=raweeg)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
            elif setting == 'unique_subj':
                print('WARNING!!! only implemented for SR v1 dataset')
                train_subjects = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']
                dev_subjects = ['ZMG']
                test_subjects = ['ZPH']
                
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in train_subjects:
                            input_sample = get_input_sample(input_dataset_dict[key][i],
                                                          self.tokenizer,
                                                          eeg_type,
                                                          bands=bands, 
                                                          add_CLS_token=is_add_CLS_token, 
                                                          subj=key)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in dev_subjects:
                            input_sample = get_input_sample(input_dataset_dict[key][i],
                                                          self.tokenizer,
                                                          eeg_type,
                                                          bands=bands, 
                                                          add_CLS_token=is_add_CLS_token, 
                                                          subj=key)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in test_subjects:
                            input_sample = get_input_sample(input_dataset_dict[key][i],
                                                          self.tokenizer,
                                                          eeg_type,
                                                          bands=bands, 
                                                          add_CLS_token=is_add_CLS_token, 
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

if __name__ == '__main__':
    check_dataset = 'ZuCo'
    
    if check_dataset == 'ZuCo':
        whole_dataset_dicts = []
        
        # Load Task 1 - Sentiment Reading
        dataset_path_task1 = '/kaggle/input/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        # Load Task 2 - Natural Reading
        dataset_path_task2 = '/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        # Load Task 2 v2
        dataset_path_task2_v2 = '/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_task2_v2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        print()
        for key in whole_dataset_dicts[0]:
            print(f'task2_v2, sentence num in {key}:', len(whole_dataset_dicts[0][key]))
        print()

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        dataset_setting = 'unique_sent'
        subject_choice = 'ALL'
        print(f'![Debug]using {subject_choice}')
        eeg_type_choice = 'GD'
        print(f'[INFO]eeg type {eeg_type_choice}') 
        bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
        print(f'[INFO]using bands {bands_choice}')

        # Create datasets with augmentation for training
        train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, 
                                subject=subject_choice, 
                                eeg_type=eeg_type_choice, 
                                bands=bands_choice, 
                                setting=dataset_setting, 
                                raweeg=True,
                                use_augmentation=True,  # Enable augmentation for training
                                num_augmentations=1)    # Number of augmented copies per sample

        # Create validation and test sets without augmentation
        dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, 
                              subject=subject_choice, 
                              eeg_type=eeg_type_choice, 
                              bands=bands_choice, 
                              setting=dataset_setting, 
                              raweeg=True)

        test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, 
                               subject=subject_choice, 
                               eeg_type=eeg_type_choice, 
                               bands=bands_choice, 
                               setting=dataset_setting, 
                               raweeg=True)

        print('trainset size:', len(train_set))
        print('devset size:', len(dev_set))
        print('testset size:', len(test_set))
