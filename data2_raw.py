import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import json
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Load sentiment labels
ZUCO_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
SST_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/stanfordsentiment/stanfordSentimentTreebank/ternary_dataset.json'))

def butter_bandpass_filter(signal, lowcut, highcut, fs=500, order=5):
    """Enhanced bandpass filter for EEG signal preprocessing"""
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return torch.Tensor(y).float()

def normalize_1d(input_tensor, eps=1e-8):
    """Enhanced normalization with numerical stability"""
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor) + eps
    return (input_tensor - mean) / std

def temporal_align_signal(signal, target_len=1000):
    """Temporally align signal using interpolation"""
    if len(signal) == target_len:
        return signal
    indices = np.linspace(0, len(signal)-1, target_len)
    return np.interp(indices, np.arange(len(signal)), signal)

def get_input_sample(sent_obj, tokenizer, eeg_type='GD', 
                    bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                    max_len=56, add_CLS_token=False, subj='unspecified', raw_eeg=False):
    """Enhanced input sample processing with temporal alignment"""
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        content = word_obj['content']
        
        for band in bands:
            band_features = word_obj['word_level_EEG'][eeg_type][eeg_type+band]
            # Apply bandpass filtering
            filtered_features = butter_bandpass_filter(
                band_features,
                lowcut=0.5,  # Adjust based on frequency band
                highcut=70.0 # Adjust based on frequency band
            )
            frequency_features.append(filtered_features)
            
        word_eeg_embedding = np.concatenate(frequency_features)
        
        if len(word_eeg_embedding) != 105*len(bands):
            print(f'Expected word: {content} of subj: {subj} eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
            
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_word_raweeg_tensor(word_obj):
        """Enhanced raw EEG processing with temporal alignment"""
        word_raw_eeg = word_obj['rawEEG'][0]
        
        # Apply temporal alignment
        aligned_signal = np.array([temporal_align_signal(channel) for channel in word_raw_eeg.T]).T
        
        # Apply bandpass filtering
        filtered_signal = butter_bandpass_filter(
            aligned_signal,
            lowcut=0.5,
            highcut=70.0
        )
        
        return torch.from_numpy(filtered_signal)

    def get_sent_eeg(sent_obj, bands):
        """Enhanced sentence-level EEG processing"""
        sent_eeg_features = []
        
        for band in bands:
            key = 'mean'+band
            band_features = sent_obj['sentence_level_EEG'][key]
            # Apply filtering
            filtered_features = butter_bandpass_filter(
                band_features,
                lowcut=0.5,
                highcut=70.0
            )
            sent_eeg_features.append(filtered_features)
            
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        return None

    input_sample = {}
    
    # Get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(
        target_string, 
        padding='max_length', 
        max_length=max_len, 
        truncation=True, 
        return_tensors='pt', 
        return_attention_mask=True
    )
    input_sample['target_ids'] = target_tokenized['input_ids'][0]

    # Get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # Handle sentiment labels
    target_string = target_string.replace('emp11111ty', 'empty').replace('film.1', 'film.')
    
    if target_string in ZUCO_SENTIMENT_LABELS:
        input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1)
    else:
        input_sample['sentiment_label'] = torch.tensor(-100)

    # Process word embeddings
    word_embeddings = []
    word_raw_embeddings = []
    word_contents = []

    if add_CLS_token:
        word_embeddings.append(torch.ones(104*len(bands)))

    for word in sent_obj['word']:
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands)
        
        if raw_eeg:
            try:
                word_level_raw_eeg_tensor = get_word_raweeg_tensor(word)
            except:
                print('Error in raw EEG processing:', word['content'])
                return None

        if word_level_eeg_tensor is None or torch.isnan(word_level_eeg_tensor).any():
            return None
            
        word_contents.append(word['content'])
        word_embeddings.append(word_level_eeg_tensor)

        if raw_eeg:
            word_level_raw_eeg_tensor = word_level_raw_eeg_tensor[:,:104]
            word_raw_embeddings.append(word_level_raw_eeg_tensor)

    if len(word_embeddings) < 1:
        return None

    # Padding
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))
        if raw_eeg:
            word_raw_embeddings.append(torch.zeros(1,104))

    # Tokenize word contents
    word_contents_tokenized = tokenizer(
        ' '.join(word_contents), 
        padding='max_length', 
        max_length=max_len, 
        truncation=True, 
        return_tensors='pt', 
        return_attention_mask=True
    )
    
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

    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = seq_len
    
    if seq_len == 0:
        print('Discarding zero-length instance:', target_string)
        return None

    input_sample['subject'] = subj
    return input_sample

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL',
                 eeg_type='GD', bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'],
                 raweeg=False, setting='unique_sent', is_add_CLS_token=False):
        """Enhanced dataset class with improved data loading and validation"""
        
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]Loading {len(input_dataset_dicts)} task datasets')

        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]Using subjects:', subjects)
            else:
                subjects = [subject]
            
            total_num_sentence = len(input_dataset_dict[subjects[0]])
            train_divider = int(0.8 * total_num_sentence)
            dev_divider = train_divider + int(0.1 * total_num_sentence)
            
            print(f'Train divider = {train_divider}')
            print(f'Dev divider = {dev_divider}')

            if setting == 'unique_sent':
                if phase == 'train':
                    print('[INFO]Initializing train set...')
                    range_start, range_end = 0, train_divider
                elif phase == 'dev':
                    print('[INFO]Initializing dev set...')
                    range_start, range_end = train_divider, dev_divider
                elif phase == 'test':
                    print('[INFO]Initializing test set...')
                    range_start, range_end = dev_divider, total_num_sentence
                elif phase == 'all':
                    print('[INFO]Initializing complete dataset...')
                    range_start, range_end = 0, total_num_sentence
                
                for key in subjects:
                    for i in range(range_start, range_end):
                        input_sample = get_input_sample(
                            input_dataset_dict[key][i],
                            self.tokenizer,
                            eeg_type,
                            bands=bands,
                            add_CLS_token=is_add_CLS_token,
                            subj=key,
                            raw_eeg=raweeg
                        )
                        if input_sample is not None:
                            self.inputs.append(input_sample)

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
