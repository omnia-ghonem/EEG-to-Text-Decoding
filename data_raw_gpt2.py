import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


ZUCO_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
SST_SENTIMENT_LABELS = json.load(open('/kaggle/input/dataset/stanfordsentiment/stanfordSentimentTreebank/ternary_dataset.json'))

from scipy.signal import butter, lfilter
from scipy.signal import freqz
def butter_bandpass_filter(signal, lowcut, highcut, fs=500, order=5):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    
    return torch.Tensor(y).float()

def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 

def get_input_sample(sent_obj, tokenizer, eeg_type='GD', bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len=56, add_CLS_token=False, subj='unspecified', raw_eeg=False):
    # Add detailed error logging
    if sent_obj is None:
        print(f'[ERROR] Sentence object is None for subject {subj}')
        return None
        
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        try:
            frequency_features = []
            content = word_obj.get('content', '')
            
            # Verify EEG data structure
            if 'word_level_EEG' not in word_obj or eeg_type not in word_obj['word_level_EEG']:
                print(f'[ERROR] Missing EEG data for word: {content}')
                return None
                
            for band in bands:
                band_key = eeg_type + band
                if band_key not in word_obj['word_level_EEG'][eeg_type]:
                    print(f'[ERROR] Missing band {band_key} for word: {content}')
                    return None
                frequency_features.append(word_obj['word_level_EEG'][eeg_type][band_key])
                
            word_eeg_embedding = np.concatenate(frequency_features)
            expected_dim = 105 * len(bands)
            
            if len(word_eeg_embedding) != expected_dim:
                print(f'[ERROR] Wrong embedding dimension for word "{content}": got {len(word_eeg_embedding)}, expected {expected_dim}')
                return None
                
            return_tensor = torch.from_numpy(word_eeg_embedding).float()
            return normalize_1d(return_tensor)
            
        except Exception as e:
            print(f'[ERROR] Failed to process word embedding: {str(e)}')
            return None

    def get_sent_eeg(sent_obj, bands):
        try:
            if 'sentence_level_EEG' not in sent_obj:
                print('[ERROR] Missing sentence level EEG data')
                return None
                
            sent_eeg_features = []
            for band in bands:
                key = 'mean' + band
                if key not in sent_obj['sentence_level_EEG']:
                    print(f'[ERROR] Missing sentence level band {key}')
                    return None
                sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
                
            sent_eeg_embedding = np.concatenate(sent_eeg_features)
            expected_dim = 105 * len(bands)
            
            if len(sent_eeg_embedding) != expected_dim:
                print(f'[ERROR] Wrong sentence embedding dimension: got {len(sent_eeg_embedding)}, expected {expected_dim}')
                return None
                
            return_tensor = torch.from_numpy(sent_eeg_embedding).float()
            return normalize_1d(return_tensor)
            
        except Exception as e:
            print(f'[ERROR] Failed to process sentence EEG: {str(e)}')
            return None

    input_sample = {}
    
    # Get and validate target string
    target_string = sent_obj.get('content', '')
    if not target_string or not isinstance(target_string, str):
        print(f'[ERROR] Invalid target string for subject {subj}: {target_string}')
        return None

    try:
        target_tokenized = tokenizer(
            target_string,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        input_sample['target_ids'] = target_tokenized['input_ids'][0]
    except Exception as e:
        print(f'[ERROR] Tokenization failed for "{target_string}": {str(e)}')
        return None

    # Get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if sent_level_eeg_tensor is None or torch.isnan(sent_level_eeg_tensor).any():
        print(f'[ERROR] Invalid sentence level EEG tensor for: {target_string}')
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # Process sentiment label
    target_string = target_string.replace('emp11111ty', 'empty').replace('film.1', 'film.')
    input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS.get(target_string, -100) + 1)

    # Process word embeddings
    word_embeddings = []
    word_raw_embeddings = []
    word_contents = []

    if add_CLS_token:
        word_embeddings.append(torch.ones(105 * len(bands)))

    # Validate word list
    if 'word' not in sent_obj or not sent_obj['word']:
        print(f'[ERROR] Missing or empty word list for: {target_string}')
        return None

    for word in sent_obj['word']:
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands)
        
        if word_level_eeg_tensor is None or torch.isnan(word_level_eeg_tensor).any():
            print(f'[ERROR] Invalid word level EEG tensor for word in: {target_string}')
            return None
            
        word_contents.append(word.get('content', ''))
        word_embeddings.append(word_level_eeg_tensor)

        if raw_eeg:
            try:
                if 'rawEEG' not in word or not word['rawEEG']:
                    print(f'[ERROR] Missing raw EEG data for word in: {target_string}')
                    return None
                word_level_raw_eeg_tensor = torch.from_numpy(word['rawEEG'][0][:, :104]).float()
                word_raw_embeddings.append(word_level_raw_eeg_tensor)
            except Exception as e:
                print(f'[ERROR] Failed to process raw EEG: {str(e)}')
                return None

    if len(word_embeddings) < 1:
        print(f'[ERROR] No valid word embeddings for: {target_string}')
        return None

    # Pad sequences
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105 * len(bands)))
        if raw_eeg:
            word_raw_embeddings.append(torch.zeros(1, 104))

    # Process word contents
    try:
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
    except Exception as e:
        print(f'[ERROR] Word contents tokenization failed: {str(e)}')
        return None

    # Create attention masks
    input_sample['input_embeddings'] = torch.stack(word_embeddings)
    if raw_eeg:
        input_sample['input_raw_embeddings'] = word_raw_embeddings

    seq_len = len(sent_obj['word'])
    mask_len = seq_len + 1 if add_CLS_token else seq_len

    input_sample['input_attn_mask'] = torch.zeros(max_len)
    input_sample['input_attn_mask'][:mask_len] = torch.ones(mask_len)

    input_sample['input_attn_mask_invert'] = torch.ones(max_len)
    input_sample['input_attn_mask_invert'][:mask_len] = torch.zeros(mask_len)

    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = seq_len
    input_sample['subject'] = subj

    return input_sample

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', eeg_type='GD', 
                 bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], raweeg=False, 
                 setting='unique_sent', is_add_CLS_token=False):
        self.inputs = []
        self.tokenizer = tokenizer
        
        print(f'[INFO] Initializing dataset:')
        print(f'  Phase: {phase}')
        print(f'  Subject: {subject}')
        print(f'  EEG Type: {eeg_type}')
        print(f'  Setting: {setting}')
        
        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
            
        total_processed = 0
        total_failed = 0
        
        for dataset_idx, input_dataset_dict in enumerate(input_dataset_dicts):
            print(f'\n[INFO] Processing dataset {dataset_idx + 1}/{len(input_dataset_dicts)}')
            
            subjects = [subject] if subject != 'ALL' else list(input_dataset_dict.keys())
            print(f'[INFO] Processing subjects: {subjects}')
            
            for subj in subjects:
                if subj not in input_dataset_dict:
                    print(f'[WARNING] Subject {subj} not found in dataset')
                    continue
                    
                subject_data = input_dataset_dict[subj]
                total_sentences = len(subject_data)
                
                if total_sentences == 0:
                    print(f'[WARNING] No sentences found for subject {subj}')
                    continue
                    
                print(f'[INFO] Subject {subj}: {total_sentences} sentences')
                
                train_split = int(0.8 * total_sentences)
                dev_split = train_split + int(0.1 * total_sentences)
                
                if setting == 'unique_sent':
                    if phase == 'train':
                        sentence_range = range(train_split)
                    elif phase == 'dev':
                        sentence_range = range(train_split, dev_split)
                    elif phase == 'test':
                        sentence_range = range(dev_split, total_sentences)
                    elif phase == 'all':
                        sentence_range = range(total_sentences)
                    else:
                        print(f'[ERROR] Invalid phase: {phase}')
                        continue
                        
                    print(f'[INFO] Processing {len(sentence_range)} sentences for {phase} phase')
                    
                    for idx in sentence_range:
                        try:
                            input_sample = get_input_sample(
                                subject_data[idx],
                                self.tokenizer,
                                eeg_type,
                                bands=bands,
                                add_CLS_token=is_add_CLS_token,
                                subj=subj,
                                raw_eeg=raweeg
                            )
                            
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                                total_processed += 1
                            else:
                                total_failed += 1
                                
                        except Exception as e:
                            print(f'[ERROR] Failed to process sentence {idx} for subject {subj}: {str(e)}')
                            total_failed += 1
                            
                        if (idx + 1) % 100 == 0:
                            print(f'[INFO] Processed {idx + 1}/{len(sentence_range)} sentences')
        
        print(f'\n[FINAL SUMMARY]')
        print(f'  Successfully processed: {total_processed}')
        print(f'  Failed to process: {total_failed}')
        print(f'  Final dataset size: {len(self.inputs)}')
        
        if len(self.inputs) == 0:
            print('[ERROR] No valid samples were loaded. Common issues:')
            print('  - Missing or corrupted EEG data')
            print('  - Mismatched dimensions in EEG features')
            print('  - Invalid sentence content')
            print('  - Problems with word-level processing')
            raise ValueError("Dataset is empty. Please check the detailed error messages above.")

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
            input_sample.get('input_raw_embeddings', []),
            input_sample['word_contents'],
            input_sample['word_contents_attn'],
            input_sample['subject']
        )
