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

def get_input_sample(sent_obj, tokenizer, eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 56, add_CLS_token = False, subj='unspecified',raw_eeg=False):
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        content=word_obj['content']
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            print(f'expect word: {content} of subj: {subj} eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
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
        # print(f'  - skip bad sentence')   
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    
    # Sanitize input string
    if not target_string or not isinstance(target_string, str):
        print(f'Invalid target string for subject {subj}: {target_string}')
        return None
    
    try:
        # Add error handling for tokenization
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
        print(f'Tokenization error for string "{target_string}": {e}')
        return None

    
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # get sentiment label
    # handle some weird cases
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    if target_string in ZUCO_SENTIMENT_LABELS:
        input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1) # 0:Negative, 1:Neutral, 2:Positive
    else:
        input_sample['sentiment_label'] = torch.tensor(-100)

    # get input embeddings
    word_embeddings = []
    word_raw_embeddings = []
    word_contents = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(104*len(bands)))

    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)
        if raw_eeg:
            try:
                word_level_raw_eeg_tensor = get_word_raweeg_tensor(word)
            except:
                print('error in raw eeg')
                print(word['content'])
                print(sent_obj['content'])
                print()
                return None
        # check none, for v2 dataset
        if word_level_eeg_tensor is None:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            return None
            
        word_contents.append(word['content'])
        word_embeddings.append(word_level_eeg_tensor)

        if raw_eeg:
            word_level_raw_eeg_tensor = word_level_raw_eeg_tensor[:,:104]
            word_raw_embeddings.append(word_level_raw_eeg_tensor)

    if len(word_embeddings)<1:
        return None
    

    # pad to max_len
    n_eeg_representations = len(word_embeddings)
    while len(word_embeddings) < max_len:
        # TODO: FBCSP
        word_embeddings.append(torch.zeros(105*len(bands)))
        if raw_eeg:
            word_raw_embeddings.append(torch.zeros(1,104))

    try:
        # Add error handling for word contents tokenization
        word_contents_tokenized = tokenizer(' '.join(word_contents), padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
    
        input_sample['word_contents'] = word_contents_tokenized['input_ids'][0]
        input_sample['word_contents_attn'] = word_contents_tokenized['attention_mask'][0] #bart
    except Exception as e:
        print(f'Word contents tokenization error: {e}')
        return None

    input_sample['input_embeddings'] = torch.stack(word_embeddings) # max_len * (105*num_bands)
    
    if raw_eeg:
        input_sample['input_raw_embeddings'] = word_raw_embeddings

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    # clean 0 length data
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    # subject
    input_sample['subject']= subj

    return input_sample

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'],raweeg=False, setting = 'unique_sent', is_add_CLS_token = False):
        self.inputs = []
        self.tokenizer = tokenizer

        # Enhanced logging and error handling
        print(f'[DETAILED INFO] Initializing dataset with:')
        print(f'  Phase: {phase}')
        print(f'  Subject: {subject}')
        print(f'  EEG Type: {eeg_type}')
        print(f'  Bands: {bands}')
        print(f'  Setting: {setting}')
        print(f'  Add CLS Token: {is_add_CLS_token}')
        print(f'  Raw EEG: {raweeg}')

        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')

        total_processed_samples = 0
        total_failed_samples = 0

        for dataset_idx, input_dataset_dict in enumerate(input_dataset_dicts):
            print(f'[DETAILED INFO] Processing dataset {dataset_idx + 1}')
            
            # Determine subjects to process
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print(f'[INFO]using ALL subjects: {subjects}')
            else:
                subjects = [subject]
            
            print(f'[DETAILED INFO] Subjects to process: {subjects}')

            # Validate input dataset
            for subj in subjects:
                if subj not in input_dataset_dict:
                    print(f'[WARNING] Subject {subj} not found in dataset')
                    continue

                subject_data = input_dataset_dict[subj]
                total_num_sentence = len(subject_data)
                
                print(f'[DETAILED INFO] Subject {subj}: Total sentences = {total_num_sentence}')

                train_divider = int(0.8*total_num_sentence)
                dev_divider = train_divider + int(0.1*total_num_sentence)
                
                print(f'[DETAILED INFO] Dividers - Train: {train_divider}, Dev: {dev_divider}, Test: {total_num_sentence}')

                if setting == 'unique_sent':
                    # Determine range of sentences based on phase
                    if phase == 'train':
                        sentence_range = range(train_divider)
                        print(f'[INFO]initializing train set for {subj}')
                    elif phase == 'dev':
                        sentence_range = range(train_divider, dev_divider)
                        print(f'[INFO]initializing dev set for {subj}')
                    elif phase == 'all':
                        sentence_range = range(total_num_sentence)
                        print(f'[INFO]initializing all dataset for {subj}')
                    elif phase == 'test':
                        sentence_range = range(dev_divider, total_num_sentence)
                        print(f'[INFO]initializing test set for {subj}')
                    else:
                        print(f'[ERROR] Invalid phase: {phase}')
                        continue

                    # Process sentences in the determined range
                    for i in sentence_range:
                        try:
                            input_sample = get_input_sample(
                                subject_data[i], 
                                self.tokenizer, 
                                eeg_type, 
                                bands=bands, 
                                add_CLS_token=is_add_CLS_token, 
                                subj=subj, 
                                raw_eeg=raweeg
                            )
                            
                            if input_sample is not None:
                                input_sample['subject'] = subj
                                self.inputs.append(input_sample)
                                total_processed_samples += 1
                            else:
                                total_failed_samples += 1
                                print(f'[DEBUG] Failed to process sample {i} for subject {subj}')
                        except Exception as e:
                            print(f'[ERROR] Failed to process sample {i} for subject {subj}: {e}')
                            total_failed_samples += 1

        # Final logging
        print(f'[FINAL SUMMARY]')
        print(f'  Total processed samples: {total_processed_samples}')
        print(f'  Total failed samples: {total_failed_samples}')
        print(f'  Final dataset size: {len(self.inputs)}')

        # Raise an error if no samples were loaded
        if len(self.inputs) == 0:
            raise ValueError("No samples could be loaded from the dataset. Please check your data and preprocessing.")

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
