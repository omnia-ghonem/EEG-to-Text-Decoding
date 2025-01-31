import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io
import os
from transformers import BartTokenizer
from torch.nn.utils.rnn import pad_sequence

class HandwritingBCI_Dataset(Dataset):
    def __init__(self, data_dir="/kaggle/input/handwriting-bci/handwritingBCIData/Datasets", mode="sentences", 
                 tokenizer=None, max_len=201, phase='train'):
        """
        Dataset loader for Handwriting BCI dataset.
        Args:
            data_dir: Directory containing the .mat files
            mode: "sentences" or "characters" 
            tokenizer: BART tokenizer for text processing
            max_len: Maximum sequence length
            phase: 'train', 'dev', or 'test'
        """
        self.session_ids = [
            't5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09',
            't5.2019.12.11', 't5.2019.12.18', 't5.2019.12.20',
            't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13',
            't5.2020.01.15'
        ]
        
        # Split sessions into train/dev/test
        if phase == 'train':
            self.session_ids = self.session_ids[:7]  # First 7 sessions for training
        elif phase == 'dev':
            self.session_ids = self.session_ids[7:9]  # 2 sessions for validation
        else:  # test
            self.session_ids = self.session_ids[9:]  # Last session for testing
            
        self.inputs = []
        self.tokenizer = tokenizer if tokenizer else BartTokenizer.from_pretrained("facebook/bart-large")
        self.max_len = max_len
        
        # Load data from sessions
        for session in self.session_ids:
            try:
                # Construct path to session directory
                session_dir = os.path.join(data_dir, session)
                
                if mode == "sentences":
                    file_path = os.path.join(session_dir, "sentences.mat")
                else:
                    file_path = os.path.join(session_dir, "chars.mat")
                
                if not os.path.exists(file_path):
                    print(f"Warning: File not found - {file_path}")
                    continue
                
                print(f"Loading data from {file_path}")
                data = scipy.io.loadmat(file_path)
                
                # Extract neural data and text
                neural_data = data.get('neuralActivityTimeSeries', [])  # Shape: (n_trials, n_timepoints, n_channels)
                text_data = data.get('sentencePrompt', []) if mode == "sentences" else data.get('characterCues', [])
                
                if len(neural_data) == 0 or len(text_data) == 0:
                    print(f"Warning: Empty data in {file_path}")
                    continue
                
                print(f"Found {len(neural_data)} trials in {session}")
                
                # Process each trial
                for trial_idx in range(len(neural_data)):
                    sample = self.get_input_sample(
                        neural_data[trial_idx], 
                        text_data[trial_idx][0] if mode == "sentences" else text_data[trial_idx]
                    )
                    if sample is not None:
                        self.inputs.append(sample)
                        
            except Exception as e:
                print(f"Error processing {session}: {str(e)}")
                continue
                    
        print(f'[INFO] Loaded {len(self.inputs)} samples from {len(self.session_ids)} sessions for {phase}')

    def normalize_neural_data(self, neural_data):
        """Normalize neural data using z-score normalization"""
        mean = np.mean(neural_data, axis=0, keepdims=True)
        std = np.std(neural_data, axis=0, keepdims=True)
        return (neural_data - mean) / (std + 1e-8)

    def get_input_sample(self, neural_activity, text):
        """Process a single input sample"""
        try:
            input_sample = {}
            
            # Convert text to string if needed
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            elif not isinstance(text, str):
                text = str(text)
            
            # Clean text
            text = text.strip()
            if not text:
                return None
                
            # Tokenize target text
            target_tokenized = self.tokenizer(
                text, 
                padding='max_length',
                max_length=self.max_len,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            
            input_sample['target_ids'] = target_tokenized['input_ids'][0]
            input_sample['target_mask'] = target_tokenized['attention_mask'][0]
            
            # Process neural data
            if isinstance(neural_activity, np.ndarray):
                # Normalize neural data
                neural_activity = self.normalize_neural_data(neural_activity)
                
                # Convert to tensor
                neural_tensor = torch.tensor(neural_activity, dtype=torch.float32)
                
                # Ensure shape is correct (time, channels)
                if len(neural_tensor.shape) == 1:
                    neural_tensor = neural_tensor.unsqueeze(1)
                elif len(neural_tensor.shape) > 2:
                    neural_tensor = neural_tensor.reshape(-1, neural_tensor.shape[-1])
                
                # Original sequence length before padding
                seq_len = neural_tensor.shape[0]
                
                # Pad or truncate to max_len
                if neural_tensor.shape[0] < self.max_len:
                    padding = torch.zeros((self.max_len - neural_tensor.shape[0], neural_tensor.shape[1]))
                    neural_tensor = torch.cat((neural_tensor, padding), dim=0)
                else:
                    neural_tensor = neural_tensor[:self.max_len, :]
                    seq_len = self.max_len
                
                input_sample['input_embeddings'] = neural_tensor
                input_sample['input_attn_mask'] = torch.ones(self.max_len)
                input_sample['input_attn_mask_invert'] = torch.zeros(self.max_len)
                input_sample['seq_len'] = torch.tensor(seq_len)
                input_sample['subject'] = 't5'
                
                return input_sample
            else:
                print(f"Warning: Invalid neural activity data type: {type(neural_activity)}")
                return None
                
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            return None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'],
            input_sample['seq_len'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'], 
            input_sample['target_ids'],
            input_sample['target_mask'],
            input_sample['subject']
        )
