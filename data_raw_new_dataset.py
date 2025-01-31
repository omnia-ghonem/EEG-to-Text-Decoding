import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io
import os
from transformers import BartTokenizer
from torch.nn.utils.rnn import pad_sequence

class HandwritingBCI_Dataset(Dataset):
    def __init__(self, data_dir="/kaggle/input/handwriting-bci", mode="sentences", 
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
            file_path = os.path.join(data_dir, "Datasets", session, f"{mode}.mat")
            if not os.path.exists(file_path):
                continue
                
            data = scipy.io.loadmat(file_path)
            
            # Extract neural data and labels
            if mode == "sentences":
                neural_data = data["neuralActivityTimeSeries"]
                labels = data["sentencePrompt"]
            else:  # characters mode
                neural_data = data["neuralActivityCube_A"]
                labels = data["characterCues"]
                
            for i in range(len(labels)):
                sample = self.get_input_sample(neural_data[i], labels[i])
                if sample:
                    self.inputs.append(sample)
                    
        print(f'[INFO] Loaded {len(self.inputs)} samples from {len(self.session_ids)} sessions for {phase}')

    def normalize_neural_data(self, neural_data):
        """Normalize neural data using z-score normalization"""
        mean = torch.mean(neural_data, dim=0)
        std = torch.std(neural_data, dim=0)
        return (neural_data - mean) / (std + 1e-8)

    def get_input_sample(self, neural_activity, label):
        """Process a single input sample
        Args:
            neural_activity: Raw neural activity data
            label: Text label/prompt
        Returns:
            Dictionary containing processed inputs
        """
        input_sample = {}
        
        # Convert label to string if needed
        if isinstance(label, np.ndarray):
            label = str(label[0])
            
        # Tokenize target text
        target_tokenized = self.tokenizer(
            label, 
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        input_sample['target_ids'] = target_tokenized['input_ids'][0]
        input_sample['target_mask'] = target_tokenized['attention_mask'][0]
        
        # Process neural data
        neural_tensor = torch.tensor(neural_activity, dtype=torch.float32)
        neural_tensor = self.normalize_neural_data(neural_tensor)
        
        # Pad or truncate to max_len
        if neural_tensor.shape[0] < self.max_len:
            padding = torch.zeros((self.max_len - neural_tensor.shape[0], neural_tensor.shape[1]))
            neural_tensor = torch.cat((neural_tensor, padding), dim=0)
        else:
            neural_tensor = neural_tensor[:self.max_len, :]
            
        input_sample['input_embeddings'] = neural_tensor
        input_sample['input_attn_mask'] = torch.ones(self.max_len)  # 1 indicates valid positions
        input_sample['input_attn_mask_invert'] = torch.zeros(self.max_len)  # 0 indicates valid positions
        input_sample['seq_len'] = torch.tensor(min(len(label), self.max_len))
        input_sample['subject'] = 't5'  # Single subject dataset
        
        return input_sample

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
            input_sample['subject']
        )

if __name__ == '__main__':
    # Test dataset loading
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    train_set = HandwritingBCI_Dataset(mode="sentences", tokenizer=tokenizer, phase='train')
    dev_set = HandwritingBCI_Dataset(mode="sentences", tokenizer=tokenizer, phase='dev')
    test_set = HandwritingBCI_Dataset(mode="sentences", tokenizer=tokenizer, phase='test')
    
    print(f'Train set size: {len(train_set)}')
    print(f'Dev set size: {len(dev_set)}')
    print(f'Test set size: {len(test_set)}')
