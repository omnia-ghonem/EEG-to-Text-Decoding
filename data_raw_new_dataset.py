import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_neural_data(neural_data):
    """Normalize neural data using z-score normalization"""
    mean = np.mean(neural_data, axis=0, keepdims=True)
    std = np.std(neural_data, axis=0, keepdims=True) + 1e-8
    return (neural_data - mean) / std

def smooth_neural_data(neural_data, smooth_factor=4.0):
    """Apply Gaussian smoothing to neural data"""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(neural_data, smooth_factor, axis=0)

class HandwritingBCIDataset(Dataset):
    def __init__(self, root_dir: str, phase: str, tokenizer, session_ids: Optional[List[str]] = None,
                 max_seq_len: int = 800, min_seq_len: int = 5):
        """
        Initialize HandwritingBCI dataset.
        
        Args:
            root_dir: Path to dataset root directory
            phase: 'train', 'dev', or 'test'
            tokenizer: Tokenizer for text processing
            session_ids: List of session IDs to include
            max_seq_len: Maximum sequence length (padded/truncated to this)
            min_seq_len: Minimum sequence length to include
        """
        self.root_dir = Path(root_dir)
        self.phase = phase
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        # Default session IDs
        if session_ids is None:
            session_ids = [
                't5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09',
                't5.2019.12.11', 't5.2019.12.18', 't5.2019.12.20',
                't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13',
                't5.2020.01.15'
            ]
        else:
            self.session_ids = session_ids
        
        logging.info(f"Initializing {phase} dataset from {root_dir}")
        logging.info(f"Using sessions: {self.session_ids}")
        
        # Input validation
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")
            
        dataset_dir = self.root_dir / 'Datasets'
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Datasets directory not found at: {dataset_dir}")
        
        self.inputs = []
        self._load_all_sessions()
        
        if len(self.inputs) == 0:
            raise ValueError(f"No valid samples found for {phase} set. Please check the data directory and session IDs.")
            
        logging.info(f"Loaded {len(self.inputs)} samples for {phase} set")

    def _load_all_sessions(self):
        """Load data from all specified sessions"""
        for session_id in self.session_ids:
            try:
                self._load_session(session_id)
                logging.info(f"Successfully loaded session: {session_id}")
            except Exception as e:
                logging.error(f"Error loading session {session_id}: {str(e)}")
                continue  # Continue with next session even if one fails

    def _load_session(self, session_id: str):
        """Load data from a single session"""
        try:
            # Construct full session path
            session_dir = self.root_dir / 'Datasets' / session_id
            sentences_file = session_dir / 'sentences.mat'
            
            if not sentences_file.exists():
                logging.warning(f"Sentences file not found at: {sentences_file}")
                return
            
            # Load sentence data
            sentence_data = loadmat(str(sentences_file))
            logging.info(f"Loaded sentences.mat from {session_id}")
            
            # Get splits
            total_sentences = len(sentence_data['sentencePrompt'])
            train_split = int(0.8 * total_sentences)
            dev_split = train_split + int(0.1 * total_sentences)
            
            logging.info(f"Session {session_id}: Total sentences = {total_sentences}")
            
            # Select appropriate slice based on phase
            if self.phase == 'train':
                start_idx, end_idx = 0, train_split
            elif self.phase == 'dev':
                start_idx, end_idx = train_split, dev_split
            else:  # test
                start_idx, end_idx = dev_split, total_sentences

            processed_count = 0
            excluded_count = 0
            
            # Process each sentence
            for idx in range(start_idx, end_idx):
                if 'excludedSentences' in sentence_data and sentence_data['excludedSentences'][idx][0]:
                    excluded_count += 1
                    continue
                    
                sample = self._process_sentence(sentence_data, idx)
                if sample is not None:
                    self.inputs.append(sample)
                    processed_count += 1
                
            logging.info(f"Session {session_id}: Processed {processed_count} samples, "
                        f"excluded {excluded_count} samples")
                    
        except Exception as e:
            logging.error(f"Error processing session {session_id}: {str(e)}")
            raise

    def _process_sentence(self, sentence_data: Dict, idx: int) -> Optional[Dict]:
        """Process a single sentence and its neural data"""
        try:
            # Get sentence text
            if 'intendedText' in sentence_data and sentence_data['intendedText'][idx][0].size > 0:
                text = str(sentence_data['intendedText'][idx][0][0])
            elif 'sentencePrompt' in sentence_data and sentence_data['sentencePrompt'][idx][0].size > 0:
                text = str(sentence_data['sentencePrompt'][idx][0][0])
            else:
                logging.warning(f"No text found for index {idx}")
                return None
            
            # Clean text
            text = text.replace('>', ' ').replace('~', '.').replace('#', '').strip()
            
            # Get neural data and sequence length
            neural_data = sentence_data['neuralActivityCube'][idx]
            seq_len = sentence_data['numTimeBinsPerSentence'][idx][0]
            if isinstance(seq_len, np.ndarray):
                seq_len = int(seq_len[0])
            
            # Skip if sequence too short
            if seq_len < self.min_seq_len:
                logging.debug(f"Skipping sequence with length {seq_len} < {self.min_seq_len}")
                return None
            
            # Preprocess neural data
            neural_data = neural_data[:seq_len]
            neural_data = normalize_neural_data(neural_data)
            neural_data = smooth_neural_data(neural_data)
            neural_data = torch.from_numpy(neural_data).float()

            # Tokenize text
            encoded = self.tokenizer(
                text,
                padding='max_length',
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )

            # Create input sample
            sample = {
                'neural_data': neural_data,
                'seq_len': seq_len,
                'text': text,
                'input_ids': encoded['input_ids'][0],
                'attention_mask': encoded['attention_mask'][0],
                'neural_mask': self._create_neural_mask(seq_len)
            }

            return sample

        except Exception as e:
            logging.warning(f"Failed to process sentence {idx}: {str(e)}")
            return None

    def _create_neural_mask(self, seq_len: int) -> torch.Tensor:
        """Create attention mask for neural data"""
        mask = torch.zeros(self.max_seq_len)
        mask[:min(seq_len, self.max_seq_len)] = 1
        return mask

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset"""
        sample = self.inputs[idx]
        
        # Pad or truncate neural data
        neural_data = sample['neural_data']
        if neural_data.shape[0] < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - neural_data.shape[0], neural_data.shape[1])
            neural_data = torch.cat([neural_data, padding], dim=0)
        else:
            neural_data = neural_data[:self.max_seq_len]
            
        return {
            'neural_data': neural_data,
            'input_ids': sample['input_ids'],
            'attention_mask': sample['attention_mask'],
            'neural_mask': sample['neural_mask'],
            'seq_len': sample['seq_len'],
            'text': sample['text']
        }
def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader"""
    return {
        'neural_data': torch.stack([x['neural_data'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]), 
        'neural_mask': torch.stack([x['neural_mask'] for x in batch]),
        'seq_len': torch.tensor([x['seq_len'] for x in batch]),
        'text': [x['text'] for x in batch]
    }
