import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional, Tuple, Union
import warnings
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def butter_bandpass_filter(signal, lowcut, highcut, fs=500, order=5):
    """
    Apply bandpass filter to signal.
    Not used for handwriting BCI but kept for compatibility.
    """
    return torch.Tensor(signal).float()

def normalize_1d(input_tensor):
    """Normalize a 1D tensor using z-score normalization"""
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    return (input_tensor - mean) / (std + 1e-8)

class HandwritingBCIDataset(Dataset):
    """Dataset class for loading and processing handwriting BCI data."""
    
    def __init__(self, session_paths: Union[str, List[str]], phase: str, 
                 tokenizer, max_len: int = 56, min_seq_len: int = 5):
        """
        Initialize the HandwritingBCI dataset.
        
        Args:
            session_paths: Path or list of paths to session data directories
            phase: One of 'train', 'dev', or 'test' 
            tokenizer: BART tokenizer for text processing
            max_len: Maximum sequence length for padding
            min_seq_len: Minimum sequence length to include
        """
        self.inputs = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.min_seq_len = min_seq_len
        
        # Convert single path to list
        if isinstance(session_paths, str):
            session_paths = [session_paths]
            
        logging.info(f"Initializing {phase} dataset...")
        
        # Process each session
        for session_path in session_paths:
            self._load_session_data(session_path, phase)
            
        logging.info(f"Loaded {len(self.inputs)} samples for {phase}")

    def _load_session_data(self, session_path: str, phase: str):
        """
        Load and process data from a single session.
        
        Args:
            session_path: Path to session data directory
            phase: Dataset split to use ('train', 'dev', or 'test')
        """
        try:
            # Check if session path exists
            session_path = Path(session_path)
            if not session_path.exists():
                logging.error(f"Session path does not exist: {session_path}")
                return
                
            # Load sentence data
            sentences_file = session_path / 'sentences.mat'
            if not sentences_file.exists():
                logging.warning(f"Sentences file not found at {sentences_file}")
                return
                
            sentence_data = loadmat(str(sentences_file))
            
            # Get total number of sentences
            total_sentences = len(sentence_data['sentencePrompt'])
            train_split = int(0.8 * total_sentences)
            dev_split = train_split + int(0.1 * total_sentences)
            
            # Select appropriate slice based on phase
            start_idx = 0
            end_idx = total_sentences
            if phase == 'train':
                end_idx = train_split
            elif phase == 'dev':
                start_idx = train_split
                end_idx = dev_split
            elif phase == 'test':
                start_idx = dev_split
                
            # Process sentences in the selected range
            for idx in range(start_idx, end_idx):
                # Skip excluded sentences
                if 'excludedSentences' in sentence_data and sentence_data['excludedSentences'][idx][0]:
                    continue
                    
                input_sample = self._process_sentence(sentence_data, idx)
                if input_sample is not None:
                    self.inputs.append(input_sample)
                    
        except Exception as e:
            logging.error(f"Failed to load session {session_path}: {str(e)}")

    def _process_sentence(self, sentence_data: Dict, idx: int) -> Optional[Dict]:
        """
        Process a single sentence and its corresponding neural data.
        
        Args:
            sentence_data: Dictionary containing sentence and neural data
            idx: Index of sentence to process
            
        Returns:
            Dictionary containing processed input sample or None if processing fails
        """
        try:
            # Get sentence text - handle MATLAB string array format
            if 'intendedText' in sentence_data:
                text = sentence_data['intendedText'][idx][0]
                if isinstance(text, np.ndarray):
                    text = str(text[0])
            else:
                text = sentence_data['sentencePrompt'][idx][0]
                if isinstance(text, np.ndarray):
                    text = str(text[0])

            # Clean text
            text = self._clean_text(text)
            
            # Get neural data
            neural_data = sentence_data['neuralActivityCube'][idx]
            if isinstance(neural_data, np.ndarray):
                seq_len = sentence_data['numTimeBinsPerSentence'][idx][0]
                if isinstance(seq_len, np.ndarray):
                    seq_len = int(seq_len[0])
                
                # Skip if sequence is too short
                if seq_len < self.min_seq_len:
                    logging.warning(f"Skipping sentence {idx}: sequence length {seq_len} < minimum {self.min_seq_len}")
                    return None
                    
                neural_data = neural_data[:seq_len]
            
            # Convert to torch tensor and normalize
            neural_data = torch.from_numpy(neural_data).float()
            neural_data = self._normalize_neural_data(neural_data)
            
            # Tokenize target text
            target_tokenized = self.tokenizer(
                text,
                padding='max_length',
                max_length=self.max_len,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )

            # Create attention masks
            attn_mask = torch.zeros(self.max_len)  # 0 is masked
            attn_mask[:min(seq_len, self.max_len)] = 1  # 1 is not masked
            attn_mask_invert = 1 - attn_mask  # Inverted for transformer

            # Create input sample dictionary
            input_sample = {
                'input_embeddings': neural_data,
                'seq_len': seq_len,
                'input_attn_mask': attn_mask,
                'input_attn_mask_invert': attn_mask_invert,
                'target_ids': target_tokenized['input_ids'][0],
                'target_mask': target_tokenized['attention_mask'][0],
                'sentiment_label': torch.tensor(0),  # Placeholder
                'sent_level_EEG': torch.zeros(1),    # Placeholder
                'word_contents': target_tokenized['input_ids'][0],
                'word_contents_attn': target_tokenized['attention_mask'][0],
                'subject': 'T5'  # Single subject in dataset
            }

            # Add padding if needed
            if seq_len < self.max_len:
                padding = torch.zeros(self.max_len - seq_len, neural_data.shape[1])
                input_sample['input_embeddings'] = torch.cat([neural_data, padding], dim=0)
            else:
                input_sample['input_embeddings'] = neural_data[:self.max_len]
                
            return input_sample

        except Exception as e:
            logging.warning(f"Failed to process sentence {idx}: {str(e)}")
            return None
            
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text string"""
        text = text.replace('>', ' ')  # Replace special space character
        text = text.replace('~', '.')  # Replace special period character
        text = ' '.join(text.split())  # Normalize whitespace
        return text.strip()

    def _normalize_neural_data(self, neural_data: torch.Tensor) -> torch.Tensor:
        """
        Normalize neural data using z-score normalization.
        
        Args:
            neural_data: Tensor of neural activity data
            
        Returns:
            Normalized neural data tensor
        """
        mean = torch.mean(neural_data, dim=0, keepdim=True)
        std = torch.std(neural_data, dim=0, keepdim=True)
        return (neural_data - mean) / (std + 1e-8)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            Tuple containing all necessary data for model training/inference
        """
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
            input_sample['input_embeddings'].unsqueeze(0),
            input_sample['word_contents'],
            input_sample['word_contents_attn'],
            input_sample['subject']
        )

def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched and padded tensors
    """
    # Unzip the batch into separate lists
    (input_embeddings, seq_lens, input_masks, input_mask_inverts, target_ids, 
     target_masks, sentiment_labels, sent_level_EEGs, raw_embeddings, 
     word_contents, word_contents_attn, subjects) = zip(*batch)
    
    # Convert sequences to padded tensors
    input_embeddings_padded = pad_sequence(input_embeddings, batch_first=True, padding_value=0)
    input_masks_padded = pad_sequence(input_masks, batch_first=True, padding_value=0)
    input_mask_inverts_padded = pad_sequence(input_mask_inverts, batch_first=True, padding_value=1)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=1)
    target_masks_padded = pad_sequence(target_masks, batch_first=True, padding_value=0)
    word_contents_padded = pad_sequence(word_contents, batch_first=True, padding_value=1)
    word_contents_attn_padded = pad_sequence(word_contents_attn, batch_first=True, padding_value=0)
    
    # Process raw embeddings
    raw_embeddings_lengths = []
    raw_embeddings_list = []
    for emb in raw_embeddings:
        raw_embeddings_lengths.append(torch.tensor([e.size(0) for e in emb]))
        padded = pad_sequence(emb, batch_first=True, padding_value=0).permute(1, 0, 2)
        raw_embeddings_list.append(padded)
    
    raw_embeddings_padded = pad_sequence(raw_embeddings_list, batch_first=True, padding_value=0)
    raw_embeddings_padded = raw_embeddings_padded.permute(0, 2, 1, 3)
    
    return (
        input_embeddings_padded,
        torch.tensor(seq_lens),
        input_masks_padded,
        input_mask_inverts_padded,
        target_ids_padded,
        target_masks_padded,
        torch.stack(sentiment_labels),
        torch.stack(sent_level_EEGs),
        raw_embeddings_padded,
        raw_embeddings_lengths,
        word_contents_padded,
        word_contents_attn_padded,
        list(subjects)
    )

