import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import json
import time
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
import sys
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from bert_score import score
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()
import sys
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/data_raw_new_dataset.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/model_decoding_raw_new_dataset.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/config_new_dataset.py')
for path in sys.path:
    print(path)



import data_raw_new_dataset 
import config_new_dataset
import model_decoding_raw_new_dataset
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from bert_score import score

import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()
torch.autograd.set_detect_anomaly(True)
# Set up tensorboard logging
from torch.utils.tensorboard import SummaryWriter
LOG_DIR = "runs_handwriting"
train_writer = SummaryWriter(os.path.join(LOG_DIR, "train"))
val_writer = SummaryWriter(os.path.join(LOG_DIR, "train_full"))
dev_writer = SummaryWriter(os.path.join(LOG_DIR, "dev_full"))



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.get('cuda', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.setup_paths()
        
        # Initialize tokenizer and datasets
        self.setup_data()
        
        # Create model
        self.setup_model()
        
        # Setup logging
        self.writer = {
            'train': SummaryWriter(os.path.join(args['log_dir'], 'train')),
            'val': SummaryWriter(os.path.join(args['log_dir'], 'val')),
            'test': SummaryWriter(os.path.join(args['log_dir'], 'test'))
        }

    def setup_paths(self):
        """Setup directory paths"""
        # Create directories if they don't exist
        for dir_name in ['checkpoint_dir', 'log_dir']:
            if dir_name in self.args:
                path = Path(self.args[dir_name])
                path.mkdir(parents=True, exist_ok=True)

    def setup_data(self):
        """Initialize datasets and dataloaders"""
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        
        # Create datasets
        self.datasets = {
            phase: data_raw_new_dataset.HandwritingBCIDataset(
                root_dir=self.args['data_root'],
                phase=phase,
                tokenizer=self.tokenizer,
                session_ids=self.args.get('session_ids', None)
            )
            for phase in ['train', 'dev', 'test']
        }
        
        # Create dataloaders
        self.dataloaders = {
            'train': DataLoader(
                self.datasets['train'],
                batch_size=self.args['batch_size'],
                shuffle=True,
                collate_fn=data_raw_new_dataset.collate_fn,
                num_workers=self.args.get('num_workers', 4)
            ),
            'dev': DataLoader(
                self.datasets['dev'],
                batch_size=1,
                shuffle=False,
                collate_fn=data_raw_new_dataset.collate_fn,
                num_workers=self.args.get('num_workers', 4)
            ),
            'test': DataLoader(
                self.datasets['test'],
                batch_size=1,
                shuffle=False,
                collate_fn=data_raw_new_dataset.collate_fn,
                num_workers=self.args.get('num_workers', 4)
            )
        }

    def setup_model(self):
        """Initialize model, optimizer, and scheduler"""
        # Create BART model
        bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        
        # Create brain translator model
        self.model = model_decoding_raw_new_dataset.BrainTranslator(
            bart_model=bart,
            input_dim=192,  # Number of electrodes
            hidden_dim=self.args.get('hidden_dim', 512),
            embedding_dim=self.args.get('embedding_dim', 1024)
        ).to(self.device)

        # Training utilities
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        if not self.args.get('skip_step_one', False):
            # Step 1: Train neural encoder
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args['learning_rate_step1']
            )
        else:
            # Step 2: Fine-tune BART
            self.model.freeze_neural_encoder()
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args['learning_rate_step2']
            )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=3,
            factor=0.5
        )

    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        patience = self.args.get('patience', 5)
        patience_counter = 0
        
        for epoch in range(self.args['num_epochs']):
            logging.info(f"Epoch {epoch+1}/{self.args['num_epochs']}")
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss = self.validate()
            
            # Log metrics
            self.writer['train'].add_scalar('loss', train_loss, epoch)
            self.writer['val'].add_scalar('loss', val_loss, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % self.args.get('save_every', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Early stopping
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break

        # Final evaluation
        self.evaluate()

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(self.dataloaders['train'], desc='Training') as pbar:
            for batch in pbar:
                # Move data to device
                neural_data = batch['neural_data'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                neural_mask = batch['neural_mask'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                loss, _ = self.model(neural_data, neural_mask, input_ids, attention_mask)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        return total_loss / len(self.dataloaders['train'])

    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['dev'], desc='Validation'):
                # Move data to device
                neural_data = batch['neural_data'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                neural_mask = batch['neural_mask'].to(self.device)
                
                # Forward pass
                loss, _ = self.model(neural_data, neural_mask, input_ids, attention_mask)
                total_loss += loss.item()
                
        return total_loss / len(self.dataloaders['dev'])

    def evaluate(self):
        """Evaluation loop"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['test'], desc='Testing'):
                # Move data to device
                neural_data = batch['neural_data'].to(self.device)
                neural_mask = batch['neural_mask'].to(self.device)
                
                # Generate text
                outputs = self.model(neural_data, neural_mask)
                
                # Decode predictions
                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Store results
                results.extend(zip(batch['text'], predictions))
                
        # Save results
        self.save_results(results)
        
        return results

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = Path(self.args['checkpoint_dir']) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args
        }, path)
        logging.info(f"Saved checkpoint to {path}")

    def save_results(self, results):
        """Save evaluation results"""
        path = Path(self.args['log_dir']) / 'results.txt'
        with open(path, 'w') as f:
            for true_text, pred_text in results:
                f.write(f"True: {true_text}\n")
                f.write(f"Pred: {pred_text}\n")
                f.write("-" * 50 + "\n")

if __name__ == '__main__':
    # Get config arguments
    args = config_new_dataset.get_config('train_decoding')
    
    # Set random seeds
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    # Setup paths
    os.makedirs(args['save_path'], exist_ok=True)
    log_dir = os.path.join(args['save_path'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Add log dir to args
    args['log_dir'] = log_dir
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()



    print("\nTraining complete!")
