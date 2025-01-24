import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import json
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from transformers import MBartForConditionalGeneration, MBart50TokenizerFastimport sys
import warnings
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from bert_score import score
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim import lr_scheduler
from glob import glob
import time
import sys

sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/data_raw.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/model_decoding_raw_enhancement.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/config.py')

import data_raw
import config
import model_decoding_raw_enhancement

import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter

logging.set_verbosity_error()
torch.autograd.set_detect_anomaly(True)

LOG_DIR = "runs_h"
train_writer = SummaryWriter(os.path.join(LOG_DIR, "train"))
val_writer = SummaryWriter(os.path.join(LOG_DIR, "train_full"))
dev_writer = SummaryWriter(os.path.join(LOG_DIR, "dev_full"))



SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 
            'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS', 
            'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target, temporal_pred=None, temporal_target=None):
        ce = self.ce_loss(pred, target)
        if temporal_pred is not None and temporal_target is not None:
            mse = self.mse_loss(temporal_pred, temporal_target)
            return self.alpha * ce + (1 - self.alpha) * mse
        return ce

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, 
                num_epochs=25, checkpoint_path_best='best.pt', 
                checkpoint_path_last='last.pt', stepone=False):
    since = time.time()
    best_loss = float('inf')
    train_losses, val_losses = [], []
    index_plot, index_plot_dev = 0, 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print(f"lr: {scheduler.get_lr()}")
        print('-' * 10)

        for phase in ['train', 'dev', 'test']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0

            if phase == 'test':
                target_tokens_list = []
                target_string_list = []
                pred_tokens_list = []
                pred_string_list = []

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for batch_data in tepoch:
                    _, seq_len, input_masks, input_mask_invert, target_ids, target_mask, \
                    sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, \
                    word_contents, word_contents_attn, subject_batch = batch_data

                    input_embeddings_batch = input_raw_embeddings.float().to(device)
                    input_embeddings_lengths_batch = torch.stack([torch.tensor(a.clone().detach()) 
                                                                for a in input_raw_embeddings_lengths], 0).to(device)
                    input_masks_batch = torch.stack(input_masks, 0).to(device)
                    input_mask_invert_batch = torch.stack(input_mask_invert, 0).to(device)
                    target_ids_batch = torch.stack(target_ids, 0).to(device)
                    word_contents_batch = torch.stack(word_contents, 0).to(device)
                    word_contents_attn_batch = torch.stack(word_contents_attn, 0).to(device)
                    subject_batch = np.array(subject_batch)

                    if phase == 'test' and not stepone:
                        target_tokens = tokenizer.convert_ids_to_tokens(
                            target_ids_batch[0].tolist(), skip_special_tokens=True)
                        target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
                        target_tokens_list.append([target_tokens])
                        target_string_list.append(target_string)

                    optimizer.zero_grad()

                    # Enhanced forward pass with temporal features
                    outputs = model(input_embeddings_batch, input_masks_batch, 
                                 input_mask_invert_batch, target_ids_batch,
                                 input_embeddings_lengths_batch, word_contents_batch, 
                                 word_contents_attn_batch, stepone, subject_batch, device)

                    target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                    if stepone:
                        loss = outputs
                    else:
                        temporal_pred, temporal_target = None, None
                        if isinstance(outputs, tuple):
                            logits, temporal_features = outputs
                            temporal_pred = temporal_features
                            temporal_target = model.temporal_align(input_embeddings_batch)
                            loss = criterion(logits.permute(0, 2, 1), target_ids_batch.long(),
                                          temporal_pred, temporal_target)
                        else:
                            loss = criterion(outputs.permute(0, 2, 1), target_ids_batch.long())

                    if phase == 'test' and not stepone:
                        logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                        probs = logits[0].softmax(dim=1)
                        values, predictions = probs.topk(1)
                        predictions = torch.squeeze(predictions)
                        predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
                        
                        predictions = predictions.tolist()
                        truncated_prediction = []
                        for t in predictions:
                            if t != tokenizer.eos_token_id:
                                truncated_prediction.append(t)
                            else:
                                break
                        pred_tokens = tokenizer.convert_ids_to_tokens(
                            truncated_prediction, skip_special_tokens=True)
                        pred_tokens_list.append(pred_tokens)
                        pred_string_list.append(predicted_string)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * input_embeddings_batch.size()[0]
                    tepoch.set_postfix(loss=loss.item(), lr=scheduler.get_lr())

                    if phase == 'train':
                        val_writer.add_scalar("train_full", loss.item(), index_plot)
                        index_plot += 1
                    if phase == 'dev':
                        dev_writer.add_scalar("dev_full", loss.item(), index_plot_dev)
                        index_plot_dev += 1

                    if phase == 'train':
                        scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_epoch_loss = epoch_loss
                train_writer.add_scalar("train", epoch_loss, epoch)
                torch.save(model.state_dict(), checkpoint_path_last)
            elif phase == 'dev':
                val_losses.append(epoch_loss)
                train_writer.add_scalar("val", epoch_loss, epoch)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path_best)
                    print(f'Update best on dev checkpoint: {checkpoint_path_best}')

            if phase == 'test' and not stepone:
                print("Evaluation on test")
                try:
                    for n in range(1, 5):
                        weights = tuple([1.0/n] * n)
                        bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weights)
                        print(f'BLEU-{n} score: {bleu_score}')

                    rouge = Rouge()
                    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
                    print("ROUGE scores:", rouge_scores)

                    P, R, F1 = score(pred_string_list, target_string_list, lang='en', 
                                   device=device, model_type="bert-large-uncased")
                    print(f"BERTScore - P: {P.mean()}, R: {R.mean()}, F1: {F1.mean()}")
                except Exception as e:
                    print(f"Evaluation failed: {str(e)}")

            print(f'{phase} Loss: {epoch_loss:.4f}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')
    torch.save(model.state_dict(), checkpoint_path_last)
    return model

def show_require_grad_layers(model):
    print('\nRequire grad layers:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

if __name__ == '__main__':
    args = config.get_config('train_decoding')
    
    # Setup directories
    CHECKPOINT_DIR_BEST = '/kaggle/working/checkpoints/decoding_raw/best'
    CHECKPOINT_DIR_LAST = '/kaggle/working/checkpoints/decoding_raw/last'
    CONFIG_DIR = '/kaggle/working/config/decoding_raw'
    os.makedirs(CHECKPOINT_DIR_BEST, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR_LAST, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Training parameters
    dataset_setting = 'unique_sent'
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    model_name = args['model_name']
    task_name = args['task_name']
    save_path = args['save_path']
    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    upload_first_run_step1 = args['upload_first_run_step1']

    # Save configuration
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'

    output_checkpoint_name_best = f'{CHECKPOINT_DIR_BEST}/{save_name}.pt'
    output_checkpoint_name_last = f'{CHECKPOINT_DIR_LAST}/{save_name}.pt'

    # Setup device and seeds
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device(args['cuda'] if torch.cuda.is_available() else "cpu")

    # Load datasets
    whole_dataset_dicts = []
    if 'task1' in task_name:
        with open('/kaggle/input/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        with open('/kaggle/input/dataset2/task2-NR-dataset_wRaw.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        with open('/kaggle/input/dataset3/task2-NR-2.0-dataset_wRaw.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    with open(f'{CONFIG_DIR}/{save_name}.json', 'w') as out_config:
        json.dump(args, out_config, indent=4)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # Create datasets
    train_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=args['subjects'],
                                     eeg_type=args['eeg_type'], bands=args['eeg_bands'], 
                                     setting=dataset_setting, raweeg=True)
    dev_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=args['subjects'],
                                   eeg_type=args['eeg_type'], bands=args['eeg_bands'], 
                                   setting=dataset_setting, raweeg=True)
    test_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=args['subjects'],
                                    eeg_type=args['eeg_type'], bands=args['eeg_bands'], 
                                    setting=dataset_setting, raweeg=True)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set), 'test': len(test_set)}

    # Create data loaders
    def collate_fn(batch):
        input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, \
        sentiment_labels, sent_level_EEG, input_raw_embeddings, word_contents, word_contents_attn, subject = zip(*batch)

        raw_eeg = []
        input_raw_embeddings_lengths = []
        for sentence in input_raw_embeddings:
            input_raw_embeddings_lengths.append(torch.Tensor([a.size(0) for a in sentence]))
            raw_eeg.append(pad_sequence(sentence, batch_first=True, padding_value=0).permute(1, 0, 2))
    
        input_raw_embeddings = pad_sequence(raw_eeg, batch_first=True, padding_value=0).permute(0, 2, 1, 3)
        return input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, \
               sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, \
               word_contents, word_contents_attn, subject

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn),
        'dev': DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn),
        'test': DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    }

    # Initialize model
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    if model_name == 'BrainTranslator':
        model = model_decoding_raw_enhancement.BrainTranslator(
            pretrained_bart, 
            in_feature=1024, 
            decoder_embedding_size=1024,
            additional_encoder_nhead=8, 
            additional_encoder_dim_feedforward=4096
        )

    model.to(device)

    # Training process
    if model_name in ['BrainTranslator']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if any(x in name for x in ['shared', 'embed_positions', 'encoder.layers.0']):
                    continue
                param.requires_grad = False

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = '/kaggle/input/train-eeg-to-text/checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b20_10_25_5e-05_5e-05_unique_sent.pt'
            model.load_state_dict(torch.load(stepone_checkpoint))
        
        model.freeze_pretrained_brain()
        optimizer_step2 = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=step2_lr,
            momentum=0.9
        )
        scheduler_step2 = optim.lr_scheduler.CyclicLR(
            optimizer_step2,
            base_lr=5e-7,
            max_lr=5e-5,
            mode="triangular2"
        )
        
        criterion = CombinedLoss()
        trained_model = train_model(
            dataloaders, device, model, criterion, optimizer_step2, scheduler_step2,
            num_epochs=num_epochs_step2,
            checkpoint_path_best=output_checkpoint_name_best,
            checkpoint_path_last=output_checkpoint_name_last,
            stepone=False
        )
    else:
        if upload_first_run_step1:
            stepone_checkpoint_not_first = '/kaggle/input/xlnet-step1-second-time/checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b20_10_2_5e-05_5e-05_unique_sent.pt'
            model.load_state_dict(torch.load(stepone_checkpoint_not_first))
        
        model.freeze_pretrained_bart()
        optimizer_step1 = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=step1_lr,
            momentum=0.9
        )
        scheduler_step1 = optim.lr_scheduler.CyclicLR(
            optimizer_step1,
            base_lr=step1_lr,
            max_lr=5e-3,
            mode="triangular2"
        )
        
        criterion = nn.MSELoss()
        model = train_model(
            dataloaders, device, model, criterion, optimizer_step1, scheduler_step1,
            num_epochs=num_epochs_step1,
            checkpoint_path_best=output_checkpoint_name_best,
            checkpoint_path_last=output_checkpoint_name_last,
            stepone=True
        )

    # Cleanup
    for writer in [train_writer, val_writer, dev_writer]:
        writer.flush()
        writer.close()
