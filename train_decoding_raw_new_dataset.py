import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
from glob import glob
import time
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
import sys
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/data_raw_new_dataset.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/model_decoding_raw_new_dataset.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/config.py')
for path in sys.path:
    print(path)

import data_raw_new_dataset
import config
import model_decoding_raw_new_dataset
from torch.nn.utils.rnn import pad_sequence

from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from bert_score import score

import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()
torch.autograd.set_detect_anomaly(True)

from torch.utils.tensorboard import SummaryWriter
LOG_DIR = "runs_h"
train_writer = SummaryWriter(os.path.join(LOG_DIR, "train"))
val_writer = SummaryWriter(os.path.join(LOG_DIR, "train_full"))
dev_writer = SummaryWriter(os.path.join(LOG_DIR, "dev_full"))

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, 
                checkpoint_path_best='/kaggle/working/checkpoints/decoding_raw/best/temp_decoding.pt', 
                checkpoint_path_last='/kaggle/working/checkpoints/decoding_raw/last/temp_decoding.pt', stepone=False):
    since = time.time()

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    index_plot = 0
    index_plot_dev = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print(f"lr: {scheduler.get_lr()}")
        print('-' * 10)

        for phase in ['train', 'dev', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            if phase == 'test':
                target_tokens_list = []
                target_string_list = []
                pred_tokens_list = []
                pred_string_list = []

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for batch_idx, (neural_data, seq_len, input_masks, input_mask_invert, target_ids, target_mask, subject_batch) in enumerate(tepoch):
                    try:
                        # Move data to device
                        input_embeddings_batch = neural_data.float().to(device)
                        input_masks_batch = input_masks.to(device)
                        input_mask_invert_batch = input_mask_invert.to(device)
                        target_ids_batch = target_ids.to(device)
                        seq_len_batch = seq_len.to(device)

                        if phase == 'test' and not stepone:
                            target_tokens = tokenizer.convert_ids_to_tokens(
                                target_ids_batch[0].tolist(), skip_special_tokens=True)
                            target_string = tokenizer.decode(
                                target_ids_batch[0], skip_special_tokens=True)
                            target_tokens_list.append([target_tokens])
                            target_string_list.append(target_string)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            seq2seqLMoutput = model(
                                input_embeddings_batch,
                                input_masks_batch,
                                input_mask_invert_batch,
                                target_ids_batch,
                                seq_len_batch,
                                None,  # word_contents_batch
                                None,  # word_contents_attn_batch
                                stepone,
                                subject_batch,
                                device
                            )

                            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100
                            
                            if stepone:
                                loss = seq2seqLMoutput
                            else:
                                loss = criterion(seq2seqLMoutput.permute(0, 2, 1), target_ids_batch.long())

                            if phase == 'test' and not stepone:
                                logits = seq2seqLMoutput
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
                                pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
                                pred_tokens_list.append(pred_tokens)
                                pred_string_list.append(predicted_string)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        running_loss += loss.item() * input_embeddings_batch.size(0)
                        tepoch.set_postfix(loss=loss.item(), lr=scheduler.get_lr())

                        if phase == 'train':
                            val_writer.add_scalar("train_full", loss.item(), index_plot)
                            index_plot += 1
                        if phase == 'dev':
                            dev_writer.add_scalar("dev_full", loss.item(), index_plot_dev)
                            index_plot_dev += 1
                        
                        if phase == 'train':
                            scheduler.step()
                            
                    except RuntimeError as e:
                        print(f"Error in batch {batch_idx}: {str(e)}")
                        continue

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_epoch_loss = epoch_loss
                train_writer.add_scalar("train", epoch_loss, epoch)
                torch.save(model.state_dict(), checkpoint_path_last)
            elif phase == 'dev':
                val_losses.append(epoch_loss)
                train_writer.add_scalar("val", epoch_loss, epoch)
                train_writer.add_scalars('loss train/val', {
                    'train': train_epoch_loss,
                    'val': epoch_loss,
                }, epoch)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path_best)
                    print(f'update best on dev checkpoint: {checkpoint_path_best}')

            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'test' and not stepone:
                print("Evaluation on test")
                try:
                    weights_list = [(1.0,), (0.5, 0.5), (1./3., 1./3., 1./3.), (0.25, 0.25, 0.25, 0.25)]
                    for weight in weights_list:
                        bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
                        print(f'BLEU-{len(weight)} score: {bleu_score:.4f}')

                    rouge = Rouge()
                    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
                    print("ROUGE scores:", rouge_scores)

                    P, R, F1 = score(pred_string_list, target_string_list, lang='en', device=device)
                    print(f"BERTScore - P: {P.mean():.4f}, R: {R.mean():.4f}, F1: {F1.mean():.4f}")
                except Exception as e:
                    print(f"Evaluation failed: {str(e)}")

        print()

    torch.cuda.empty_cache()

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation loss: {best_loss:4f}')

    return model

def show_require_grad_layers(model):
    print('\nTrainable layers:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'  {name}')

if __name__ == "__main__":
    CHECKPOINT_DIR_BEST = '/kaggle/working/checkpoints/decoding_raw/best'
    CHECKPOINT_DIR_LAST = '/kaggle/working/checkpoints/decoding_raw/last'
    os.makedirs(CHECKPOINT_DIR_BEST, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR_LAST, exist_ok=True)
    dataset_setting = 'unique_sent'
    args = config.get_config('train_decoding')

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
    use_random_init = False

    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')

    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'

    if use_random_init:
        save_name = 'randinit_' + save_name

    output_checkpoint_name_best = os.path.join(CHECKPOINT_DIR_BEST, f'{save_name}.pt')
    output_checkpoint_name_last = os.path.join(CHECKPOINT_DIR_LAST, f'{save_name}.pt')

    with open(f'/kaggle/working/config/decoding_raw/{save_name}.json', 'w') as out_config:
        json.dump(args, out_config, indent=4)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    train_set = data_raw_new_dataset.HandwritingBCI_Dataset(mode="sentences", tokenizer=tokenizer, phase='train')
    dev_set = data_raw_new_dataset.HandwritingBCI_Dataset(mode="sentences", tokenizer=tokenizer, phase='dev')
    test_set = data_raw_new_dataset.HandwritingBCI_Dataset(mode="sentences", tokenizer=tokenizer, phase='test')

    print(f'Train set size: {len(train_set)}')
    print(f'Dev set size: {len(dev_set)}')
    print(f'Test set size: {len(test_set)}')

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    dataloaders = {
        'train': train_dataloader,
        'dev': val_dataloader,
        'test': test_dataloader
    }

    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = model_decoding_raw_new_dataset.BrainTranslator(
        bart, 
        in_feature=192,
        decoder_embedding_size=1024,
        additional_encoder_nhead=8,
        additional_encoder_dim_feedforward=2048
    ).to(device)

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = '/kaggle/input/eeg-to-text-gpt2/checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b20_10_10_5e-05_5e-05_unique_sent.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')

        model.freeze_pretrained_brain()

        optimizer_step2 = optim.SGD(filter(
                lambda p: p.requires_grad, model.parameters()), lr=step2_lr, momentum=0.9)

        exp_lr_scheduler_step2 = lr_scheduler.CyclicLR(optimizer_step2, 
                        base_lr = 0.0000005,
                        max_lr = 0.00005,
                        mode = "triangular2")

        criterion = nn.CrossEntropyLoss()

        print()
        print('=== start Step2 training ... ===')
        show_require_grad_layers(model)

        model.to(device)

        trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2,
                                    checkpoint_path_best=output_checkpoint_name_best, checkpoint_path_last=output_checkpoint_name_last, stepone=False)

        train_writer.flush()
        train_writer.close()
        val_writer.flush()
        val_writer.close()
        dev_writer.flush()
        dev_writer.close()

    else:
        if upload_first_run_step1:
            stepone_checkpoint_not_first = '/kaggle/input/xlnet-step1-second-time/checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b20_10_2_5e-05_5e-05_unique_sent.pt'
            print(f'not first run for step 1, load checkpoint: {stepone_checkpoint_not_first}')
            model.load_state_dict(torch.load(stepone_checkpoint_not_first))
        
        model.to(device)

        optimizer_step1 = optim.SGD(filter(
            lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)

        exp_lr_scheduler_step1 = lr_scheduler.CyclicLR(optimizer_step1, 
                     base_lr = step1_lr,
                     max_lr = 5e-3,
                     mode = "triangular2")

        criterion = nn.MSELoss()
        model.freeze_pretrained_bart()

        print('=== start Step1 training ... ===')
        show_require_grad_layers(model)
        
        model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1,
                            checkpoint_path_best=output_checkpoint_name_best, checkpoint_path_last=output_checkpoint_name_last, stepone=True)
        
        train_writer.flush()
        train_writer.close()
        val_writer.flush()
        val_writer.close()
        dev_writer.flush()
        dev_writer.close()
