
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
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/data_raw.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/model_decoding_raw_bart_lora.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/config.py')
for path in sys.path:
    print(path)

import data_raw
import config
import model_decoding_raw_bart_lora
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




SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 
            'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS', 'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, 
                checkpoint_path_best='/kaggle/working/checkpoints/decoding_raw/best/temp_decoding.pt', 
                checkpoint_path_last='/kaggle/working/checkpoints/decoding_raw/last/temp_decoding.pt', stepone=False):
    since = time.time()
    best_loss = 100000000000
    train_losses = []
    val_losses = []
    index_plot = 0
    index_plot_dev = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
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
                for _, (_, seq_len, input_masks, input_mask_invert, target_ids, target_mask, 
                       sentiment_labels, sent_level_EEG, input_raw_embeddings, 
                       input_raw_embeddings_lengths, word_contents, word_contents_attn, subject_batch) in enumerate(tepoch):

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

                    seq2seqLMoutput = model(
                        input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                        target_ids_batch, input_embeddings_lengths_batch, word_contents_batch,
                        word_contents_attn_batch, stepone, subject_batch, device)

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
                        predicted_string = tokenizer.decode(predictions, skip_special_tokens=True)

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
                        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
                        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)

                    rouge = Rouge()
                    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True, ignore_empty=True)
                    print(rouge_scores)

                    P, R, F1 = score(pred_string_list, target_string_list, lang='en', device="cuda:0", 
                                   model_type="bert-large-uncased")
                    print(f"bert_score P: {np.mean(np.array(P))}")
                    print(f"bert_score R: {np.mean(np.array(R))}")
                    print(f"bert_score F1: {np.mean(np.array(F1))}")
                except:
                    print("failed")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')
    return model

def show_require_grad_layers(model):
    print('\nrequire_grad layers:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

if __name__ == '__main__':
    CHECKPOINT_DIR_BEST = '/kaggle/working/checkpoints/decoding_raw/best'
    CHECKPOINT_DIR_LAST = '/kaggle/working/checkpoints/decoding_raw/last'
    CONFIG_DIR = '/kaggle/working/config/decoding_raw'
    LOG_DIR = "runs_h"
    os.makedirs(CHECKPOINT_DIR_BEST, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR_LAST, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    args = config.get_config('train_decoding')
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
    use_random_init = False

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4

    print(f'[INFO]using model: {model_name}')
    print(f'[INFO]using use_random_init: {use_random_init}')

    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'

    if use_random_init:
        save_name = 'randinit_' + save_name

    output_checkpoint_name_best = f'/kaggle/working/checkpoints/decoding_raw/best/{save_name}.pt'
    output_checkpoint_name_last = f'/kaggle/working/checkpoints/decoding_raw/last/{save_name}.pt'

    subject_choice = args['subjects']
    eeg_type_choice = args['eeg_type']
    bands_choice = args['eeg_bands']

    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device(args['cuda'] if torch.cuda.is_available() else "cpu")
    print(f'[INFO]using device {device}')

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

    with open(f'/kaggle/working/config/decoding_raw/{save_name}.json', 'w') as out_config:
        json.dump(args, out_config, indent=4)

    tokenizer = T5Tokenizer.from_pretrained('t5-large')

    train_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice,
                                     eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, raweeg=True)
    dev_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice,
                                   eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, raweeg=True)
    test_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice,
                                    eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, raweeg=True)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set), 'test': len(test_set)}



    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    print('[INFO]test_set size: ', len(test_set))

    def pad_and_sort_batch(data_loader_batch):
        """
        data_loader_batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, word_contents, word_contents_attn, subject = tuple(
            zip(*data_loader_batch))

        raw_eeg = []
        input_raw_embeddings_lenghts = []
        for sentence in input_raw_embeddings:
            input_raw_embeddings_lenghts.append(
                torch.Tensor([a.size(0) for a in sentence]))
            raw_eeg.append(pad_sequence(
                sentence, batch_first=True, padding_value=0).permute(1, 0, 2))

        input_raw_embeddings = pad_sequence(
            raw_eeg, batch_first=True, padding_value=0).permute(0, 2, 1, 3)

        return input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lenghts, word_contents, word_contents_attn, subject  # lengths


    # Set up dataloaders
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                num_workers=0, collate_fn=pad_and_sort_batch)
    val_dataloader = DataLoader(dev_set, batch_size=1, shuffle=False, 
                              num_workers=0, collate_fn=pad_and_sort_batch)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, 
                               num_workers=0, collate_fn=pad_and_sort_batch)
    dataloaders = {'train': train_dataloader, 'dev': val_dataloader, 'test': test_dataloader}

    # Initialize T5 model
    if model_name == 'BrainTranslator':
        pretrained = T5ForConditionalGeneration.from_pretrained('t5-large')
        model = model_decoding_raw_bart_lora.BrainTranslator(pretrained, in_feature=1024, 
                                                 decoder_embedding_size=1024,
                                                 additional_encoder_nhead=8, 
                                                 additional_encoder_dim_feedforward=4096)
    
    model.to(device)

    # Training setup
    if model_name in ['BrainTranslator']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                else:
                    param.requires_grad = False

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = '/kaggle/input/notebookb08ade2eb3/checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b20_10_1_5e-05_5e-05_unique_sent.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
        
        # Step two training
        model.freeze_pretrained_brain()
        optimizer_step2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=step2_lr, momentum=0.9)
        exp_lr_scheduler_step2 = lr_scheduler.CyclicLR(optimizer_step2, 
                                                     base_lr=0.0000005,
                                                     max_lr=0.00005,
                                                     mode="triangular2")
        criterion = nn.CrossEntropyLoss()

        print('\n=== start Step2 training ... ===')
        show_require_grad_layers(model)
        model.to(device)

        trained_model = train_model(dataloaders, device, model, criterion, 
                                  optimizer_step2, exp_lr_scheduler_step2,
                                  num_epochs=num_epochs_step2,
                                  checkpoint_path_best=output_checkpoint_name_best,
                                  checkpoint_path_last=output_checkpoint_name_last,
                                  stepone=False)

    else:
        # Step one training
        if upload_first_run_step1:
            stepone_checkpoint_not_first = '/kaggle/input/t5-eeg-to-text-step-1-first/checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b20_10_1_5e-05_5e-05_unique_sent.pt'
            print(f'not first run for step 1, load checkpoint: {stepone_checkpoint_not_first}')
            model.load_state_dict(torch.load(stepone_checkpoint_not_first))
        
        model.to(device)
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=step1_lr, momentum=0.9)
        exp_lr_scheduler_step1 = lr_scheduler.CyclicLR(optimizer_step1,
                                                     base_lr=step1_lr,
                                                     max_lr=5e-3,
                                                     mode="triangular2")
        criterion = nn.MSELoss()
        model.freeze_pretrained_t5()

        print('=== start Step1 training ... ===')
        show_require_grad_layers(model)
        
        model = train_model(dataloaders, device, model, criterion,
                          optimizer_step1, exp_lr_scheduler_step1,
                          num_epochs=num_epochs_step1,
                          checkpoint_path_best=output_checkpoint_name_best,
                          checkpoint_path_last=output_checkpoint_name_last,
                          stepone=True)

    # Close tensorboard writers
    train_writer.close()
    val_writer.close() 
    dev_writer.close()
