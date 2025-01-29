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
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/config.py')
for path in sys.path:
    print(path)



from data_raw_new_dataset import HandwritingBCIDataset, collate_fn
import config
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




def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25,
                checkpoint_path_best='checkpoints/best/handwriting_model.pt',
                checkpoint_path_last='checkpoints/last/handwriting_model.pt',
                stepone=False):
    """
    Train the model on handwriting BCI data.
    """
    since = time.time()
    best_loss = float('inf')
    index_plot = 0
    index_plot_dev = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print(f"Learning rate: {scheduler.get_lr()}")
        print('-' * 10)

        # Each epoch has training, validation and test phases
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

            # Iterate over data
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for batch_idx, (input_embeddings, seq_len, input_masks, input_mask_invert,
                              target_ids, target_mask, sentiment_labels, sent_level_EEG,
                              input_raw_embeddings, input_raw_embeddings_lengths,
                              word_contents, word_contents_attn, subject_batch) in enumerate(tepoch):

                    # Move data to device
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
                        # Store ground truth for metrics calculation
                        target_tokens = tokenizer.convert_ids_to_tokens(
                            target_ids_batch[0].tolist(), skip_special_tokens=True)
                        target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
                        target_tokens_list.append([target_tokens])
                        target_string_list.append(target_string)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass
                    seq2seqLMoutput = model(
                        input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                        target_ids_batch, input_embeddings_lengths_batch, word_contents_batch,
                        word_contents_attn_batch, stepone, subject_batch, device)

                    # Replace padding ids with -100 for loss calculation
                    target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                    # Calculate loss
                    if stepone:
                        loss = seq2seqLMoutput
                    else:
                        loss = criterion(seq2seqLMoutput.permute(0, 2, 1), target_ids_batch.long())

                    # Generate predictions during testing
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

                    # Backward pass and optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Statistics
                    running_loss += loss.item() * input_embeddings_batch.size()[0]

                    tepoch.set_postfix(loss=loss.item(), lr=scheduler.get_lr())

                    # Log metrics
                    if phase == 'train':
                        val_writer.add_scalar("train_full", loss.item(), index_plot)
                        index_plot += 1
                    if phase == 'dev':
                        dev_writer.add_scalar("dev_full", loss.item(), index_plot_dev)
                        index_plot_dev += 1

                    if phase == 'train':
                        scheduler.step()

            # Calculate epoch loss
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # Log epoch metrics
            if phase == 'train':
                train_writer.add_scalar("train", epoch_loss, epoch)
                train_epoch_loss = epoch_loss
            elif phase == 'dev':
                train_writer.add_scalar("val", epoch_loss, epoch)
                train_writer.add_scalars('loss train/val', {
                    'train': train_epoch_loss,
                    'val': epoch_loss,
                }, epoch)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # Save best model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'Saved best model checkpoint to: {checkpoint_path_best}')

            # Calculate test metrics
            if phase == 'test' and not stepone:
                print("Evaluating test performance...")
                try:
                    # Calculate BLEU scores
                    for n in range(1, 5):
                        weights = tuple([1.0/n] * n)
                        bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weights)
                        print(f'BLEU-{n} score: {bleu_score:.4f}')

                    # Calculate ROUGE scores
                    rouge = Rouge()
                    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
                    print("ROUGE scores:", rouge_scores)

                    # Calculate BERTScore
                    P, R, F1 = score(pred_string_list, target_string_list,
                                   lang='en', device=device, model_type="bert-large-uncased")
                    print(f"BERTScore - P: {torch.mean(P):.4f}, R: {torch.mean(R):.4f}, F1: {torch.mean(F1):.4f}")
                except Exception as e:
                    print(f"Error calculating metrics: {e}")

    # Training complete
    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation loss: {best_loss:4f}')

    # Save final model
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'Saved final model checkpoint to: {checkpoint_path_last}')

    return model

def show_require_grad_layers(model):
    """Print layers that require gradients."""
    print('\nLayers requiring gradients:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f' {name}')

if __name__ == '__main__':
    # Set up directories
    CHECKPOINT_DIR_BEST = 'checkpoints/best'
    CHECKPOINT_DIR_LAST = 'checkpoints/last'
    CONFIG_DIR = 'config/handwriting'
    os.makedirs(CHECKPOINT_DIR_BEST, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR_LAST, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Load configuration
    args = config.get_config('train_decoding')

    # Training parameters
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    model_name = args['model_name']
    task_name = args['task_name']
    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']

    # Create checkpoint name
    if skip_step_one:
        save_name = f'handwriting_{task_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}'
    else:
        save_name = f'handwriting_{task_name}_2step_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}'

    output_checkpoint_name_best = f'/kaggle/working/checkpoints/decoding_raw/best/{save_name}.pt'
    output_checkpoint_name_last = f'/kaggle/working/checkpoints/decoding_raw/best/{save_name}.pt'

    # Set random seeds
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Set device
    device = torch.device(args['cuda'] if torch.cuda.is_available() else "cpu")
    print(f'[INFO] Using device: {device}')

    # Initialize tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # Get handwriting BCI dataset paths
    data_root = "/kaggle/input/handwriting-bci/handwritingBCIData/Datasets"
    session_paths = [
        os.path.join(data_root, session) for session in [
            "t5.2019.05.08", "t5.2019.11.25", "t5.2019.12.09", "t5.2019.12.11",
            "t5.2019.12.18", "t5.2019.12.20", "t5.2020.01.06", "t5.2020.01.08",
            "t5.2020.01.13", "t5.2020.01.15"
        ]
    ]
    # Create datasets
    train_set = data_raw_new_dataset.HandwritingBCIDataset(session_paths, 'train', tokenizer)
    dev_set = data_raw_new_dataset.HandwritingBCIDataset(session_paths, 'dev', tokenizer)
    test_set = data_raw_new_dataset.HandwritingBCIDataset(session_paths, 'test', tokenizer)

    print(f'[INFO] Train set size: {len(train_set)}')
    print(f'[INFO] Dev set size: {len(dev_set)}')
    print(f'[INFO] Test set size: {len(test_set)}')

    # Create dataloaders
    train_dataloader = DataLoader(train_set, batch_size=batch_size, 
                                shuffle=True, num_workers=0, collate_fn=data_raw_new_dataset.collate_fn)
    val_dataloader = DataLoader(dev_set, batch_size=1, 
                              shuffle=False, num_workers=0, collate_fn=data_raw_new_dataset.collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=1, 
                               shuffle=False, num_workers=0, collate_fn=data_raw_new_dataset.collate_fn)

    dataloaders = {
        'train': train_dataloader,
        'dev': val_dataloader,
        'test': test_dataloader
    }

    # Initialize model
    if model_name == 'BrainTranslator':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = model_decoding_raw_new_dataset.BrainTranslator(
            pretrained,
            in_feature=1024,
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=4096
        )
    model.to(device)

    # Save config
    withopen(f'/kaggle/working/config/decoding_raw/{save_name}.json', 'w') as f:
        json.dump(args, f, indent=4)

    if skip_step_one:
        # Load step 1 checkpoint if specified
        if load_step1_checkpoint:
            print(f'Loading step 1 checkpoint: {args["step1_checkpoint"]}')
            model.load_state_dict(torch.load(args["step1_checkpoint"]))
        else:
            print('Skipping step one, starting from scratch at step two')

        # Step 2 training
        model.freeze_pretrained_brain()
        optimizer_step2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=step2_lr, momentum=0.9)
        scheduler_step2 = lr_scheduler.CyclicLR(
            optimizer_step2,
            base_lr=0.0000005,
            max_lr=0.00005,
            mode="triangular2"
        )

        # Set up loss function
        criterion = nn.CrossEntropyLoss()

        print('\n=== Starting Step 2 training... ===')
        show_require_grad_layers(model)

        # Train model
        trained_model = train_model(
            dataloaders,
            device,
            model,
            criterion,
            optimizer_step2,
            scheduler_step2,
            num_epochs=num_epochs_step2,
            checkpoint_path_best=output_checkpoint_name_best,
            checkpoint_path_last=output_checkpoint_name_last,
            stepone=False
        )

    else:
        # Step 1 training
        if args.get('upload_first_run_step1', False):
            print(f'Loading previous step 1 checkpoint: {args["step1_checkpoint_first"]}')
            model.load_state_dict(torch.load(args["step1_checkpoint_first"]))

        # Set up optimizer and scheduler for step 1
        optimizer_step1 = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=step1_lr,
            momentum=0.9
        )

        scheduler_step1 = lr_scheduler.CyclicLR(
            optimizer_step1,
            base_lr=step1_lr,
            max_lr=5e-3,
            mode="triangular2"
        )

        # Set up loss function
        criterion = nn.MSELoss()
        
        # Freeze BART parameters
        model.freeze_pretrained_bart()

        print('\n=== Starting Step 1 training... ===')
        show_require_grad_layers(model)

        # Train model
        model = train_model(
            dataloaders,
            device,
            model,
            criterion,
            optimizer_step1,
            scheduler_step1,
            num_epochs=num_epochs_step1,
            checkpoint_path_best=output_checkpoint_name_best,
            checkpoint_path_last=output_checkpoint_name_last,
            stepone=True
        )

    # Clean up
    train_writer.flush()
    train_writer.close()
    val_writer.flush()
    val_writer.close()
    dev_writer.flush()
    dev_writer.close()

    print("\nTraining complete!")
