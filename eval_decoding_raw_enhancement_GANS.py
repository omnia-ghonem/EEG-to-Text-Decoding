import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BartForConditionalGeneration
from data_raw import ZuCo_dataset
from model_decoding_raw import BrainTranslator
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from config import get_config
from bert_score import score
import sys
from torch.nn.utils.rnn import pad_sequence


from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/data_raw.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/model_decoding_raw_enhancement_GANS.py')
sys.path.insert(1, '/kaggle/working/EEG-to-Text-Decoding/config.py')

import data_raw
import config
import model_decoding_raw_enhancement_GANS

# LLMs: Get predictions from ChatGPT
def chatgpt_refinement(corrupted_text, api_key):
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4", max_tokens=256, openai_api_key=api_key)
    messages = [
        SystemMessage(content="As a text reconstructor, your task is to restore corrupted sentences to their original form while making minimum changes. You should adjust the spaces and punctuation marks as necessary. Do not introduce any additional information. If you are unable to reconstruct the text, respond with [False]."),
        HumanMessage(content=f"Reconstruct the following text: [{corrupted_text}].")
    ]

    output = llm(messages).content
    output = output.replace('[','').replace(']','')
    
    if len(output)<10 and 'False' in output:
        return corrupted_text
    
    return output


def eval_model(dataloaders, device, tokenizer, criterion, model, api_key='1234', output_all_results_path='/kaggle/working/results_raw/temp.txt'):
    model.eval()
    running_loss = 0.0
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    
    with open(output_all_results_path, 'w') as f:
        for batch_data in dataloaders['test']:
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

            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)

            f.write(f'Target: {target_string}\n')
            
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            with torch.no_grad():
                outputs = model(input_embeddings_batch, input_masks_batch, 
                              input_mask_invert_batch, target_ids_batch,
                              input_embeddings_lengths_batch, word_contents_batch, 
                              word_contents_attn_batch, False, subject_batch, device)

                logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                loss = criterion(logits.permute(0, 2, 1), target_ids_batch.long())
                running_loss += loss.item() * input_embeddings_batch.size()[0]

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

                f.write(f'Prediction: {predicted_string}\n')
                f.write('-' * 50 + '\n\n')

    epoch_loss = running_loss / len(dataloaders['test'].dataset)
    print(f'Test Loss: {epoch_loss:.4f}')

    # Calculate BLEU scores
    for n in range(1, 5):
        weights = tuple([1.0/n] * n)
        bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weights)
        print(f'BLEU-{n}: {bleu_score:.4f}')

    # Calculate ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    print("\nROUGE scores:", rouge_scores)

    # Calculate BERTScore
    P, R, F1 = score(pred_string_list, target_string_list, lang='en', 
                     device=device, model_type="bert-large-uncased")
    print(f"\nBERTScore:\nP: {P.mean():.4f}\nR: {R.mean():.4f}\nF1: {F1.mean():.4f}")

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

if __name__ == '__main__':
    args = config.get_config('eval_decoding')
    training_config = json.load(open(args['config_path']))
    
    # Setup
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device(args['cuda'] if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Load datasets
    whole_dataset_dicts = []
    if 'task1' in training_config['task_name']:
        with open('/kaggle/input/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in training_config['task_name']:
        with open('/kaggle/input/dataset2/task2-NR-dataset_wRaw.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in training_config['task_name']:
        with open('/kaggle/input/dataset3/task2-NR-2.0-dataset_wRaw.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # Create test dataset
    test_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer,
                                    subject=training_config['subjects'],
                                    eeg_type=training_config['eeg_type'],
                                    bands=training_config['eeg_bands'],
                                    setting='unique_sent', raweeg=True)
    print(f'Test set size: {len(test_set)}')

    dataloaders = {
        'test': DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    }

    # Initialize model
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = model_decoding_raw_enhancement_GANS.BrainTranslator(
        pretrained_bart,
        in_feature=1024,
        decoder_embedding_size=1024,
        additional_encoder_nhead=8,
        additional_encoder_dim_feedforward=4096
    )

    print('Loading checkpoint...')
    model.load_state_dict(torch.load(args['checkpoint_path']))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    output_path = f'/kaggle/working/results_raw/{training_config["task_name"]}-{training_config["model_name"]}-results.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f'Saving results to: {output_path}')
    
    eval_model(dataloaders, device, tokenizer, criterion, model, args['api_key'], output_path)
