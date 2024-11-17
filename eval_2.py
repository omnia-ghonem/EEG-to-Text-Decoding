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

from torch.nn.utils.rnn import pad_sequence


from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


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


def eval_model(dataloaders, device, tokenizer, model, api_key, output_all_results_path='/kaggle/working/results_raw/temp.txt'):
    # Ensure the model is in evaluation mode
    model.eval()
    print("Saving predictions to:", output_all_results_path)
    gpt=False
    # Open the output file for writing predictions
    with open(output_all_results_path, 'w') as f:
        for _, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, word_contents, word_contents_attn, subject_batch in dataloaders['test']:
            
            # Prepare input embeddings for the model
            input_embeddings_batch = input_raw_embeddings.to(device).float()
            input_embeddings_lengths_batch = torch.stack([torch.tensor(a.clone().detach()) for a in input_raw_embeddings_lengths], 0).to(device)
            input_masks_batch = torch.stack(input_masks, 0).to(device)
            input_mask_invert_batch = torch.stack(input_mask_invert, 0).to(device)
            word_contents_batch = torch.stack(word_contents, 0).to(device)
            word_contents_attn_batch = torch.stack(word_contents_attn, 0).to(device)
            
            subject_batch = np.array(subject_batch)
            
            # Forward pass to predict tokens
            seq2seqLMoutput = model(
                input_embeddings_batch, input_masks_batch, input_mask_invert_batch, None, input_embeddings_lengths_batch, word_contents_batch, word_contents_attn_batch, False, subject_batch, device
            )
            
            logits = seq2seqLMoutput
            probs = logits[0].softmax(dim=1)
            _, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
                        # Write predictions to the output file
            f.write(f'Predicted string (BART): {predicted_string}\n')
            f.write('################################################\n\n\n')
            print(f"Predicted (BART): {predicted_string}")

            if gpt:
            # GPT-4 refinement
                predicted_string_chatgpt = chatgpt_refinement(predicted_string, api_key).replace('\n', '')
                f.write(f'Predicted string (GPT-4): {predicted_string_chatgpt}\n')
                print(f"Predicted (GPT-4): {predicted_string_chatgpt}")




if __name__ == '__main__':
    ''' Get arguments '''
    args = get_config('eval_decoding')

    ''' Load training config '''
    training_config = json.load(open(args['config_path']))
    api_key = args['api_key']
    batch_size = 1

    subject_choice = training_config['subjects']
    print(f'[INFO] Subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO] EEG type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO] Using bands: {bands_choice}')

    dataset_setting = 'unique_sent'
    task_name = training_config['task_name']
    model_name = training_config['model_name']

    output_all_results_path = f'/kaggle/working/results_raw/{task_name}-{model_name}-all_decoding_results.txt'

    ''' Set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' Set up device '''
    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'[INFO] Using device {dev}')

    ''' Set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/kaggle/input/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # Test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, raweeg=True)

    dataset_sizes = {"test_set": len(test_set)}
    print('[INFO] Test set size:', len(test_set))

    def pad_and_sort_batch(data_loader_batch):
        """Pad and sort batch data."""
        input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, word_contents, word_contents_attn, subject = tuple(
            zip(*data_loader_batch))

        raw_eeg = []
        input_raw_embeddings_lengths = []
        for sentence in input_raw_embeddings:
            input_raw_embeddings_lengths.append(
                torch.Tensor([a.size(0) for a in sentence]))
            raw_eeg.append(pad_sequence(
                sentence, batch_first=True, padding_value=0).permute(1, 0, 2))

        input_raw_embeddings = pad_sequence(
            raw_eeg, batch_first=True, padding_value=0).permute(0, 2, 1, 3)

        return input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, word_contents, word_contents_attn, subject

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=pad_and_sort_batch)

    dataloaders = {'test': test_dataloader}

    ''' Set up model '''
    checkpoint_path = args['checkpoint_path']
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    if model_name == 'BrainTranslator':
        model = BrainTranslator(pretrained_bart, in_feature=1024, decoder_embedding_size=1024,
                                additional_encoder_nhead=8, additional_encoder_dim_feedforward=4096)

    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    ''' Evaluate '''
    eval_model(dataloaders, device, tokenizer, model, api_key, output_all_results_path=output_all_results_path)
