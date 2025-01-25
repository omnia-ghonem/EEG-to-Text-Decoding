import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_raw import ZuCo_dataset
from model_decoding_raw import BrainTranslator
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from bert_score import score
from config import get_config
from torch.nn.utils.rnn import pad_sequence
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def chatgpt_refinement(corrupted_text, api_key):
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4", max_tokens=256, openai_api_key=api_key)
    messages = [
        SystemMessage(content="As a text reconstructor, your task is to restore corrupted sentences to their original form while making minimum changes. You should adjust the spaces and punctuation marks as necessary. Do not introduce any additional information. If you are unable to reconstruct the text, respond with [False]."),
        HumanMessage(content=f"Reconstruct the following text: [{corrupted_text}].")
    ]
    output = llm(messages).content
    output = output.replace('[','').replace(']','')
    return corrupted_text if len(output)<10 and 'False' in output else output

def eval_model(dataloaders, device, tokenizer, criterion, model, api_key='1234', output_all_results_path='/kaggle/working/results_raw/temp.txt'):
    gpt = False
    print("Saving to:", output_all_results_path)
    model.eval()
    running_loss = 0.0
    target_tokens_list, target_string_list = [], []
    pred_tokens_list, pred_string_list = [], []
    refine_tokens_list, refine_string_list = [], []

    with open(output_all_results_path, 'w') as f:
        for batch in dataloaders['test']:
            _, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, word_contents, word_contents_attn, subject_batch = batch

            # Prepare batch inputs
            input_embeddings_batch = input_raw_embeddings.to(device).float()
            input_embeddings_lengths_batch = torch.stack([torch.tensor(a.clone().detach()) for a in input_raw_embeddings_lengths], 0).to(device)
            input_masks_batch = torch.stack(input_masks, 0).to(device)
            input_mask_invert_batch = torch.stack(input_mask_invert, 0).to(device)
            target_ids_batch = torch.stack(target_ids, 0).to(device)
            word_contents_batch = torch.stack(word_contents, 0).to(device)
            word_contents_attn_batch = torch.stack(word_contents_attn, 0).to(device)
            subject_batch = np.array(subject_batch)

            # Get target tokens and strings
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)

            # Write target info
            f.write(f'target string: {target_string}\n')
            f.write(f'target tokens: [{" ".join(target_tokens)}]\n')

            # Handle padding
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            # Forward pass
            seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, 
                                  target_ids_batch, input_embeddings_lengths_batch, word_contents_batch, 
                                  word_contents_attn_batch, False, subject_batch, device)

            # Calculate loss
            loss = criterion(seq2seqLMoutput.permute(0,2,1), target_ids_batch.long())
            running_loss += loss.item() * input_embeddings_batch.size()[0]

            # Get predictions
            logits = seq2seqLMoutput
            probs = logits[0].softmax(dim=1)
            predictions = torch.squeeze(probs.topk(1)[1])
            predicted_string = tokenizer.decode(predictions, skip_special_tokens=True)

            # Process predictions
            truncated_prediction = []
            for t in predictions.tolist():
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
            
            # Store predictions
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)

            # Write predictions
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'predicted tokens (truncated): [{" ".join(pred_tokens)}]\n')
            f.write('################################################\n\n')

            # GPT refinement if enabled
            if gpt:
                predicted_string_chatgpt = chatgpt_refinement(predicted_string, api_key).replace('\n','')
                f.write(f'refined string: {predicted_string_chatgpt}\n')
                refine_tokens = tokenizer.convert_ids_to_tokens(
                    tokenizer(predicted_string_chatgpt)['input_ids'], skip_special_tokens=True)
                refine_tokens_list.append(refine_tokens)
                refine_string_list.append(predicted_string_chatgpt)

    # Calculate metrics
    epoch_loss = running_loss / dataset_sizes['test_set']
    print(f'test loss: {epoch_loss:4f}\n')

    print("Predicted outputs")
    weights_list = [(1.0,), (0.5,0.5), (1./3.,1./3.,1./3.), (0.25,0.25,0.25,0.25)]
    for weight in weights_list:
        bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
        print(f'corpus BLEU-{len(list(weight))} score: {bleu_score}')

    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    print('\nROUGE scores:', rouge_scores)

    P, R, F1 = score(pred_string_list, target_string_list, lang='en', device="cuda:0", 
                    model_type="bert-large-uncased")
    print(f"\nBERT scores:\nP: {np.mean(P):.4f}\nR: {np.mean(R):.4f}\nF1: {np.mean(F1):.4f}")

    if gpt:
        print("\nRefined outputs with GPT4")
        for weight in weights_list:
            bleu_score = corpus_bleu(target_tokens_list, refine_tokens_list, weights=weight)
            print(f'corpus BLEU-{len(list(weight))} score: {bleu_score}')
        
        rouge_scores = rouge.get_scores(refine_string_list, target_string_list, avg=True)
        print('\nROUGE scores:', rouge_scores)
        
        P, R, F1 = score(refine_string_list, target_string_list, lang='en', device="cuda:0", 
                        model_type="bert-large-uncased")
        print(f"\nBERT scores:\nP: {np.mean(P):.4f}\nR: {np.mean(R):.4f}\nF1: {np.mean(F1):.4f}")

if __name__ == '__main__':
    args = get_config('eval_decoding')
    training_config = json.load(open(args['config_path']))
    
    # Setup configuration
    api_key = args['api_key']
    subject_choice = training_config['subjects']
    eeg_type_choice = training_config['eeg_type']
    bands_choice = training_config['eeg_bands']
    task_name = training_config['task_name']
    model_name = training_config['model_name']
    output_all_results_path = f'/kaggle/working/results_raw/{task_name}-{model_name}-all_decoding_results.txt'

    # Set random seeds
    torch.manual_seed(312)
    torch.cuda.manual_seed_all(312)
    np.random.seed(312)

    # Setup device
    device = torch.device(args['cuda'] if torch.cuda.is_available() else "cpu")
    print(f'[INFO]using device {args["cuda"]}')

    # Load datasets
    whole_dataset_dicts = []
    dataset_paths = {
        'task1': '/kaggle/input/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle',
        'task2': '/kaggle/input/dataset2/task2-NR-dataset_wRaw.pickle',
        'taskNRv2': '/kaggle/input/dataset3/task2-NR-2.0-dataset_wRaw.pickle'
    }
    
    for task, path in dataset_paths.items():
        if task in task_name:
            with open(path, 'rb') as handle:
                whole_dataset_dicts.append(pickle.load(handle))

    # Initialize tokenizer and dataset
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice,
                           eeg_type=eeg_type_choice, bands=bands_choice, 
                           setting='unique_sent', raweeg=True)
    dataset_sizes = {"test_set": len(test_set)}
    print(f'[INFO]test_set size: {len(test_set)}')

    # Setup dataloader
    def pad_and_sort_batch(batch):
        input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, \
        sentiment_labels, sent_level_EEG, input_raw_embeddings, word_contents, word_contents_attn, subject = zip(*batch)

        raw_eeg = []
        input_raw_embeddings_lenghts = []
        for sentence in input_raw_embeddings:
            input_raw_embeddings_lenghts.append(torch.Tensor([a.size(0) for a in sentence]))
            raw_eeg.append(pad_sequence(sentence, batch_first=True, padding_value=0).permute(1, 0, 2))

        input_raw_embeddings = pad_sequence(raw_eeg, batch_first=True, padding_value=0).permute(0, 2, 1, 3)

        return input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, \
               sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lenghts, \
               word_contents, word_contents_attn, subject

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, 
                               num_workers=4, collate_fn=pad_and_sort_batch)
    dataloaders = {'test': test_dataloader}

    # Initialize model
    pretrained_t5 = T5ForConditionalGeneration.from_pretrained('t5-large')
    model = BrainTranslator(pretrained_t5, in_feature=1024, decoder_embedding_size=1024,
                           additional_encoder_nhead=8, additional_encoder_dim_feedforward=4096)
    
    # Load checkpoint and move to device
    model.load_state_dict(torch.load(args['checkpoint_path']))
    model.to(device)
    
    # Set loss criterion and evaluate
    criterion = nn.CrossEntropyLoss()
    eval_model(dataloaders, device, tokenizer, criterion, model, api_key, output_all_results_path)
