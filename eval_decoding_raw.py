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

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_actual_api_key_here"  # Replace with your OpenAI API key

# Check if the API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please set it before running the code.")

# LLMs: Get predictions from ChatGPT
def chatgpt_refinement(corrupted_text):
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4", max_tokens=256)

    messages = [
        SystemMessage(content="As a text reconstructor, your task is to restore corrupted sentences to their original form while making minimum changes. You should adjust the spaces and punctuation marks as necessary. Do not introduce any additional information. If you are unable to reconstruct the text, respond with [False]."),
        HumanMessage(content=f"Reconstruct the following text: [{corrupted_text}].")
    ]

    output = llm(messages).content
    output = output.replace('[', '').replace(']', '')
    
    if len(output) < 10 and 'False' in output:
        return corrupted_text
    
    return output

def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path='/kaggle/working/results_raw/temp.txt'):
    gpt = True

    print("Saving to: ", output_all_results_path)
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    sample_count = 0
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    refine_tokens_list = []
    refine_string_list = []

    with open(output_all_results_path, 'w') as f:
        for _, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, word_contents, word_contents_attn, subject_batch in dataloaders['test']:

            # Load in batch
            input_embeddings_batch = input_raw_embeddings.to(device).float()
            input_embeddings_lengths_batch = torch.stack([torch.tensor(a.clone().detach()) for a in input_raw_embeddings_lengths], 0).to(device)
            input_masks_batch = torch.stack(input_masks, 0).to(device)
            input_mask_invert_batch = torch.stack(input_mask_invert, 0).to(device)
            target_ids_batch = torch.stack(target_ids, 0).to(device)
            word_contents_batch = torch.stack(word_contents, 0).to(device)
            word_contents_attn_batch = torch.stack(word_contents_attn, 0).to(device)
            
            subject_batch = np.array(subject_batch)
                        
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)

            f.write(f'target string: {target_string}\n')
            target_tokens_string = "[" + " ".join(map(str, target_tokens)) + "]"
            f.write(f'target tokens: {target_tokens_string}\n')

            # Add to list for later calculate BLEU metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            # Replace padding ids in target_ids with -100
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            # Forward
            seq2seqLMoutput = model(
                input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch, input_embeddings_lengths_batch, word_contents_batch, word_contents_attn_batch, False, subject_batch, device)

            # Calculate loss
            loss = criterion(seq2seqLMoutput.permute(0, 2, 1), target_ids_batch.long())

            # Get predicted tokens
            logits = seq2seqLMoutput
            probs = logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
            f.write(f'predicted string: {predicted_string}\n')
            
            # Convert to int list
            predictions = predictions.tolist()
            truncated_prediction = [t for t in predictions if t != tokenizer.eos_token_id]
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)

            # ChatGPT refinement and tokenizer decode
            if gpt:
                predicted_string_chatgpt = chatgpt_refinement(predicted_string).replace('\n', '')
                f.write(f'refined string: {predicted_string_chatgpt}\n')
                refine_tokens_list.append(tokenizer.convert_ids_to_tokens(tokenizer(predicted_string_chatgpt)['input_ids'], skip_special_tokens=True))
                refine_string_list.append(predicted_string_chatgpt)

            pred_tokens_string = "[" + " ".join(map(str, pred_tokens)) + "]"
            f.write(f'predicted tokens (truncated): {pred_tokens_string}\n')
            f.write(f'################################################\n\n\n')

            sample_count += 1
            # Statistics
            running_loss += loss.item() * input_embeddings_batch.size()[0]  # Batch loss

    epoch_loss = running_loss / dataset_sizes['test_set']
    print('test loss: {:4f}'.format(epoch_loss))

    print("Predicted outputs")
    # Calculate corpus BLEU score
    weights_list = [(1.0,), (0.5, 0.5), (1./3., 1./3., 1./3.), (0.25, 0.25, 0.25, 0.25)]
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
    print()
    # Calculate ROUGE score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    print(rouge_scores)
    print()
    # Calculate BERTScore
    from bert_score import score
    P, R, F1 = score(pred_string_list, target_string_list, lang='en', device="cuda:0", model_type="bert-large-uncased")
    print(f"bert_score P: {np.mean(np.array(P))}")
    print(f"bert_score R: {np.mean(np.array(R))}")
    print(f"bert_score F1: {np.mean(np.array(F1))}")
    print("*************************************")

    if gpt:
        print()
        print("Refined outputs with GPT4")
        # Calculate corpus BLEU score
        for weight in weights_list:
            corpus_bleu_score = corpus_bleu(target_tokens_list, refine_tokens_list, weights=weight)
            print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
        print()
        # Calculate ROUGE score
        rouge_scores = rouge.get_scores(refine_string_list, target_string_list, avg=True)
        print(rouge_scores)
        print()
        # Calculate BERTScore
        P, R, F1 = score(refine_string_list, target_string_list, lang='en', device="cuda:0", model_type="bert-large-uncased")
        print(f"bert_score P: {np.mean(np.array(P))}")
        print(f"bert_score R: {np.mean(np.array(R))}")
        print(f"bert_score F1: {np.mean(np.array(F1))}")
        print("*************************************")
        print("*************************************")

if __name__ == '__main__': 
    # Get args
    args = get_config('eval_decoding')

    # Load training config
    training_config = json.load(open(args['config_path']))
    
    batch_size = 1
    
    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')

    dataset_setting = 'unique_sent'
    task_name = training_config['task_name']    
    model_name = training_config['model_name']

    output_all_results_path = f'/kaggle/working/results_raw/{task_name}-{model_name}-all_decoding_results.txt'
    # Set random seeds
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Set up tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Prepare dataset
    dataset = ZuCo_dataset(subjects=subject_choice, eeg_type=eeg_type_choice, dataset_setting=dataset_setting, bands=bands_choice)
    
    dataloaders = {
        'test': DataLoader(dataset['test_set'], batch_size=batch_size, shuffle=False, num_workers=1)
    }

    dataset_sizes = {x: len(dataset[x]) for x in ['test_set']}

    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path)
