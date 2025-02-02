import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob

from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from data_raw import ZuCo_dataset
from model_decoding_raw import BrainTranslator
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from config import get_config

from torch.nn.utils.rnn import pad_sequence

# Deepseek refinement function
def deepseek_refinement(corrupted_text, device):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct").to(device)
    
    prompt = f"""As a text reconstructor, your task is to restore the following corrupted sentence to its original form while making minimum changes. Adjust spaces and punctuation marks as necessary. Do not introduce any additional information. If you are unable to reconstruct the text, respond with [False].

Corrupted text: [{corrupted_text}]

Please provide only the reconstructed text without any additional commentary."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=256, temperature=0.2, do_sample=False)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the actual response (after the prompt)
    response = output.split("Please provide only the reconstructed text without any additional commentary.")[-1].strip()
    response = response.replace('[', '').replace(']', '')
    
    if len(response) < 10 and 'False' in response:
        return corrupted_text
    
    return response

def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path='/kaggle/working/results_raw/temp.txt'):
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
    
    # Buffer for collecting results before writing
    results_buffer = []
    BUFFER_SIZE = 10  # Number of results to collect before writing to file
    
    def write_buffer_to_file(buffer, f):
        if buffer:
            f.write(''.join(buffer))
            f.flush()  # Explicitly flush after writing
            return []  # Return empty buffer
        return buffer

    with open(output_all_results_path, 'w', buffering=1) as f:  # Use line buffering
        for _, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, word_contents, word_contents_attn, subject_batch in dataloaders['test']:
            # Load batch data
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

            # Collect results in buffer
            results_buffer.append(f'target string: {target_string}\n')
            target_tokens_string = "[" + " ".join(str(el) for el in target_tokens) + "]"
            results_buffer.append(f'target tokens: {target_tokens_string}\n')

            # Add to list for later BLEU metric calculation
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            # Replace padding ids in target_ids with -100
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            # Forward pass
            seq2seqLMoutput = model(
                input_embeddings_batch, input_masks_batch, input_mask_invert_batch, 
                target_ids_batch, input_embeddings_lengths_batch, word_contents_batch, 
                word_contents_attn_batch, False, subject_batch, device)

            # Calculate loss
            loss = criterion(seq2seqLMoutput.permute(0,2,1), target_ids_batch.long())

            # Get predicted tokens
            logits = seq2seqLMoutput
            probs = logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
            results_buffer.append(f'predicted string: {predicted_string}\n')
            
            # Convert to int list and truncate at EOS
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

            # Deepseek refinement
            predicted_string_refined = deepseek_refinement(predicted_string, device).replace('\n', '')
            results_buffer.append(f'refined string: {predicted_string_refined}\n')
            refine_tokens_list.append(tokenizer.convert_ids_to_tokens(
                tokenizer(predicted_string_refined)['input_ids'], skip_special_tokens=True))
            refine_string_list.append(predicted_string_refined)

            pred_tokens_string = "[" + " ".join(str(el) for el in pred_tokens) + "]"
            results_buffer.append(f'predicted tokens (truncated): {pred_tokens_string}\n')
            results_buffer.append(f'################################################\n\n\n')
            
            # Write buffer to file when it reaches the threshold
            if len(results_buffer) >= BUFFER_SIZE:
                results_buffer = write_buffer_to_file(results_buffer, f)

            sample_count += 1
            running_loss += loss.item() * input_embeddings_batch.size()[0]
            
            # Print progress to show the script is still running
            if sample_count % 5 == 0:
                print(f'Processed {sample_count} samples...')

    epoch_loss = running_loss / dataset_sizes['test_set']
    # Write any remaining results in buffer
    write_buffer_to_file(results_buffer, f)
    print('test loss: {:4f}'.format(epoch_loss))

    print("\nPredicted outputs")
    # Calculate corpus BLEU score
    weights_list = [(1.0,), (0.5,0.5), (1./3.,1./3.,1./3.), (0.25,0.25,0.25,0.25)]
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
    
    # Calculate ROUGE score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    print("\nROUGE scores:", rouge_scores)
    
    # Calculate BERTScore
    from bert_score import score
    P, R, F1 = score(pred_string_list, target_string_list, lang='en', device=device, model_type="bert-large-uncased")
    print(f"\nBERT Score metrics:")
    print(f"P: {np.mean(np.array(P))}")
    print(f"R: {np.mean(np.array(R))}")
    print(f"F1: {np.mean(np.array(F1))}")
    print("*************************************")

    print("\nRefined outputs with Deepseek")
    # Calculate corpus BLEU score for refined outputs
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, refine_tokens_list, weights=weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
    
    # Calculate ROUGE score for refined outputs
    rouge_scores = rouge.get_scores(refine_string_list, target_string_list, avg=True)
    print("\nROUGE scores:", rouge_scores)
    
    # Calculate BERTScore for refined outputs
    P, R, F1 = score(refine_string_list, target_string_list, lang='en', device=device, model_type="bert-large-uncased")
    print(f"\nBERT Score metrics:")
    print(f"P: {np.mean(np.array(P))}")
    print(f"R: {np.mean(np.array(R))}")
    print(f"F1: {np.mean(np.array(F1))}")
    print("*************************************")

if __name__ == '__main__':
    args = get_config('eval_decoding')
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

    # Set up device
    device = torch.device(args['cuda'] if torch.cuda.is_available() else "cpu")
    print(f'[INFO]using device {device}')

    # Set up dataloader
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/kaggle/input/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '/kaggle/input/dataset2/task2-NR-dataset_wRaw.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = '/kaggle/input/dataset3/task2-NR-2.0-dataset_wRaw.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice, 
                           eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, raweeg=True)
    dataset_sizes = {"test_set": len(test_set)}
    print('[INFO]test_set size: ', len(test_set))

    def pad_and_sort_batch(data_loader_batch):
        input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, word_contents, word_contents_attn, subject = tuple(zip(*data_loader_batch))

        raw_eeg = []
        input_raw_embeddings_lenghts = []
        for sentence in input_raw_embeddings:
            input_raw_embeddings_lenghts.append(torch.Tensor([a.size(0) for a in sentence]))
            raw_eeg.append(pad_sequence(sentence, batch_first=True, padding_value=0).permute(1, 0, 2))

        input_raw_embeddings = pad_sequence(raw_eeg, batch_first=True, padding_value=0).permute(0, 2, 1, 3)

        return input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lenghts, word_contents, word_contents_attn, subject

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=pad_and_sort_batch)
    dataloaders = {'test': test_dataloader}

    # Set up model
    checkpoint_path = args['checkpoint_path']
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    if model_name == 'BrainTranslator':
        model = BrainTranslator(pretrained_bart, in_feature=1024, decoder_embedding_size=1024,
                               additional_encoder_nhead=8, additional_encoder_dim_feedforward=4096)
        
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path=output_all_results_path)
