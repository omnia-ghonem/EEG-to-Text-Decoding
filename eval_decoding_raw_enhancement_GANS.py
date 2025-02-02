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

def deepseek_refinement(corrupted_text, model, tokenizer):
    """Refines corrupted text using DeepSeek model"""
    messages = [
        {"role": "user", "content": f"As a text reconstructor, your task is to restore the following corrupted sentence to its original form while making minimum changes. Adjust spaces and punctuation marks as necessary. Do not introduce any additional information. If you are unable to reconstruct the text, respond with [False]. Here is the text to reconstruct: [{corrupted_text}]"}
    ]
    
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(
        input_tensor.to(model.device), 
        max_new_tokens=100,
        temperature=0.2
    )
    
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    result = result.strip()
    
    if len(result) < 10 and 'False' in result:
        return corrupted_text
        
    return result

def eval_model(dataloaders, device, tokenizer, criterion, model,api_key = '1234', output_all_results_path = '/kaggle/working/results_raw/temp.txt' ):
    # Initialize DeepSeek model
    deepseek_name = "deepseek-ai/DeepSeek-V2-Chat"
    deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_name, trust_remote_code=True)
    max_memory = {i: "75GB" for i in range(8)}
    deepseek_model = AutoModelForCausalLM.from_pretrained(
        deepseek_name, 
        trust_remote_code=True, 
        device_map="sequential",
        torch_dtype=torch.bfloat16, 
        max_memory=max_memory,
        attn_implementation="eager"
    )
    deepseek_model.generation_config = GenerationConfig.from_pretrained(deepseek_name)
    deepseek_model.generation_config.pad_token_id = deepseek_model.generation_config.eos_token_id

    print("Saving to: ", output_all_results_path)
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data
    sample_count = 0
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    refine_tokens_list = []
    refine_string_list = []

    with open(output_all_results_path, 'w') as f:
        for _, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, word_contents, word_contents_attn, subject_batch in dataloaders['test']:
            # Load batch
            input_embeddings_batch = input_raw_embeddings.to(device).float()
            input_embeddings_lengths_batch = torch.stack([torch.tensor(
                a.clone().detach()) for a in input_raw_embeddings_lengths], 0).to(device)
            input_masks_batch = torch.stack(input_masks, 0).to(device)
            input_mask_invert_batch = torch.stack(input_mask_invert, 0).to(device)
            target_ids_batch = torch.stack(target_ids, 0).to(device)
            word_contents_batch = torch.stack(word_contents, 0).to(device)
            word_contents_attn_batch = torch.stack(word_contents_attn, 0).to(device)
            
            subject_batch = np.array(subject_batch)
                        
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)

            f.write(f'target string: {target_string}\n')
            target_tokens_string = "["
            for el in target_tokens:
                target_tokens_string = target_tokens_string + str(el) + " "
            target_tokens_string += "]"
            f.write(f'target tokens: {target_tokens_string}\n')

            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            # Replace padding ids in target_ids with -100
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            # Forward pass
            seq2seqLMoutput = model(
                input_embeddings_batch, input_masks_batch, input_mask_invert_batch, 
                target_ids_batch, input_embeddings_lengths_batch, word_contents_batch, 
                word_contents_attn_batch, False, subject_batch, device
            )

            # Calculate loss
            loss = criterion(seq2seqLMoutput.permute(0,2,1), target_ids_batch.long())

            # Get predicted tokens
            logits = seq2seqLMoutput
            probs = logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>','')
            f.write(f'predicted string: {predicted_string}\n')
            
            # Convert to int list and process predictions
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

            # DeepSeek refinement
            predicted_string_refined = deepseek_refinement(predicted_string, deepseek_model, deepseek_tokenizer)
            f.write(f'refined string: {predicted_string_refined}\n')
            refine_tokens_list.append(tokenizer.convert_ids_to_tokens(
                tokenizer(predicted_string_refined)['input_ids'], 
                skip_special_tokens=True
            ))
            refine_string_list.append(predicted_string_refined)

            pred_tokens_string = "["
            for el in pred_tokens:
                pred_tokens_string = pred_tokens_string + str(el) + " "
            pred_tokens_string += "]"
            f.write(f'predicted tokens (truncated): {pred_tokens_string}\n')
            f.write(f'################################################\n\n\n')

            sample_count += 1
            running_loss += loss.item() * input_embeddings_batch.size()[0]

    epoch_loss = running_loss / dataset_sizes['test_set']
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
    print(rouge_scores)
    
    # Calculate BERTScore
    P, R, F1 = score(pred_string_list, target_string_list, lang='en', device="cuda:0", model_type="bert-large-uncased")
    print(f"bert_score P: {np.mean(np.array(P))}")
    print(f"bert_score R: {np.mean(np.array(R))}")
    print(f"bert_score F1: {np.mean(np.array(F1))}")
    print("*************************************")

    print("\nRefined outputs with DeepSeek")
    # Calculate corpus BLEU score for refined outputs
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, refine_tokens_list, weights=weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
    
    # Calculate ROUGE score for refined outputs
    rouge_scores = rouge.get_scores(refine_string_list, target_string_list, avg=True)
    print(rouge_scores)
    
    # Calculate BERTScore for refined outputs
    P, R, F1 = score(refine_string_list, target_string_list, lang='en', device="cuda:0", model_type="bert-large-uncased")
    print(f"bert_score P: {np.mean(np.array(P))}")
    print(f"bert_score R: {np.mean(np.array(R))}")
    print(f"bert_score F1: {np.mean(np.array(F1))}")
    print("*************************************")
    print("*************************************")


if __name__ == '__main__': 
    ''' get args'''
    args = get_config('eval_decoding')

    ''' load training config'''
    training_config = json.load(open(args['config_path']))
    api_key = args['api_key']  # Fetch the API key from args
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
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')


    ''' set up dataloader '''
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
    print()
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, raweeg=True)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # Allows to pad and get real size of eeg vectors
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


    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4,collate_fn=pad_and_sort_batch)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    if model_name == 'BrainTranslator':
        model = BrainTranslator(pretrained_bart, in_feature=1024, decoder_embedding_size=1024,
                                        additional_encoder_nhead=8, additional_encoder_dim_feedforward=4096)
        
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    eval_model(dataloaders, device, tokenizer, criterion, model, api_key, output_all_results_path = output_all_results_path)
