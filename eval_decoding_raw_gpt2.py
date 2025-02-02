import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_raw_gpt2 import ZuCo_dataset
from model_decoding_raw_gpt2 import BrainTranslator
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from config import get_config
from torch.nn.utils.rnn import pad_sequence

def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path='/kaggle/working/results_raw/temp.txt'):
    print("Saving to:", output_all_results_path)
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    
    # Iterate over data.
    sample_count = 0
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []

    with open(output_all_results_path, 'w') as f:
        for _, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, word_contents, word_contents_attn, subject_batch in dataloaders['test']:
            
            # load in batch
            input_embeddings_batch = input_raw_embeddings.to(device).float()
            input_embeddings_lengths_batch = torch.stack([torch.tensor(
                a.clone().detach()) for a in input_raw_embeddings_lengths], 0).to(device)
            input_masks_batch = torch.stack(input_masks, 0).to(device)
            input_mask_invert_batch = torch.stack(
                input_mask_invert, 0).to(device)
            target_ids_batch = torch.stack(target_ids, 0).to(device)
            word_contents_batch = torch.stack(word_contents, 0).to(device)
            word_contents_attn_batch = torch.stack(
                word_contents_attn, 0).to(device)
            
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
            
            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            # forward
            logits = model(
                input_embeddings_batch, input_masks_batch, input_mask_invert_batch, 
                target_ids_batch, input_embeddings_lengths_batch, word_contents_batch, 
                word_contents_attn_batch, False, subject_batch, device)

            """calculate loss"""
            loss = criterion(logits.permute(0,2,1), target_ids_batch.long())

            # get predicted tokens
            probs = logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('<|endoftext|>')[0]
            
            f.write(f'predicted string: {predicted_string}\n')
            
            # convert to int list
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

    print("Predicted outputs")
    """ calculate corpus bleu score """
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
    print()
    
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    print(rouge_scores)
    print()
    
    """ calculate bertscore"""
    from bert_score import score
    P, R, F1 = score(pred_string_list, target_string_list, lang='en', device="cuda:0", model_type="bert-large-uncased")
    print(f"bert_score P: {np.mean(np.array(P))}")
    print(f"bert_score R: {np.mean(np.array(R))}")
    print(f"bert_score F1: {np.mean(np.array(F1))}")
    print("*************************************")

if __name__ == '__main__': 
    ''' get args'''
    args = get_config('eval_decoding')

    ''' load training config'''
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
    
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
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
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, 
                           subject=subject_choice, eeg_type=eeg_type_choice, 
                           bands=bands_choice, setting=dataset_setting, raweeg=True)

    dataset_sizes = {"test_set": len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    def pad_and_sort_batch(data_loader_batch):
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

        return input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lenghts, word_contents, word_contents_attn, subject

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, 
                               num_workers=4, collate_fn=pad_and_sort_batch)
    dataloaders = {'test': test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    pretrained_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
    
    if model_name == 'BrainTranslator':
        model = BrainTranslator(pretrained_gpt2, in_feature=1024, 
                              decoder_embedding_size=768,
                              additional_encoder_nhead=8, 
                              additional_encoder_dim_feedforward=4096)
        
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    eval_model(dataloaders, device, tokenizer, criterion, model, 
              output_all_results_path=output_all_results_path)
