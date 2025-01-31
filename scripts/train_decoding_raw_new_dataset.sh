
python /kaggle/working/EEG-to-Text-Decoding/train_decoding_raw_new_dataset.py\
    --model_name BrainTranslator \
    --task_name task1_task2_taskNRv2 \
    --two_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --first_run \
    --num_epoch_step1 10 \
    --num_epoch_step2 10 \
    -lr1 0.00005 \
    -lr2 0.00005 \
    -b 20\
    -s /kaggle/working/checkpoints/decoding_raw \
    -cuda cuda:0
