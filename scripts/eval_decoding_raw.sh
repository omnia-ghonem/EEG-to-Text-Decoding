API_KEY="1234"
python /kaggle/working/EEG-to-Text-Decoding/eval_decoding_raw.py \
    --checkpoint_path /kaggle/input/train-step-2-second-time/checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b20_14_11_5e-05_5e-05_unique_sent.pt \
    --config_path /kaggle/input/train-step-2-second-time/config/decoding_raw/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b20_14_11_5e-05_5e-05_unique_sent.json \
    --api_key $API_KEY \
    -cuda cuda:0

