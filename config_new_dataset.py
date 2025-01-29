import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(case):
    if case == 'train_decoding': 
        # args config for training Brain-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for training Brain-To-Text decoder')
        
        # Model and task configuration
        parser.add_argument('-m', '--model_name', help='choose from {BrainTranslator}', 
                          default="BrainTranslator", required=True)
        parser.add_argument('-t', '--task_name', help='task name', 
                          default="handwriting", required=True)
        
        # Training strategy
        parser.add_argument('-1step', '--one_step', dest='skip_step_one', action='store_true')
        parser.add_argument('-2step', '--two_step', dest='skip_step_one', action='store_false')
        
        parser.add_argument('-pre', '--pretrained', dest='use_random_init', action='store_false')
        parser.add_argument('-rand', '--rand_init', dest='use_random_init', action='store_true')
        
        parser.add_argument('-load1', '--load_step1_checkpoint', dest='load_step1_checkpoint', 
                          action='store_true')
        parser.add_argument('-no-load1', '--not_load_step1_checkpoint', dest='load_step1_checkpoint', 
                          action='store_false')
        
        parser.add_argument('-1run', '--first_run', dest='upload_first_run_step1', action='store_false')
        parser.add_argument('-2run', '--not_first_run', dest='upload_first_run_step1', action='store_true')
        
        # Training hyperparameters
        parser.add_argument('-ne1', '--num_epoch_step1', type=int, help='num_epoch_step1', 
                          default=20, required=True)
        parser.add_argument('-ne2', '--num_epoch_step2', type=int, help='num_epoch_step2', 
                          default=30, required=True)
        parser.add_argument('-lr1', '--learning_rate_step1', type=float, help='learning_rate_step1', 
                          default=0.00005, required=True)
        parser.add_argument('-lr2', '--learning_rate_step2', type=float, help='learning_rate_step2', 
                          default=0.0000005, required=True)
        parser.add_argument('-b', '--batch_size', type=int, help='batch_size', 
                          default=32, required=True)
        
        # Paths and device
        parser.add_argument('-s', '--save_path', help='checkpoint save path', 
                          default='./checkpoints/decoding', required=True)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', 
                          default='cuda:0')
        
        # Additional settings                  
        parser.add_argument('--data_dir', type=str, help='path to handwriting BCI dataset', 
                          default='/kaggle/working/handwritingBCI')
        parser.add_argument('--hidden_dim', type=int, help='hidden dimension size', 
                          default=512)
        parser.add_argument('--embedding_dim', type=int, help='embedding dimension size', 
                          default=1024)
        parser.add_argument('--save_every', type=int, help='save checkpoint every N epochs', 
                          default=5)
        parser.add_argument('--patience', type=int, help='early stopping patience', 
                          default=5)
        parser.add_argument('--num_workers', type=int, help='number of dataloader workers', 
                          default=4)
        
        args = vars(parser.parse_args())
        
    elif case == 'eval_decoding':
        # args config for evaluating Brain-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for evaluate Brain-To-Text decoder')
        parser.add_argument('-checkpoint', '--checkpoint_path', help='specify model checkpoint', required=True)
        parser.add_argument('-conf', '--config_path', help='specify training config json', required=True)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', 
                          default='cuda:0')
        args = vars(parser.parse_args())

    return args
