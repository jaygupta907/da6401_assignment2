import argparse 

def get_args():
    """
    Parses command-line arguments and returns them.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Multilayer Feedforward Neural Network')

    # Adding various arguments for training configuration
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--apply_augmentation', type=bool, default=True, help='Whether to apply augmentation or not')
    parser.add_argument('apply_batch_norm',type=bool,default=True,help='Whether to use the batch normalization layer')
    parser.add_argument('device',type='str',default='cuda',help='Whether to use cpu or gpu')
    parser.add_argument('eval_frequency',type=int,default=5,help="What is the frequency of evaluation on validation dataset")
    parser.add_argument('learning_rate',type=float,default=0.001,help="Learning rate of the optimizer")

    return parser.parse_args()