import argparse 

def get_args():
    """
    Parses command-line arguments and returns them.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Convolutional Neural Network')

    # Adding various arguments for training configuration
    parser.add_argument('--wandb_project', type=str, default='Convolution Neural Networks', help='Project Name')
    parser.add_argument('--wandb_entity', type=str, default='jay_gupta-indian-institute-of-technology-madras', help='WandB entity name')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--eval_frequency',type=int,default=1,help="What is the frequency of evaluation on validation dataset")
    parser.add_argument('--learning_rate',type=float,default=0.001,help="Learning rate of the optimizer")
    parser.add_argument('--save_frequency', type=int, default=40, help='Frequency of saving the model')
    parser.add_argument('--model_name', type=str, default='vit', help='Model name for transfer learning')
    parser.add_argument('--apply_augmentation', type=bool, default=True, help='Whether to apply augmentation or not')

    return parser.parse_args()