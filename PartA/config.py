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
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--apply_augmentation', type=bool, default=True, help='Whether to apply augmentation or not')
    parser.add_argument('--apply_batch_norm',type=bool,default=True,help='Whether to use the batch normalization layer')
    parser.add_argument('--eval_frequency',type=int,default=1,help="What is the frequency of evaluation on validation dataset")
    parser.add_argument('--learning_rate',type=float,default=0.0001,help="Learning rate of the optimizer")
    parser.add_argument('--filter_depth',type=str,default='increasing',help='Whether to keep the filter depth same, increase by a factor of 2 or decrease by a factor of 2')
    parser.add_argument('--kernel_size',type=str,default='increasing',help='Whether to keep the kernel size same, increase by a factor of 2 or decrease by a factor of 2')
    parser.add_argument('--dropout_prob', type=float, default=0.3, help='Dropout probability in dense layers')
    parser.add_argument('--dense_neurons', type=int, default=4096, help='Number of neurons in the dense layer')
    parser.add_argument('--activation', type=str, default='silu', help='Activation function to use')
    parser.add_argument('--save_frequency', type=int, default=10, help='Frequency of saving the model')

    return parser.parse_args()