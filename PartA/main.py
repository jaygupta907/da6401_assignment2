import logging
from dataset import iNaturalistDataset
from model import SmallCNN
import torch
import torch.nn as nn
import torch.optim as optim
from training import Trainer
from config import get_args
import wandb


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

args = get_args()

#experiment_name = f"lr_{args.learning_rate}_filter_{args.filter_depth}_kernel_{args.kernel_size}_aug_{args.apply_augmentation}_batchnorm_{args.apply_batch_norm}_drop_{args.dropout_prob}_batch_{args.batch_size}_dense_{args.dense_neurons}_activation_{args.activation}"
experiment_name ="best_run"
wandb.init(project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=experiment_name
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")


    run = wandb.init()
    config = wandb.config

    # Load dataset
    dataset = iNaturalistDataset(
        dataset_path="../iNaturalist_dataset",
        batch_size=args.batch_size,
        apply_augmentation=args.apply_augmentation,
        download_url="https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    num_filters = None
    if args.filter_depth == 'same':
        num_filters = [64,64,64,64,64]
    elif args.filter_depth == 'increasing':
        num_filters = [64,128,256,512,1024]
    elif args.filter_depth == 'decreasing':
        num_filters = [512,256,128,64,32]


    kernel_size = None
    if args.kernel_size == 'same':
        kernel_size = [5,5,5,5,5]
    elif args.kernel_size == 'increasing':
        kernel_size = [3,3,5,5,7]
    elif args.kernel_size == 'decreasing':
        kernel_size = [7,5,5,3,3]
    
    # Initialize model
    model = SmallCNN(
        input_channels=3,num_layers=5, num_filters=num_filters, kernel_size=kernel_size,
        activation=args.activation, dense_neurons=args.dense_neurons, apply_batch_norm=args.apply_batch_norm,
        num_classes=10,input_size=[128,128]
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir="model",
        config=config
    )

    # Train the model
    logger.info("Training started...")
    trainer.train()

    # Test the model
    trainer.test()

    wandb.finish()
    logger.info("Training and testing completed.")


if __name__ == "__main__":
    main()
