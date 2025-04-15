import logging
from dataset import iNaturalistDataset
from model import PretrainedModel
import torch
import torch.nn as nn
import torch.optim as optim
from finetuning import Trainer
from config import get_args
import wandb



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

args = get_args()

experiment_name = f"Pretrained_{args.model_name}"
wandb.init(project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=experiment_name
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")


    input_size = {
    "resnet50": 224,
    "vgg16": 224,
    "googlenet": 224,
    "inception_v3": 299,
    "efficientnet": 384,
    "vit": 224,
    }[args.model_name]


    # Load dataset
    dataset = iNaturalistDataset(
        dataset_path="../iNaturalist_dataset",
        batch_size=args.batch_size,
        apply_augmentation=args.apply_augmentation,
        download_url="https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders(input_size=input_size)

    
    # Initialize model
    pretrained_model = PretrainedModel(
        num_classes=10,
        model_name=args.model_name
    )
    model = pretrained_model.get_trainable_model(strategy=args.strategy,k=args.k)

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
        args=args,
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
