import wandb
import plotly
from dataset import iNaturalistDataset
import plotly.express as px
from model import SmallCNN
import torch


def log_predictions(args):
    """
    Logs predictions using wandb.
    """
    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    # Load dataset
    dataset = iNaturalistDataset(
        dataset_path="../iNaturalist_dataset",
        batch_size=32,
        apply_augmentation=False,
        download_url="https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    model = SmallCNN(input_channels=3, num_layers=5, num_filters=[64, 64, 64, 64, 64],
                     kernel_size=[5, 5, 5, 5, 5], activation='relu', dense_neurons=128,
                     apply_batch_norm=False, num_classes=10, input_size=[128, 128])
    # Load the model
    model.load_state_dict(torch.load("model.pt"))
    images,labels = test_loader.dataset[0]