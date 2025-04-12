import torchvision
import torchvision.transforms as transforms
import os
import zipfile
from torch.utils.data import DataLoader, random_split
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class iNaturalistDataset:
    def __init__(self, dataset_path="iNaturalist_dataset",
                 batch_size=32,
                 apply_augmentation=False,
                 download_url="https://storage.googleapis.com/wandb_datasets/nature_12K.zip"):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.apply_augmentation = apply_augmentation
        self.download_url = download_url
        self._download_and_extract_dataset()

    def _download_and_extract_dataset(self):
        zip_path = "dataset.zip"
        if not os.path.exists(self.dataset_path):

            logging.info("Downloading the Dataset ..........")
            subprocess.run(["wget", self.download_url, "-O", zip_path], check=True)

            logging.info("Extracting the Dataset ..........")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.dataset_path)
            os.remove(zip_path)
            logging.info("Dataset downloaded and extracted")
        else:
            logging.warning("Dataset already exists")

    def _get_transforms(self):
        # Define data transformations
        transform_list = [
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        if self.apply_augmentation:
            # Apply data augmentation techniques like rotation, translation, etc.
            augmentations = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(translate=(0.1, 0.1),degrees=0),
                transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
            ])
            transform_list = [augmentations] + transform_list
        
        return transforms.Compose(transform_list)

    def get_dataloaders(self):
        transform = self._get_transforms()

        # Load the train dataset
        train_dir = os.path.join(self.dataset_path, "inaturalist_12K/train")
        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
        
        # Split the train dataset into train (80%) and validation (20%) sets
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Load the test dataset
        test_dir = os.path.join(self.dataset_path, "inaturalist_12K/val")
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
