import torch
import torch.nn as nn
import torch.optim as optim
import logging
from model import SmallCNN
from dataset import iNaturalistDataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10, device="cuda",eval_frequency=2):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.eval_frequency = eval_frequency
    
    def train(self):
        self.model.to(self.device)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / total
            train_accuracy = 100 * correct / total
            if (epoch + 1) % self.eval_frequency == 0:
                logger.info("Evaluating on validation dataset")
                val_accuracy = self.evaluate(self.val_loader)
                logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Val Acccuracy: {val_accuracy:.2f}%")

            else: 
                logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

    
    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def test(self):
        test_accuracy = self.evaluate(self.test_loader)
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        return test_accuracy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = iNaturalistDataset(
        dataset_path="iNaturalist_dataset",
        batch_size=32,
        apply_augmentation=True,
        download_url="https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    # Initialize model
    model = SmallCNN(
        input_channels=3,num_layers=5, num_filters=[64,64,32,32,32], kernel_size=[5,5,3,3,3],
        activation='mish', dense_neurons=512, apply_batch_norm=True,
        num_classes=10,input_size=[128,128]
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
        eval_frequency=2
    )

    # Train the model
    logger.info("Training started...")
    trainer.train()

    # Test the model
    trainer.test()

if __name__ == "__main__":
    main()
