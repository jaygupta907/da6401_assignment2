import torch
import logging
from tqdm import tqdm
import wandb
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, config, device="cuda", save_dir="model"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = config.num_epochs
        self.eval_frequency = config.eval_frequency
        self.save_frequency = config.save_frequency
        self.save_dir = save_dir
        self.device = device

    def train(self):
        self.model.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
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
            wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "epoch": epoch + 1})

            if (epoch + 1) % self.eval_frequency == 0:
                val_accuracy, val_loss = self.evaluate(self.val_loader)
                wandb.log({"val_accuracy": val_accuracy, "val_loss": val_loss, "epoch": epoch + 1})
                logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
            else:
                logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Acc: {train_accuracy:.2f}%")

            if (epoch + 1) % self.save_frequency == 0:
                self.save_model(epoch + 1)

    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(loader, leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                running_loss += self.criterion(outputs, labels).item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total, running_loss / total

    def test(self):
        test_accuracy, test_loss = self.evaluate(self.test_loader)
        wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss})
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        return test_accuracy, test_loss

    def save_model(self, epoch):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")
