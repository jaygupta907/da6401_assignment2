import wandb
from dataset import iNaturalistDataset
from model import SmallCNN
import torch
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
import torchvision.transforms.functional as TF
from config import get_args
import torchvision.utils as vutils
from PIL import Image
import io
import matplotlib.pyplot as plt

class visualize:
    def __init__(self,args):
        # Load dataset
        self.args = args
        dataset = iNaturalistDataset(
            dataset_path="../iNaturalist_dataset",
            batch_size=args.batch_size,
            apply_augmentation=False,
            download_url="https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
        )
        _, _, test_loader = dataset.get_dataloaders()

        self.num_filters = None
        if args.filter_depth == 'same':
            self.num_filters = [64,64,64,64,64]
        elif args.filter_depth == 'increasing':
            self.num_filters = [64,128,256,512,1024]
        elif args.filter_depth == 'decreasing':
            self.num_filters = [512,256,128,64,32]


        self.kernel_size = None
        if args.kernel_size == 'same':
            self.kernel_size = [5,5,5,5,5]
        elif args.kernel_size == 'increasing':
            self.kernel_size = [3,3,5,5,7]
        elif args.kernel_size == 'decreasing':
            self.kernel_size = [7,5,5,3,3]

        self.model = SmallCNN(input_channels=3, num_layers=5, num_filters=self.num_filters,
                        kernel_size=self.kernel_size, activation=args.activation, dense_neurons=args.dense_neurons,
                        apply_batch_norm=True, num_classes=10, input_size=[128, 128]
                        )
        # Load the model
        self.model.load_state_dict(torch.load("model/model_epoch_50.pt"))

        images, labels = next(iter(test_loader))  
        self.images, self.labels = images[:30], labels[:30]


    def log_predictions(self):
        with torch.no_grad():
            outputs = self.model(self.images)
            preds = torch.argmax(outputs, dim=1)

        classes = ["Amphibia","Animalia","Arachnida","Aves","Fungi","Insecta","Mammalia","Mollusca","Plantae","Reptilia"]

        wandb_images = []
        for i in range(len(self.images)):
            img = self.images[i]          # Tensor: (C, H, W)
            true_label = classes[self.labels[i]]
            pred_label = classes[preds[i]]

            # Convert tensor to PIL Image for wandb.Image
            mean = torch.tensor([0.5,0.5,0.5]).view(3, 1, 1)
            std = torch.tensor([0.5,0.5,0.5]).view(3, 1, 1)
            img = img*std+mean
            img = torch.clamp(img, 0, 1)
            pil_img = TF.to_pil_image(img.cpu())
            caption = f"True: {true_label} | Pred: {pred_label}"

            wandb_images.append(wandb.Image(pil_img, caption=caption))

        # Log them as a grid (auto-formats in W&B UI)
        wandb.log({"Prediction Grid": wandb_images})

    def visualize_first_layer_features(self):
        """
        Logs all feature maps from the first convolutional layer to W&B.
        """
        feature_maps = []

        def hook_fn(module, input, output):
            feature_maps.append(output)

        # Register hook to the first conv layer
        first_conv_layer = None
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                first_conv_layer = layer
                break

        assert first_conv_layer is not None, "No Conv2D layer found in model features."
        hook_handle = first_conv_layer.register_forward_hook(hook_fn)


        image_tensor = self.images[0]
        # Run the model to capture feature maps
        with torch.no_grad():
            _ = self.model(image_tensor.unsqueeze(0))  # Add batch dim

        # Unregister the hook
        hook_handle.remove()

        # Extract and process feature maps
        fmap_tensor = feature_maps[0].squeeze(0)  # Remove batch dim: shape (C, H, W)
        fmap_tensor = fmap_tensor.cpu()

        images = []
        for i in range(fmap_tensor.shape[0]):
            fmap = fmap_tensor[i]
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)  # Normalize to [0,1]

            # Plot with viridis colormap
            plt.figure(figsize=(8,8))  # Increase image size
            plt.axis('off')
            plt.imshow(fmap.numpy(), cmap='gray')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)

            img = Image.open(buf).convert("RGB")
            img = TF.to_tensor(img)  # (C, H, W)
            images.append(img)

        # Create 8x8 grid
        grid_img = vutils.make_grid(images, nrow=8, padding=4)
        grid_pil = TF.to_pil_image(grid_img)

        # Normalize input image from [-1, 1] to [0, 1]
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        input_image = image_tensor * std + mean
        input_image = torch.clamp(input_image, 0, 1)
        input_image = TF.to_pil_image(input_image.cpu())

        # Log to wandb
        wandb.log({"First layer Feature Maps": wandb.Image(grid_pil)})
        wandb.log({"Input Image": wandb.Image(input_image)})

if __name__ == "__main__":
    args = get_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,config=args,name="Visualization")
    visualizer = visualize(args)
    visualizer.log_predictions()
    visualizer.visualize_first_layer_features()
