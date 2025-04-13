import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmallCNN(nn.Module):
    """
    A configurable small convolutional neural network for image classification tasks.

    Args:
        input_channels (int): Number of input channels (default: 3 for RGB images).
        num_filters (list[int]): Number of filters for each convolutional layer.
        kernel_size (list[int]): Kernel size for each convolutional layer.
        activation (str): Activation function to use ('relu', 'gelu', 'silu', 'mish').
        dense_neurons (int): Number of neurons in the first fully connected layer.
        num_layers (int): Number of convolutional layers to build.
        num_classes (int): Number of output classes for classification.
        apply_batch_norm (bool): Whether to apply Batch Normalization after each Conv layer.
        dropout_prob (float): Dropout probability for regularization.
        input_size (list[int]): Height and width of input images (H, W).

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the network.

    Returns:
        torch.Tensor: Output logits of shape (batch_size, num_classes).
    """
    def __init__(
            self,
            input_channels=3,                 # Number of input image channels (e.g., 3 for RGB)
            num_filters=[64,64,64,64,64],     # Filters for each Conv layer
            kernel_size=[5, 5, 5, 5, 5],      # Kernel size for each Conv layer
            activation='relu',                # Activation function type
            dense_neurons=128,                # Neurons in the first dense layer
            num_layers=5,                     # Number of Conv layers
            num_classes=10,                   # Output classes
            apply_batch_norm=False,           # Apply BatchNorm after Conv
            dropout_prob=0.3,                 # Dropout rate for regularization
            input_size=[128, 128]             # Input image size (height, width)
        ):
        super(SmallCNN, self).__init__()

        # Save config values
        self.input_channels = input_channels
        self.activation = self._get_activation_function(activation)
        self.apply_batch_norm = apply_batch_norm
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dense_neurons = dense_neurons
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout_prob = dropout_prob

        # Build convolutional layers
        layers = []
        in_channels = self.input_channels
        for i in range(self.num_layers): 
            # Conv2d layer with padding to preserve spatial dimensions
            layers.append(nn.Conv2d(in_channels, self.num_filters[i], kernel_size=self.kernel_size[i], padding=self.kernel_size[i]//2))
            if self.apply_batch_norm:
                # Optional batch normalization
                layers.append(nn.BatchNorm2d(self.num_filters[i]))
            # Activation function
            layers.append(self.activation)
            # Max pooling to downsample
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = self.num_filters[i]

        self.conv_layers = nn.Sequential(*layers)  # Stack all conv layers

        self.flatten = nn.Flatten()  # Flatten before feeding to FC layer

        # Calculate size of flattened feature map after conv layers
        self.conv_out_shape = self.get_conv_output_size()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_out_shape, self.dense_neurons),
            self.activation,
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dense_neurons, self.num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier initialization for conv and linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_conv_output_size(self):
        # Create a dummy input to compute the output feature size after conv layers
        dummy_input = torch.ones(1, 3, *self.input_size)
        x = self.conv_layers(dummy_input)
        return x.view(x.size(0), -1).size(1)  # Flatten and get feature count

    def _get_activation_function(self, activation):
        # Map string to actual PyTorch activation function
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'mish':
            return nn.Mish()
        else:
            raise logging.error("Given activation function is not implemented")

    def forward(self, x):
        # Forward pass through conv layers, flatten, then FC layers
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
