import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmallCNN(nn.Module):
    def __init__(
            self,
            input_channels=3,
            num_filters=[32,32,32,32,32],
            kernel_size=[5,5,5,5,5],
            activation='relu',
            dense_neurons=128,
            num_layers=5,
            num_classes=10,
            apply_batch_norm=False):
        super(SmallCNN, self).__init__()
        self.activation = self._get_activation_function(activation)
        
        layers = []
        in_channels = input_channels
        for i in range(num_layers): 
            layers.append(nn.Conv2d(in_channels, num_filters[i], kernel_size=kernel_size[i], padding=kernel_size[i]//2))
            layers.append(self.activation)
            if self.apply_batch_norm:
                layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters[i]
        
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(num_filters, dense_neurons),
            self.activation,
            nn.Linear(dense_neurons, num_classes)
        )
        
    def _get_activation_function(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'mish':
            return nn.Mish()
        else:
            raise  logging.error("Given activation function is not implemented")

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x