import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmallCNN(nn.Module):
    def __init__(
            self,
            input_channels=3,
            num_filters=[264,64,128,256,256],
            kernel_size=[5,5,5,5,5],
            activation='relu',
            dense_neurons=128,
            num_layers=5,
            num_classes=10,
            apply_batch_norm=False,
            dropout_prob = 0.3,
            input_size=[128,128]):
        super(SmallCNN, self).__init__()
        self.input_channels = input_channels
        self.activation = self._get_activation_function(activation)
        self.apply_batch_norm = apply_batch_norm
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dense_neurons = dense_neurons
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout_prob =  dropout_prob
        
        layers = []
        in_channels = self.input_channels
        for i in range(self.num_layers): 
            layers.append(nn.Conv2d(in_channels, self.num_filters[i], kernel_size=self.kernel_size[i], padding=self.kernel_size[i]//2))
            if self.apply_batch_norm:
                layers.append(nn.BatchNorm2d(self.num_filters[i]))
            layers.append(self.activation)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = self.num_filters[i]
        
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.conv_out_shape = self.get_conv_output_size()
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_out_shape, self.dense_neurons),
            self.activation,
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dense_neurons, self.num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_conv_output_size(self):
        dummy_input = torch.ones(1, 3, *self.input_size)
        x = self.conv_layers(dummy_input)
        return x.view(x.size(0), -1).size(1)
        
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