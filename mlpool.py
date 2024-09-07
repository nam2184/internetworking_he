import time

import numpy as np
import torch
import torch.utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from concrete.ml.torch.compile import compile_torch_model

#Encryption can't handle RELU and average pooling
#Replace with approximation for RELU and max pooling


class CNNnoRELU(nn.Module):
    """A small CNN to classify the 8x8 monochrome dataset."""

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()
        # 8x8 monochrome image
        # Adjust the network architecture to match the 8x8 input size
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)  # Output size: (6x6)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)  # Output size: (4x4)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)  # Output size: (2x2)
        
        # Replace MaxPool2d with AveragePool2d for FHE compatibility
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Reduces feature map size by 2x2

        # Fully connected layer: After conv and pool layers, output is (1x1)
        self.fc1 = nn.Linear(32 * 1 * 1, n_classes)

    def forward(self, x):
        """Run inference on the CNN, apply polynomial activations and average pooling."""
        # Convolutional layers with polynomial approximation for activation
        x = self.conv1(x)
        x = x ** 2  # Polynomial approximation for ReLU (x^2)
        x = self.pool(x)

        x = self.conv2(x)
        x = x ** 2  # Polynomial approximation for ReLU (x^2)
        x = self.pool(x)

        x = self.conv3(x)
        x = x ** 2  # Polynomial approximation for ReLU (x^2)
        x = self.pool(x)  # Final pooling to (1x1)

        # Flatten the output and apply the fully connected layer
        x = x.flatten(1)
        x = self.fc1(x)

        return x
