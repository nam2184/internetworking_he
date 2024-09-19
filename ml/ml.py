import time

import numpy as np
import torch
import torch.utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from concrete.ml.torch.compile import compile_torch_model

class CNN(nn.Module):
    """A very small CNN to classify the sklearn digits data-set."""

    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=1)  # Output: 8x8
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # Pooling
        #average it
        self.fc1 = nn.Linear(4 * 4 * 4, n_classes)  # Adjust for simplified layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)  # Output: 4x4
        x = x.flatten(1)  # Flatten
        x = self.fc1(x)
        return x
    

