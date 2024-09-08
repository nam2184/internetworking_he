import torch
import torch.nn as nn

class CNN(nn.Module):
    """A CNN to classify the MNIST 8x8 monochrome data-set."""

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)  # Output: 8x8
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)  # Output: 8x8
        self.conv3 = nn.Conv2d(16, 32, 2, stride=1, padding=0)  # Output: 2x2 -> 1x1

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output size halves with pooling

        # Fully connected layer (after conv3 the size is 32 channels of 1x1 feature maps)
        self.fc1 = nn.Linear(32, n_classes)  # Adjusted for 32 channels
    
    def forward(self, x):
        """Run inference on the CNN."""
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)  # Output: 4x4

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)  # Output: 2x2

        x = self.conv3(x)
        x = torch.relu(x)  # Output: 1x1 (32 channels)

        x = x.flatten(1)  # Flatten the 1x1 feature map for the fully connected layer
        x = self.fc1(x)
        return x
