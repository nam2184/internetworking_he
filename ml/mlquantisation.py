import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

#Quantization of Weights and Activations: 
# Quantization involves reducing the precision 
# of weights and activations from floating-point representations (e.g., 32-bit) 
# to lower-precision formats like integers (e.g., 8-bit, 4-bit). 
# This reduces the computational load and memory footprint, 
# which is important in FHE where operations are expensive.


class QuantizedCNN(nn.Module):
    """A small quantized CNN adapted for FHE to classify a simple dataset."""

    def __init__(self, n_classes, n_bits=8):
        """Construct the CNN with quantization."""
        super(QuantizedCNN, self).__init__()
        # 8x8 monochrome image
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)  # Output size: (6x6)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)  # Output size: (4x4)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)  # Output size: (2x2)

        # Efficient pooling using average pooling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Reduces size by 2x2

        # Fully connected layer
        self.fc1 = nn.Linear(32 * 1 * 1, n_classes)

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Save the number of bits for quantization
        self.n_bits = n_bits

    def forward(self, x):
        """Run inference on the CNN with quantized operations."""
        # Quantize input
        x = self.quant(x)

        # Convolution layers with average pooling and polynomial activation approximation
        x = self.conv1(x)
        x = x ** 2  # Polynomial approximation for ReLU (x^2)
        x = self.pool(x)

        x = self.conv2(x)
        x = x ** 2  # Polynomial approximation for ReLU (x^2)
        x = self.pool(x)

        x = self.conv3(x)
        x = x ** 2  # Polynomial approximation for ReLU (x^2)
        x = self.pool(x)

        # Flatten for fully connected layer
        x = x.flatten(1)
        x = self.fc1(x)

        # Dequantize output
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse the convolution and activation layers for efficient FHE operations."""
        # Fuse conv and relu layers for efficient FHE compatibilit                                                                                                                                              
        torch.quantization.fuse_modules(self, [['conv1', 'relu1'], ['conv2', 'relu2'], ['conv3', 'relu3']], inplace=True)

    def quantize_model(self):
        """Convert the model to a quantized version with lower precision."""
        # Apply post-training static quantization
        torch.quantization.prepare(self, inplace=True)
        torch.quantization.convert(self, inplace=True)


# Example usage
if __name__ == "__main__":
    # Dummy input: (batch_size, channels, height, width)
    X = torch.randn((1, 1, 8, 8))  # Example batch with one 8x8 monochrome image

    # Create the CNN model
    model = QuantizedCNN(n_classes=10)

    # Quantize the model to lower precision (e.g., 8-bit)
    model.quantize_model()

    # Run the model on the input data
    output = model(X)
    print(output)
