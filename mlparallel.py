import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np




# Multiprocessing with quantisation from mlquantisation


class QuantizedCNN(nn.Module):
    """A small quantized CNN adapted for FHE to classify a simple dataset."""

    def __init__(self, n_classes):
        super(QuantizedCNN, self).__init__()
        # Convolutional layers and average pooling
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 1 * 1, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x ** 2  # Polynomial approximation for ReLU (x^2)
        x = self.pool(x)

        x = self.conv2(x)
        x = x ** 2  # Polynomial approximation for ReLU (x^2)
        x = self.pool(x)

        x = self.conv3(x)
        x = x ** 2  # Polynomial approximation for ReLU (x^2)
        x = self.pool(x)

        x = x.flatten(1)
        x = self.fc1(x)
        return x

# Function to perform inference on a single batch
def run_inference_on_batch(model, batch_data):
    with torch.no_grad():  # Disable gradient calculation
        output = model(batch_data)
    return output

# Multiprocessing function to parallelize batch processing
def parallel_inference(model, data_batches):
    # Create a multiprocessing pool
    pool = mp.Pool(mp.cpu_count())  # Use all available CPU cores

    # Distribute the workload across processes
    results = pool.starmap(run_inference_on_batch, [(model, batch) for batch in data_batches])

    # Close the pool and wait for the tasks to complete
    pool.close()
    pool.join()

    return results

# Example usage
if __name__ == "__main__":
    # Initialize the CNN model
    model = QuantizedCNN(n_classes=10)
    
    # Example data: Batch of four 8x8 monochrome images
    batch_size = 4
    data = torch.randn((batch_size, 1, 8, 8))  # Example batch of data

    # Split data into smaller batches for parallel processing
    data_batches = torch.split(data, 1)  # Split into individual images

    # Run inference in parallel
    results = parallel_inference(model, data_batches)

    # Print the results
    for result in results:
        print(result)
