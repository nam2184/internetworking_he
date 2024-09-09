import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def quantize_input(input_data, num_bits=8):
    """Quantize input data to a fixed number of bits."""
    min_val, max_val = np.min(input_data), np.max(input_data)
    scale = (2 ** num_bits - 1) / (max_val - min_val)
    quantized_data = np.round((input_data - min_val) * scale).astype(np.uint8)
    return quantized_data

class Dataset: 
    def __init__(self, x, y, quantize=False):
        self.quantize = quantize
        if self.quantize:
            self.x_data = quantize_input(x)
        else:
            self.x_data = x
        self.y_data = y
        self.dataset = TensorDataset(torch.Tensor(self.x_data), torch.Tensor(self.y_data))
    
    def load_data(self):
        return DataLoader(self.dataset)

def test_dataset():
    X, y = load_digits(return_X_y=True)
    X = np.expand_dims(X.reshape((-1, 8, 8)), 1)

    nplot = 4
    fig, ax = plt.subplots(nplot, nplot, figsize=(6, 6))
    for i in range(0, nplot):
        for j in range(0, nplot):
            ax[i, j].imshow(X[i * nplot + j, ::].squeeze())

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42
    )

    return x_train, x_test, y_train, y_test
