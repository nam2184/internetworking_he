from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class Dataset: 
    def __init__(self, x, y):
        self.dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y)) 

    def compress(self):
        pass
    
    def getdata(self):
        return self.dataset

    def load_data(self):
        return DataLoader(self.dataset)


def test_dataset():
    X, y = load_digits(return_X_y=True)

    # The sklearn Digits data-set, though it contains digit images, keeps these images in vectors
    # so we need to reshape them to 2D first. The images are 8x8 px in size and monochrome
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

