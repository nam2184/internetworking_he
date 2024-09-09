import data
from ml import CNN
import torch
from tqdm import tqdm
from torch import nn
import numpy as np
from he import FHEBase
from concrete.ml.torch.compile import compile_torch_model
import time

def quantize_input(input_data, num_bits=8):
    """Quantize input data to a fixed number of bits."""
    min_val, max_val = np.min(input_data), np.max(input_data)
    scale = (2 ** num_bits - 1) / (max_val - min_val)
    quantized_data = np.round((input_data - min_val) * scale).astype(np.uint8)
    return quantized_data

def train_one_epoch(net, optimizer, train_loader):
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()
    avg_loss = 0
    correct_predictions = 0
    total_samples = 0
    net.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss_net = loss(output, target.long())
        predicted_classes = torch.argmax(output, dim=1)
        correct_predictions += (predicted_classes == target).sum().item()
        total_samples += target.size(0)
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()

    accuracy = correct_predictions / total_samples
    return accuracy, avg_loss / len(train_loader)


def test_training_no_fhe():
    N_EPOCHS = 20
    
    x_train, x_test, y_train, y_test = data.test_dataset()
    
    # Create a train data loader with quantization
    train_dataset = data.Dataset(x_train, y_train, quantize=True)
    train_dataloader = train_dataset.load_data()
    
    # Train the network with Adam, output the test set accuracy every epoch
    net = CNN(10)
    losses_bits = []
    optimizer = torch.optim.Adam(net.parameters())
    with tqdm(total=N_EPOCHS, unit=" samples") as pbar:
        for epoch in range(N_EPOCHS):
            accuracy, loss = train_one_epoch(net, optimizer, train_dataloader)
            losses_bits.append(loss)
            pbar.set_description(f"Epoch {epoch} - Accuracy: {accuracy:.4f} - Loss: {loss:.4f}")
            pbar.update(1)  # Update progress bar
    return net

def test_concrete(net, sample_size=None):
    x_train, x_test, y_train, y_test = data.test_dataset()
    
    if sample_size is not None:
        indices = np.random.choice(len(x_test), size=sample_size, replace=False)
        x_test = x_test[indices]
        y_test = y_test[indices]
    
    # Quantize the test data
    x_test = quantize_input(x_test)
    
    test_dataset = data.Dataset(x_test, y_test)
    test_dataloader = test_dataset.load_data()
    
    q_module = compile_torch_model(net, x_train, rounding_threshold_bits=2, p_error=0.1)
    t = time.time()
    q_module.fhe_circuit.keygen()
    
    print(f"Keygen time: {time.time()-t:.2f}s")
    fhe_model = FHEBase(net, q_module, test_dataloader)
    accuracy_test = fhe_model.test()
    elapsed_time = time.time() - t
    time_per_inference = elapsed_time / (sample_size if sample_size is not None else len(y_test))
    accuracy_percentage = 100 * accuracy_test

    print(
        f"Time per inference in FHE: {time_per_inference:.2f} "
        f"with {accuracy_percentage:.2f}% accuracy"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    net = test_training_no_fhe()  
    
    test_concrete(net) 
    #To test a certain number of samples, input sample_size
    #EG:
    #test_concrete(net, sample_size=X) 
