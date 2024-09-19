import numpy as np
from tqdm import tqdm

class FHEBase :
    def __init__(self, nn, keycompiler, data) -> None:
        self.nn = nn
        self.keycompiler = keycompiler
        self.data = data
        self.use_sim = False

    def preprocessing(self):
        pass

    def compile_model(self):
        pass

    def test(self): 
        # Casting the inputs into int64 is recommended
        all_y_pred = np.zeros((len(self.data)), dtype=np.int64)
        all_targets = np.zeros((len(self.data)), dtype=np.int64)

        # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
        idx = 0
        for data, target in tqdm(self.data):
            data = data.numpy()
            target = target.numpy()

            fhe_mode = "simulate" if self.use_sim else "execute"
            # Quantize the inputs and cast to appropriate self.data type
            y_pred = self.keycompiler.forward(data, fhe=fhe_mode)

            endidx = idx + target.shape[0]

            # Accumulate the ground truth labels
            all_targets[idx:endidx] = target

            # Get the predicted class id and accumulate the predictions
            y_pred = np.argmax(y_pred, axis=1)
            all_y_pred[idx:endidx] = y_pred

            # Update the index
            idx += target.shape[0]
        # Compute and report results
        n_correct = np.sum(all_targets == all_y_pred)
        return n_correct / len(self.data)
