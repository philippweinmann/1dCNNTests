# %%
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
# %%
def generate_uniform_data(sequence_length = 500, num_features=8):
    # (batch_size, sequence_length, num_features), output
    X = np.random.uniform(low = 0, high=1, size = (num_features, sequence_length)).astype(np.float32)
    y = np.array([1, 0]).astype(np.float32)

    return X, y

def generate_proper_dataset(n_samples=105, sequence_length = 500, num_features=8):
    # where we can test if the model actually learns something.
    n_features = sequence_length * num_features
    n_informative = n_features // 10

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_informative, n_repeated=n_informative, n_classes=2, random_state=42)
    X = X.reshape((n_samples, num_features, sequence_length)).astype(np.float32)

    return X, y
# %%
class CustomDataset(Dataset):
    def __init__(self, length, sequence_length = 500, num_features=8):
        self.length = length
        X, y = generate_proper_dataset(n_samples=self.length)
        self.X = X
        self.y = y

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # X, y = generate_uniform_data()
        current_X = self.X[index,:]
        current_y = self.y[index]
        return current_X, current_y
    
def get_dataloader(length):
    dataloader = DataLoader(CustomDataset(length), batch_size=4, shuffle=True)

    return dataloader