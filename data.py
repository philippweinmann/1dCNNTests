# %%
import numpy as np
from torch.utils.data import Dataset, DataLoader
# %%
def generate_data(sequence_length = 500, num_features=8):
    # (batch_size, sequence_length, num_features), output
    X = np.random.uniform(low = 0, high=1, size = (num_features, sequence_length)).astype(np.float32)
    y = np.array([1, 0]).astype(np.float32)

    return X, y
# %%
class CustomDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        X, y = generate_data()

        return X, y
    
def get_dataloader():
    dataloader = DataLoader(CustomDataset(length=110), batch_size=4, shuffle=True)

    return dataloader