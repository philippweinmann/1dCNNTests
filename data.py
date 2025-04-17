# %%
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification


# %%
def generate_data(n_samples, n_features, n_slices):
    X = np.random.uniform(
        low=0, high=1, size=(n_samples, n_features, n_slices)
    )  # 2D data points
    X[0 : n_samples // 2, :, 0:200] = 0
    X[n_samples // 2 :, :, 0:200] = 1

    X = X.astype(np.float32)

    y = np.ones(shape=(n_samples))
    y[0 : n_samples // 2] = 1
    y[n_samples // 2 :] = 0

    y = y.astype(np.float32)

    return X, y


# %%
class CustomDataset(Dataset):
    def __init__(self, length, sequence_length=500, num_features=8):
        self.length = length
        X, y = generate_data(
            n_samples=self.length, n_features=num_features, n_slices=sequence_length
        )
        self.X = X
        self.y = y

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # X, y = generate_uniform_data()
        current_X = self.X[index]
        current_y = self.y[index]
        return current_X, current_y


def get_dataloader(length, batch_size):
    dataloader = DataLoader(CustomDataset(length), batch_size=batch_size, shuffle=True)

    return dataloader
