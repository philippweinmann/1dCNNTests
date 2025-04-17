# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# %%
# Generate synthetic data
num_samples = 100
X = np.random.uniform(low=0, high=1, size=(num_samples, 8, 400))  # 2D data points
X[0:50, :, 0:200] = 1
X[50:, :, 0:200] = 0

X = X.astype(np.float32)

y = np.ones(shape=(num_samples))
y[0:50] = 1
y[50:] = 0

y = y.astype(np.float32)


# %%
# Create Dataset and DataLoader
class CircleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


dataset = CircleDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# %%
# Define model
class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()

        # Input shape: (batch_size, 8, 400)
        self.conv1 = nn.Conv1d(8, 16, kernel_size=3, padding=1)  # (16, 400)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)  # (16, 200)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # (64, 100)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(2)  # (64, 50)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)  # (64, 1)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.adaptiveavgpool(x)

        x = self.classifier(x)
        return x


model = CNN1D(num_classes=1)
criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid activation
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Test loop
def test_loop(model, test_dataloader, loss_fn):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred.squeeze(), y.squeeze()).item()

            print(pred)
            print(y)

            if pred > 0.5 and y == 1:
                correct += 1

            if pred < 0.5 and y == 0:
                correct += 1

    test_loss /= len(test_dataloader)
    correct /= len(test_dataloader)

    return test_loss, correct


# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)
        print(outputs)
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test_loop
    test_loss, accuracy = test_loop(model, test_dataloader, loss_fn=criterion)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
    )
# %%
