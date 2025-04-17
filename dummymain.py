# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# %%
# Generate synthetic data
num_samples = 1000
X = np.random.randn(num_samples, 2)  # 2D data points
y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(np.float32)  # Circular decision boundary

print(X.shape)
print(y.shape)
# Convert to PyTorch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float().view(-1, 1)  # Reshape to (batch_size, 1)

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

# Define model
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1))
        
    def forward(self, x):
        return self.layers(x)

model = Classifier()
criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid activation
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100

# %%
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)
        print(outputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Calculate accuracy after each epoch
    with torch.no_grad():
        predictions = torch.sigmoid(model(X))  # Apply sigmoid for probabilities
        predicted_classes = (predictions > 0.5).float()
        accuracy = (predicted_classes == y).float().mean()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Example inference
test_point = torch.tensor([[1.0, 1.0], [-0.5, 0.5]])
with torch.no_grad():
    prob = torch.sigmoid(model(test_point))
    print(f"Predicted probabilities: {prob.squeeze().numpy()}")
# %%
