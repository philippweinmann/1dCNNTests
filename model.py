# %%
import torch.nn as nn

class BASIC_CNN1D(nn.Module):
    def __init__(self, input_channels=8, num_classes=2):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding='valid')
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding='valid')
        
        # Activation and pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Adaptive pooling to handle variable input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, input_channels, sequence_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Adaptive pooling to get fixed-size output
        x = self.adaptive_pool(x)  # shape: (batch_size, 64, 1)
        x = x.squeeze(-1)          # shape: (batch_size, 64)
        x = self.fc(x)             # shape: (batch_size, num_classes)
        return x
