# %%
import torch.nn as nn


'''
Model architecture thoughts:

1. The (input) data
The data is a sequential graph. There are around 500 slices, representing the nodes of our graph.
Each node has around 8 features. The edges are all the same, we are assuming the distance between slices is similar.

Therefore a graph neural network is not ideal. It does not require message passing, a simple convolution and pooling
should suffice/be better.

Shape: (batch_size, amt_input_features, amt_slices) ~= (4, 8, 500)

2. The expected output
There is not enough data to differentiate between issues that appeared in the model. We will have to trust Isaacs classification
of 1s and 0s. Therefore it is a binary output.

2. The model
We will train a 1D convolutional neural network on the data.

a. scale separation/equivariance exploitation:
Exploit that different aspects of the data appear at different scales. This means to have multiple Convolutional Blocks after each other.
One block:
a1. Convolutional layer
a2. Batch normalization layer
a3. Dropout Layer
a4. activation function layer (RELU)
a5. Convolutional layer
a6. Batch normalization layer
a7. Dropout Layer
a8. activation function layer (RELU)
a9. pooling layer (reduce dimensionality)

We will only use a single block, the data isn't that complex.
a10. One last convolutional layer
a11. One or two FC layers.

3. reflections on the layes I plan to use:
a. What is the point of the FC layers?
There is the obvious function, to reduce the output to a tuple (class prediction 0 and class prediction 1). Flattening the spatial features.
However you can also imagine it as a way to combine the features detected at different points in the sequence.

We can use them because we do not actually require spatial preservation on our output. It doesn't matter (for the scope of our thesis)
where the issue appears. We are only classifying if it appears or not.

4. Incorporating globally available data

There is some data that applies to the entire (graph) sequence. Like the age of the patient, the gender etc... .
I do not know that much about multimodal input, but I think I can just be logical. Making the metadata go through
Convolutional layers seems useless, I don't see how it could help detect any features. Furthermore I think that their function
would be more of a shift. Therefore simple linear functions. I will therefore concatenate them after the last pooling layer: a9.

Would they profit from going through a separate preprocessing layer? Maybe, I think yes. For example, since woman and men age differenty
(I think age X means different things wheter it is a woman or a man if we look at the persons fragility). I will make them go through one simple
fully connected layer first.

Separate model:
b1: Input layer (5 ~ 7 global features)
b2: Fully connected layer
b3. Relu activation layer
---
Concat after layer a10, before layer a11 and then continue in the main model.

5. Minimal viable model!
This is obviously too much for a first model, if something goes wrong I will have a hard time to debug it.
Furthermore I will have to adapt each part or decision. This will be experimentally driven.

We want the most simple distillation from above that somehow works and represents the core of the final model.
Things we will drop and add later:
a. Seperate model
b. Dropout layers

This will give us:
a1*. Convolutional layer
a2*. Batch normalization layer
a3*. activation function layer (RELU)
a4*. Convolutional layer
a5*. Batch normalization layer
a6*. activation function layer (RELU)
a7*. pooling layer (reduce dimensionality)

We will only use a single block, the data aint that complex.
a8*. One last convolutional layer
a9*. One or two FC layers.
'''
class BASIC_CNN1D(nn.Module):
    def __init__(self, input_channels=8, num_classes=1):
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

# %%
