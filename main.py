# %%
%load_ext autoreload
%autoreload 2

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import BASIC_CNN1D
from data import get_dataloader

# %%
train_dataloader = get_dataloader(length=10, batch_size=4)
test_dataloader = get_dataloader(length=20, batch_size=4)
# %%
model = BASIC_CNN1D(num_classes=1)
criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid activation
optimizer = optim.Adam(model.parameters(), lr=0.01)


# %%
def is_correct(pred, y):
    amt_correct = 0
    for batch_num in range(pred.shape[0]):
        current_pred = pred[batch_num]
        current_label = y[batch_num]

        # let's transform the value into percentages.
        prob = torch.sigmoid(current_pred)

        if current_label == 1 and prob > 0.5:
            amt_correct += 1
            continue

        if current_label == 0 and prob < 0.5:
            amt_correct += 1
            continue

    return amt_correct


def train_loop(model, train_dataloader, loss_fn, optimizer):
    model.train()

    for inputs, labels in train_dataloader:
        # Forward pass
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs.squeeze(), labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()


# Test loop
def test_loop(model, test_dataloader, loss_fn):
    model.eval()
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            total_loss += loss_fn(pred.squeeze(), y.squeeze()).item()

            if is_correct(pred, y):
                correct += 1

    avg_loss = total_loss / len(test_dataloader)
    correct /= len(test_dataloader)

    return avg_loss, correct


# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    train_loop(model, train_dataloader, loss_fn=criterion, optimizer=optimizer)
    test_loss, accuracy = test_loop(model, test_dataloader, loss_fn=criterion)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
    )
# %%
# let's test some of them out.


def test_values(indexes):
    num_samples = 100
    X = np.random.uniform(low=0, high=1, size=(num_samples, 8, 400))  # 2D data points
    X[0:50, :, 0:200] = 1
    X[50:, :, 0:200] = 0

    X = X.astype(np.float32)

    y = np.ones(shape=(num_samples))
    y[0:50] = 1
    y[50:] = 0

    y = y.astype(np.float32)

    model.eval()

    X = torch.tensor(X)
    y = torch.tensor(y)

    for index in indexes:
        print(f"index: {index}")

        c_y = y[index]
        c_X = X[index]

        print(f"ground truth: {c_y}")
        # print(c_X)

        pred = model(c_X.unsqueeze(0))
        print(f"pred: {pred}")

        print(f"prob: {torch.sigmoid(pred)}")


test_values(indexes=[0, 1, 2, 98, 99])

# %%
