# %%
%load_ext autoreload
%autoreload 2
from model import BASIC_CNN1D
from data import get_dataloader
import torch
# %%
dataloader = get_dataloader()
# %%
def training_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    for i, (X, y) in enumerate(dataloader):
        print(X)
        print(f"y: {y}")
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        print(f"batch number: {i}, loss: {loss.item()}")

model = BASIC_CNN1D()
epochs = 10
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(epochs):
    print(f"epoch: {epoch + 1} / {epochs}")

    training_loop(dataloader=dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)


# %%
