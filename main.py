# %%
%load_ext autoreload
%autoreload 2
from model import BASIC_CNN1D
from data import get_dataloader, generate_uniform_data
import torch
import torch.nn.functional as F
# %%
train_dataloader = get_dataloader(length=100)
test_dataloader = get_dataloader(length=10)
# %%
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # print(f"y: {y}")
            pred = model(X)
            print(f"prediction: {pred}")
            # y = F.one_hot(y, num_classes=2)
            # y = y.to(torch.float32)
            # print(f"y onehot: {y}")
            test_loss += loss_fn(pred, y).item()
            # print(f"pred: {pred}")
            # print(f"pred argmax: {pred.argmax(1)}")
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct, test_loss

def training_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    for i, (X, y) in enumerate(dataloader):
        # y = F.one_hot(y, num_classes=2)
        # y = y.to(torch.float32)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        # print(f"batch number: {i}, loss: {loss.item()}")

model = BASIC_CNN1D()
epochs = 100
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(epochs):
    print(f"epoch: {epoch + 1} / {epochs}")

    training_loop(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    correct, test_loss = test_loop(dataloader=test_dataloader, model=model, loss_fn=loss_fn)

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

print("Done")
# %%
