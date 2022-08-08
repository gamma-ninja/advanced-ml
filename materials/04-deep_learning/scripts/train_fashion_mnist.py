import time

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop


# Define model
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_epoch(dataloader, model, loss_fn, optimizer, device=device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device=device), y.to(device=device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def eval_model(dataloader, model, loss_fn, device=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device=device), y.to(device=device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


if __name__ == "__main__":
    epochs = 300
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = NeuralNetwork()
    model = model.to(device=device)

    res = []

    for data_aug in [True, False]:
        for normalize in [True, False]:

            train_transform, test_transform = [], []
            if data_aug:
                train_transform.extend([
                    RandomHorizontalFlip(),
                    RandomCrop(size=28, padding=4)
                ])
            train_transform.append(ToTensor())
            c.append(ToTensor())
            if normalize:
                train_transform.append(Normalize([0.2700], [0.3500]))

            # Download training data from open datasets.
            training_data = datasets.FashionMNIST(
                root="../data",     # Where to store the data
                train=True,     # Which part of the dataset to load (train set)
                download=True,  # Download the data if necessary
                # Transform for data augmetation.
                transform=Compose(train_transform),
            )

            # Download test data from open datasets.
            test_data = datasets.FashionMNIST(
                root="../data",
                train=False,
                download=True,
                transform=Compose(test_transform),
            )

            train_dataloader = DataLoader(
                training_data, batch_size=batch_size, num_workers=10
            )
            test_dataloader = DataLoader(
                test_data, batch_size=batch_size, num_workers=10
            )
            for t in range(epochs):
                print(f"Epoch {t+1}\n{'-' * 20}")
                t_start = time.time()
                train_epoch(
                    train_dataloader, model, loss_fn, optimizer, device=device
                )
                t_train = time.time() - t_start
                t_start
                acc, loss = eval_model(
                    test_dataloader, model, loss_fn, device=device
                )
                t_eval = time.time() - t_start
                res.append(dict(
                    loss=loss, acc=acc, t_train=t_train, t_eval=t_eval,
                    data_aug=data_aug, normalize=normalize
                ))
            print("Done!")

        df = pd.DataFrame(res)
        df.to_csv("run_fashion_mnist.csv")
