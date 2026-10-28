"""Models used in the experiments."""

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """MNIST / FashionMNIST: one hidden layer of 64 units."""

    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        return self.Softmax(self.layer_hidden(self.relu(self.layer_input(x))))


class MLPBlobs(nn.Module):
    """Two-dimensional five-blob example of Fig. 1."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 5)
        self.relu = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)
        self.layer1.weight.data = torch.zeros([64, 2])
        self.layer2.weight.data = torch.zeros([5, 64])

    def forward(self, x):
        return self.Softmax(self.layer2(self.relu(self.layer1(x))))


class CNNMnist(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.Softmax(self.fc2(x))


class CNNFashionMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16),
                                    nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32),
                                    nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return self.Softmax(self.fc(out.view(out.size(0), -1)))


class CNNCifar(nn.Module):
    """CIFAR-10: LeNet-style CNN."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.Softmax(self.fc3(x))


def build_model(args, train_dataset):
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            return CNNMnist(args.num_channels, args.num_classes)
        if args.dataset == 'fmnist':
            return CNNFashionMnist()
        return CNNCifar(args.num_classes)

    # The published runs built an MLP for every partial product of the input
    # shape (e.g. inputs of size 1, 28 and 784 for MNIST) and kept the last one;
    # for the blobs example that MLP was then replaced by MLPBlobs. The throwaway
    # constructions consume the torch RNG, so they are repeated here to keep the
    # same initialization for a given seed.
    dim_in = 1
    for d in train_dataset[0][0].shape:
        dim_in *= d
        model = MLP(dim_in=dim_in, dim_hidden=64, dim_out=args.num_classes)
    return MLPBlobs() if args.dataset == 'blobs' else model
