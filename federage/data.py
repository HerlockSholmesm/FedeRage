"""Datasets and their partition across clients."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .availability import availability_profile

BLOBS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'blobs')


class TensorPairs(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def iid_split(dataset, num_users):
    """Each client receives len(dataset) / num_users samples drawn uniformly without replacement."""
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def _sorted_by_label(labels, num_items):
    idxs_labels = np.vstack((np.arange(num_items), labels[:num_items]))
    return idxs_labels[:, idxs_labels[1, :].argsort()][0, :]


def shard_split(labels, num_users, num_shards, num_imgs):
    """Sort samples by label, cut them into shards and give each client two random shards."""
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = _sorted_by_label(labels, num_shards * num_imgs)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def rare_class_split(labels, num_users, p, M=90, r=90):
    """Two label-sorted shards per client, with availability-aware placement.

    The M% most available clients (by p) get their shards from the first r% of
    the label-sorted data; the remaining clients get the last (100 - r)%, i.e.
    the least available clients hold the rarest classes.
    """
    sorted_user_indices = np.argsort(-np.array(p))
    dict_users = {i: np.array([]) for i in range(num_users)}
    pack_size = len(labels) // num_users
    num_shards, num_imgs = int(2 * len(labels) // pack_size), pack_size // 2
    idx_shard = [i for i in range(num_shards)]
    idxs = _sorted_by_label(labels, num_shards * num_imgs)

    def assign(users, shards):
        for i in users:
            rand_set = set(np.random.choice(shards, 2, replace=False))
            shards = list(set(shards) - rand_set)
            u = sorted_user_indices[i]
            for rand in rand_set:
                dict_users[u] = np.concatenate((dict_users[u], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    split = int((M / 100) * num_users)
    assign(range(split), idx_shard[:int(r / 100 * num_shards)])
    assign(range(split, num_users), idx_shard[int(r / 100 * num_shards):])
    return dict_users


def get_dataset(args):
    """Return (train_dataset, test_dataset, user_groups), user_groups mapping client -> sample indices."""
    if args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        root = os.path.join(args.data_dir, 'cifar')
        train = datasets.CIFAR10(root, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root, train=False, download=True, transform=transform)
        if args.iid:
            user_groups = iid_split(train, args.num_users)
        else:
            user_groups = shard_split(np.array(train.targets), args.num_users, num_shards=200, num_imgs=250)
        return train, test, user_groups

    if args.dataset == 'blobs':
        load = lambda name, **kw: np.loadtxt(os.path.join(BLOBS_DIR, name), **kw)
        X_train, X_test = load('X_train.txt'), load('X_test.txt')
        y_train, y_test = load('y_train.txt', dtype=int), load('y_test.txt', dtype=int)
        train = TensorPairs(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test = TensorPairs(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        # one client per blob
        user_groups = {i: np.where(y_train == i)[0] for i in range(5)}
        return train, test, user_groups

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    cls = datasets.MNIST if args.dataset == 'mnist' else datasets.FashionMNIST
    root = os.path.join(args.data_dir, args.dataset)
    train = cls(root, train=True, download=True, transform=transform)
    test = cls(root, train=False, download=True, transform=transform)

    # A second, independent draw of the availability profile decides which
    # clients hold the rare classes (identical to the first for deterministic profiles).
    p = availability_profile(args.probs, args.num_users)
    labels = train.targets.numpy()
    if args.iid == 0:
        user_groups = iid_split(train, args.num_users)
    elif args.iid == 1:
        user_groups = shard_split(labels, args.num_users, num_shards=200, num_imgs=300)
    elif args.iid == 3:
        user_groups = rare_class_split(labels, args.num_users, p=p)
    else:
        raise ValueError(f'unsupported partition --iid={args.iid}')
    return train, test, user_groups
