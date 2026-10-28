"""Client-side local training and evaluation."""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ClientData(Dataset):
    """A client's subset of a dataset, preloaded onto the training device."""

    def __init__(self, dataset, idxs, device):
        self.data = []
        for i in idxs:
            image, label = dataset[int(i)]
            self.data.append((torch.as_tensor(image).to(device), torch.as_tensor(label).to(device)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def mean_cvar_loss(loss, t, alpha, gamma):
    """gamma * loss + (1 - gamma) * (t + (loss - t)_+ / alpha).

    `gamma` weighs the plain loss and 1 - gamma the CVaR term.
    alpha >= 1 or gamma >= 1 reduces to the plain loss (FedAvg).
    """
    if alpha >= 1. or gamma >= 1.:
        return loss
    return (1 - gamma) * t + ((1 - gamma) / alpha) * torch.relu(loss - t) + gamma * loss


class Client:
    def __init__(self, args, dataset, idxs, model):
        self.args = args
        self.device = f'cuda:{args.gpu}' if args.gpu > -1 else 'cpu'
        idxs = list(idxs)
        # 80 / 10 / 10 split of the client's data into train / validation / test
        n = len(idxs)
        train, val, test = idxs[:int(0.8 * n)], idxs[int(0.8 * n):int(0.9 * n)], idxs[int(0.9 * n):]
        self.trainloader = DataLoader(ClientData(dataset, train, self.device), batch_size=args.local_bs, shuffle=True)
        self.validloader = DataLoader(ClientData(dataset, val, self.device), batch_size=len(val), shuffle=False)
        self.testloader = DataLoader(ClientData(dataset, test, self.device), batch_size=len(test), shuffle=False)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # SCAFFOLD client control variate c_i
        self.ci = {name: torch.zeros_like(p.data, device=self.device) for name, p in model.named_parameters()}

    def _optimizer(self, params, lr):
        if self.args.optimizer == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.5)
        return torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    def local_update(self, model, t, c):
        """Run local_ep epochs of local training from the global (model, t).

        Returns (t, state_dict, loss, ci). For FedeRage (`args.min == 'min'`) the
        returned (t, state_dict) is the iterate right after the step with the
        smallest observed objective, and loss is that objective; otherwise it
        is the last iterate and the mean training loss.
        """
        args = self.args
        model.train()
        optimizer = self._optimizer(model.parameters(), args.lr)
        optimizer_t = self._optimizer([t], args.lr_t)
        global_state = copy.deepcopy(model.state_dict())

        epoch_loss = []
        for _ in range(args.local_ep):
            batch_loss = []
            min_cvar_loss, min_param, min_t = 100000, None, None
            for images, labels in self.trainloader:
                model.zero_grad()
                loss = self.criterion(model(images), labels)

                if args.min == 'fedprox':
                    prox = 0
                    for param, global_param in zip(model.parameters(), global_state.values()):
                        prox += ((param - global_param) ** 2).sum()
                    loss += prox * args.mu / 2

                if args.min == 'scafold':
                    for (_, param), c_param, ci_param in zip(model.named_parameters(), c.values(), self.ci.values()):
                        if param.grad is not None:
                            param.grad.data += (ci_param - c_param)

                cvar_loss = mean_cvar_loss(loss, t, args.alpha, args.gamma)
                optimizer.zero_grad()
                optimizer_t.zero_grad()
                cvar_loss.backward()
                optimizer.step()
                optimizer_t.step()

                if cvar_loss.item() < min_cvar_loss:
                    min_cvar_loss = cvar_loss.item()
                    min_param = copy.deepcopy(model.state_dict())
                    min_t = copy.deepcopy(t)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if args.min == 'scafold':
            for name, param in model.named_parameters():
                self.ci[name] += (1 / (args.local_ep * args.lr)) * (param.data - global_state[name]) - c[name]
        if args.min == 'min':
            return min_t, min_param, min_cvar_loss, self.ci
        return t.item(), model.state_dict(), sum(epoch_loss) / len(epoch_loss), self.ci

    @torch.no_grad()
    def _accuracy(self, model, loader):
        model.eval()
        loss, total, correct = 0.0, 0, 0
        for images, labels in loader:
            outputs = model(images)
            loss += self.criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            total += len(labels)
        return correct / total, loss

    def evaluate(self, model):
        """Accuracy and loss on the client's validation split."""
        return self._accuracy(model, self.validloader)

    def test(self, model):
        """Accuracy and loss on the client's test split."""
        return self._accuracy(model, self.testloader)
