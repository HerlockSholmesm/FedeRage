"""Federated training under the random access model (Algorithms 1 and 2 of the paper).

    python -m federage.train --algorithm federage --dataset mnist --probs superexp \
        --lr 0.001 --lr_t 1e-5 --gamma 0.05 --alpha 0.01 --seed 0 --gpu 0 > run.txt

After a header with the configuration, every round prints one CSV line per client:

    user,eval_accuracy,eval_loss,test_accuracy,test_loss,epoch

evaluated with the global model on that client's validation and test splits.
"""

import copy

import numpy as np
import torch
from tqdm import tqdm

from .availability import availability_profile
from .client import Client
from .data import get_dataset
from .models import build_model
from .options import args_parser, exp_details


def average_weights(states):
    avg = copy.deepcopy(states[0])
    for key in avg.keys():
        for s in states[1:]:
            avg[key] += s[key]
        avg[key] = torch.div(avg[key], len(states))
    return avg


def main(argv=None):
    args = args_parser(argv)
    exp_details(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = f'cuda:{args.gpu}' if args.gpu > -1 else 'cpu'
    if args.gpu > -1:
        torch.cuda.set_device(device)

    # Participation probabilities of the random access model; unknown to the algorithms.
    user_probs = availability_profile(args.probs, args.num_users)
    train_dataset, test_dataset, user_groups = get_dataset(args)
    print('user,eval_accuracy,eval_loss,test_accuracy,test_loss,epoch')

    global_model = build_model(args, train_dataset).to(device)
    global_model.train()
    global_t = torch.tensor(0., requires_grad=True, device=device)
    global_c = {name: torch.zeros_like(p.data, device=device) for name, p in global_model.named_parameters()}

    clients = [Client(args, train_dataset, user_groups[i], global_model) for i in range(args.num_users)]
    m = max(int(args.frac * args.num_users), 1)

    for epoch in tqdm(range(args.epochs)):
        global_model.train()
        selected = np.random.choice(range(args.num_users), m, p=user_probs, replace=False)

        local_t, local_weights, local_cs = [], [], []
        for idx in selected:
            t, w, _, ci = clients[idx].local_update(copy.deepcopy(global_model), copy.deepcopy(global_t), c=global_c)
            local_t.append(copy.deepcopy(t))
            local_weights.append(copy.deepcopy(w))
            local_cs.append(copy.deepcopy(ci))

        # Server: uniform average over the received clients.
        global_model.load_state_dict(average_weights(local_weights))
        global_t = torch.tensor(float(sum(local_t) / len(local_t)), requires_grad=True, device=device)
        if args.min == 'scafold':
            for name in global_c:
                global_c[name] += (1 / args.num_users) * (sum(ci[name] for ci in local_cs) / len(local_cs))

        global_model.eval()
        for c, client in enumerate(clients):
            acc, loss = client.evaluate(global_model)
            acc_t, loss_t = client.test(global_model)
            print(f'{c},{acc},{loss},{acc_t},{loss_t},{epoch}')


if __name__ == '__main__':
    main()
