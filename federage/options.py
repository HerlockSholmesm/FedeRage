import argparse

# Internal names used in the published logs (and in results/runs.csv).
ALGORITHMS = {'fedavg': 'nomin', 'fedprox': 'fedprox', 'scaffold': 'scafold', 'federage': 'min'}


def args_parser(argv=None):
    p = argparse.ArgumentParser(description='Federated training with FedAvg, FedProx, SCAFFOLD or FedeRage.')

    # federated setting
    p.add_argument('--algorithm', choices=list(ALGORITHMS), default='federage')
    p.add_argument('--epochs', type=int, default=10000, help='number of communication rounds T')
    p.add_argument('--num_users', type=int, default=30, help='number of clients N')
    p.add_argument('--frac', type=float, default=0.1, help='fraction of clients selected per round (M = frac * N)')
    p.add_argument('--local_ep', type=int, default=10, help='local epochs per round')
    p.add_argument('--local_bs', type=int, default=100, help='local mini-batch size b')
    p.add_argument('--lr', type=float, default=0.001, help='step size for the model parameters')
    p.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')

    # FedeRage (mean-CVaR) parameters
    p.add_argument('--alpha', type=float, default=1.0, help='CVaR level alpha (1.0 disables CVaR)')
    p.add_argument('--gamma', type=float, default=1.0,
                   help='weight on the plain loss; the CVaR term gets weight 1 - gamma. 1.0 disables CVaR.')
    p.add_argument('--lr_t', type=float, default=1e-5, help='step size for the threshold beta (called t in the code)')

    # baselines
    p.add_argument('--mu', type=float, default=1.0, help='FedProx proximal coefficient')

    # data and availability
    p.add_argument('--dataset', choices=['mnist', 'fmnist', 'cifar', 'blobs'], default='mnist')
    p.add_argument('--data_dir', default='./data', help='where torchvision datasets are stored / downloaded')
    p.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--num_channels', type=int, default=1)
    p.add_argument('--iid', type=int, default=3,
                   help='client data partition: 0 = IID, 1 = two shards per client, '
                        '3 = two shards per client with the rarest classes on the least available clients.')
    p.add_argument('--probs', choices=['uni', 'lin', 'exp', 'supexp', 'superexp', 'blobs'], default='uni',
                   help='client availability profile (see federage/availability.py)')

    # misc
    p.add_argument('--gpu', type=int, default=-1, help='CUDA device id, -1 for CPU')
    p.add_argument('--seed', type=int, default=0)

    args = p.parse_args(argv)
    args.min = ALGORITHMS[args.algorithm]
    return args


def exp_details(args):
    # The format is parsed by scripts/summarize_logs.py; keep it stable.
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print('    Federated parameters:')
    print(f'    Data partition : {args.iid}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    print(f'    Probabilites are random: {args.probs}\n')
    print(f'    Learning rate for t: {args.lr_t}\n')
    print(f'    Gamma is:   {args.gamma}\n')
    print(f'    alpha is {args.alpha}\n')
    print(f'    minimum is return or not {args.min}\n')
    print(f'    mu is {args.mu}\n')
