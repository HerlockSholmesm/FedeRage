#!/usr/bin/env python
"""Launch the Table I experiments (configs/table1.csv x seeds), one GPU per job.

    python scripts/run_table1.py --gpus 0 1 2 --out logs/          # run everything
    python scripts/run_table1.py --dry_run                         # print the commands
    python scripts/run_table1.py --dataset mnist --algorithm federage --seeds 0

Each run writes logs/<dataset>-<probs>-<algorithm>-<seed>.txt, the format read by
scripts/summarize_logs.py. A full run is 10,000 rounds; expect hours per job on a GPU.
"""

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_NAMES = {'fedavg': 'nomin', 'fedprox': 'fedprox', 'scaffold': 'scafold', 'federage': 'min'}


def commands(args):
    with open(os.path.join(ROOT, 'configs', 'table1.csv')) as f:
        for cfg in csv.DictReader(f):
            if args.dataset and cfg['dataset'] not in args.dataset:
                continue
            if args.algorithm and cfg['algorithm'] not in args.algorithm:
                continue
            if args.regime and cfg['regime'] not in args.regime:
                continue
            for seed in args.seeds:
                cmd = [sys.executable, '-m', 'federage.train',
                       '--algorithm', cfg['algorithm'], '--dataset', cfg['dataset'], '--model', cfg['model'],
                       '--probs', cfg['probs'], '--iid', '3', '--num_users', '30', '--frac', cfg['frac'],
                       '--local_ep', cfg['local_ep'], '--local_bs', cfg['local_bs'], '--lr', cfg['lr'],
                       '--lr_t', cfg['lr_t'], '--gamma', cfg['gamma'], '--alpha', cfg['alpha'], '--mu', cfg['mu'],
                       '--epochs', str(args.rounds), '--seed', str(seed), '--data_dir', args.data_dir]
                name = f"{cfg['dataset']}-{cfg['probs']}-{LOG_NAMES[cfg['algorithm']]}-{seed}.txt"
                yield cmd, os.path.join(args.out, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', type=int, nargs='+', default=[0])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--dataset', nargs='+', choices=['mnist', 'fmnist', 'cifar'])
    p.add_argument('--algorithm', nargs='+', choices=list(LOG_NAMES))
    p.add_argument('--regime', nargs='+', choices=['uniform', 'nonuniform'])
    p.add_argument('--rounds', type=int, default=10000)
    p.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    p.add_argument('--out', default=os.path.join(ROOT, 'logs'))
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    jobs = list(commands(args))
    if args.dry_run:
        for cmd, log in jobs:
            print('python', shlex.join(cmd[1:]), '>', shlex.quote(log))
        return

    os.makedirs(args.out, exist_ok=True)
    running = {}  # gpu -> (process, log file)
    while jobs or running:
        for gpu in args.gpus:
            proc = running.get(gpu)
            if proc and proc[0].poll() is not None:
                proc[1].close()
                del running[gpu]
            if gpu not in running and jobs:
                cmd, log = jobs.pop(0)
                print(f'[gpu {gpu}] {log}', flush=True)
                fh = open(log, 'w')
                running[gpu] = (subprocess.Popen(cmd + ['--gpu', str(gpu)], cwd=ROOT, stdout=fh,
                                                 stderr=subprocess.DEVNULL), fh)
        time.sleep(5)


if __name__ == '__main__':
    main()
