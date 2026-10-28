#!/usr/bin/env python
"""Compress raw training logs into the small summary files under results/.

Each log is the stdout of one `python -m federage.train` run: a header with
the run configuration followed by one CSV line per client per round,

    user,eval_accuracy,eval_loss,test_accuracy,test_loss,epoch

Usage:
    python scripts/summarize_logs.py --logs LOG [LOG ...] --out results/ --max_rounds 10000

Writes
    curves.csv.gz      one row per run and round: mean / min accuracy over clients
    per_client.csv     one row per run and client: accuracy over the last 10 rounds
    runs.csv           one row per run: configuration parsed from the log header
"""

import argparse
import csv
import gzip
import os
import re

import numpy as np

HEADER_FIELDS = {
    'model': r'Model\s*:\s*(\S+)',
    'lr': r'Learning\s*:\s*(\S+)',
    'rounds': r'Global Rounds\s*:\s*(\S+)',
    'frac': r'Fraction of users\s*:\s*(\S+)',
    'local_bs': r'Local Batch size\s*:\s*(\S+)',
    'local_ep': r'Local Epochs\s*:\s*(\S+)',
    'probs': r'Probabilites are random:\s*(\S+)',
    'lr_t': r'Learning rate for t:\s*(\S+)',
    'gamma': r'Gamma is:\s*(\S+)',
    'alpha': r'alpha is\s*(\S+)',
    'algorithm': r'minimum is return or not\s*(\S+)',
    'mu': r'mu is\s*(\S+)',
}
LINE = re.compile(r'^(\d+),([0-9.eE+-]+),([0-9.eE+-]+),([0-9.eE+-]+),([0-9.eE+-]+),(\d+)$')
DATASET_NAMES = {'mnist': 'mnist', 'fmnist': 'fmnist', 'cifar': 'cifar'}
ALGORITHM_NAMES = {'nomin': 'fedavg', 'fedprox': 'fedprox', 'scafold': 'scaffold', 'min': 'federage'}
REGIME_NAMES = {'uni': 'uniform', 'superexp': 'nonuniform'}


def parse_log(path, max_rounds):
    config, rows = {}, []
    with open(path) as f:
        head = []
        for line in f:
            m = LINE.match(line.strip())
            if m is None:
                if not rows:
                    head.append(line)
                continue
            user, eval_acc, _, test_acc, _, rnd = m.groups()
            if int(rnd) >= max_rounds:
                break
            rows.append((int(rnd), int(user), float(eval_acc), float(test_acc)))
    text = ''.join(head)
    for key, pattern in HEADER_FIELDS.items():
        m = re.search(pattern, text)
        config[key] = m.group(1) if m else ''
    return config, np.array(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max_rounds', type=int, default=10000)
    p.add_argument('--last', type=int, default=10, help='rounds averaged for per_client.csv')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with gzip.open(os.path.join(args.out, 'curves.csv.gz'), 'wt', newline='') as fc, \
            open(os.path.join(args.out, 'per_client.csv'), 'w', newline='') as fp, \
            open(os.path.join(args.out, 'runs.csv'), 'w', newline='') as fr:
        curves, per_client, runs = csv.writer(fc), csv.writer(fp), csv.writer(fr)
        key_cols = ['dataset', 'regime', 'algorithm', 'seed']
        curves.writerow(key_cols + ['round', 'mean_test_acc', 'min_test_acc', 'mean_val_acc'])
        per_client.writerow(key_cols + ['client', 'test_acc', 'val_acc'])
        runs.writerow(key_cols + ['rounds_logged'] + list(HEADER_FIELDS))

        for path in sorted(args.logs):
            # file names look like <dataset>-<probs>-<algorithm>-<seed>[-<lr>].txt
            dataset, probs, algo, seed = os.path.basename(path)[:-4].split('-')[:4]
            key = [DATASET_NAMES[dataset], REGIME_NAMES[probs], ALGORITHM_NAMES[algo], seed]
            config, rows = parse_log(path, args.max_rounds)
            num_clients = int(rows[:, 1].max()) + 1
            # keep only rounds where every client was logged (a run may stop mid-round)
            counts = np.bincount(rows[:, 0].astype(int))
            complete = np.flatnonzero(counts == num_clients)
            rows = rows[np.isin(rows[:, 0], complete)]
            rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))].reshape(len(complete), num_clients, 4)
            runs.writerow(key + [len(complete)] + [config[k] for k in HEADER_FIELDS])

            mean_test, min_test, mean_val = rows[:, :, 3].mean(1), rows[:, :, 3].min(1), rows[:, :, 2].mean(1)
            for r, a, b, c in zip(complete, mean_test, min_test, mean_val):
                curves.writerow(key + [r, f'{a:.5f}', f'{b:.5f}', f'{c:.5f}'])
            tail = rows[-args.last:]
            for c in range(num_clients):
                per_client.writerow(key + [c, f'{tail[:, c, 3].mean():.5f}', f'{tail[:, c, 2].mean():.5f}'])
            print(f'{path}: {len(complete)} rounds')


if __name__ == '__main__':
    main()
