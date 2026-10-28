#!/usr/bin/env python
"""Table I from results/curves.csv.gz.

For each (dataset, regime, algorithm): mean and standard deviation of the
client-averaged test accuracy over the last 10 of the first 10,000 rounds,
pooled over the 5 seeds (50 values).

    python scripts/make_table.py [--results results/]
"""

import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGORITHMS = ['fedavg', 'fedprox', 'scaffold', 'federage']
COLUMNS = [(d, r) for d in ['mnist', 'fmnist', 'cifar'] for r in ['uniform', 'nonuniform']]


def table(results, last=10):
    c = pd.read_csv(os.path.join(results, 'curves.csv.gz'))
    final = c.groupby(['dataset', 'regime', 'algorithm', 'seed'])['round'].transform('max')
    tail = c[c['round'] > final - last]
    g = tail.groupby(['dataset', 'regime', 'algorithm'])['mean_test_acc']
    return 100 * g.mean(), 100 * g.std()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default=os.path.join(ROOT, 'results'))
    args = p.parse_args()
    mean, std = table(args.results)

    header = ['algorithm'] + [f'{d} {r}' for d, r in COLUMNS]
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for a in ALGORITHMS:
        cells = [f'{mean[(d, r, a)]:.2f} ± {std[(d, r, a)]:.2f}' for d, r in COLUMNS]
        print('| ' + ' | '.join([a] + cells) + ' |')


if __name__ == '__main__':
    main()
