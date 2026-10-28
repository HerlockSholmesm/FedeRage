#!/usr/bin/env python
"""Figures 4 and 6 from the summaries in results/.

    python scripts/plot_figures.py [--results results/] [--out figures/]

fig4_<dataset>.png   client-averaged test accuracy vs. round, non-uniform availability
                     (mean over seeds, shaded +/- one standard deviation)
fig6_fmnist.png      per-client test accuracy over the last 10 rounds, FashionMNIST,
                     non-uniform availability (mean over seeds)
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'scaffold': 'SCAFFOLD', 'federage': 'FedeRage'}
TITLES = {'mnist': 'MNIST', 'fmnist': 'FashionMNIST', 'cifar': 'CIFAR-10'}


def fig4(curves, out, smooth=50):
    for dataset in TITLES:
        d = curves[(curves.dataset == dataset) & (curves.regime == 'nonuniform')]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for algo, label in LABELS.items():
            runs = d[d.algorithm == algo].pivot(index='round', columns='seed', values='mean_test_acc')
            runs = 100 * runs.rolling(smooth, min_periods=1).mean()
            m, s = runs.mean(axis=1), runs.std(axis=1)
            ax.plot(m.index, m, label=label, lw=1.2)
            ax.fill_between(m.index, m - s, m + s, alpha=0.2)
        ax.set(xlabel='Communication round', ylabel='Test accuracy (%)', title=f'{TITLES[dataset]} (non-uniform)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'fig4_{dataset}.png'), dpi=200)
        plt.close(fig)


def fig6(per_client, out, dataset='fmnist'):
    d = per_client[(per_client.dataset == dataset) & (per_client.regime == 'nonuniform')]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for algo, label in LABELS.items():
        acc = 100 * d[d.algorithm == algo].groupby('client').test_acc.mean()
        line, = ax.plot(acc.index, acc.values, marker='o', ms=3, lw=1, label=label)
        ax.axhline(acc.mean(), ls='--', lw=0.8, color=line.get_color())
    ax.set(xlabel='Client ID', ylabel='Test accuracy (%)', title=f'{TITLES[dataset]}, non-uniform availability')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'fig6_{dataset}.png'), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default=os.path.join(ROOT, 'results'))
    p.add_argument('--out', default=os.path.join(ROOT, 'figures'))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig4(pd.read_csv(os.path.join(args.results, 'curves.csv.gz')), args.out)
    fig6(pd.read_csv(os.path.join(args.results, 'per_client.csv')), args.out)


if __name__ == '__main__':
    main()
