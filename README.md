# FedeRage

Code for **FedeRage: Provably Convergent Agnostic Federated Learning under General Client Drift**
by Herlock (SeyedAbolfazl) Rahimi and Dionysis Kalogerias.

FedeRage (Federated Risk-Averse Averaging) is FedAvg with a mean-CVaR local objective. Each client
keeps one extra scalar, the threshold β, which the server averages along with the model. The
algorithm never uses the clients' participation probabilities. This repository contains the
implementation of FedeRage and of the FedAvg, FedProx and SCAFFOLD baselines, the configurations
of the paper's experiments, and summaries of the runs reported in Table I and Figs. 4 and 6.

## Setup

```bash
git clone https://github.com/HerlockSholmesm/FedeRage.git
cd FedeRage
pip install -r requirements.txt
```

The published runs used Python 3.8, PyTorch 2.3.1 and torchvision 0.18.1 on NVIDIA A6000 GPUs.
MNIST, FashionMNIST and CIFAR-10 are downloaded to `./data` on first use.

## Running

```bash
python -m federage.train --algorithm federage --dataset mnist --probs superexp \
    --lr 0.001 --lr_t 1e-5 --gamma 0.05 --alpha 0.01 --epochs 10000 --seed 0 --gpu 0 > run.txt
```

| flag | meaning |
|---|---|
| `--algorithm` | `fedavg`, `fedprox`, `scaffold` or `federage` |
| `--dataset`, `--model` | `mnist` / `fmnist` with `mlp`, `cifar` with `cnn`, `blobs` (Fig. 1) with `mlp` |
| `--probs` | client availability: `uni` (the paper's "uniform" regime) or `superexp` (the "non-uniform" regime of Fig. 5) |
| `--num_users`, `--frac` | N clients, M = frac · N selected per round (30 and 0.1 in the paper) |
| `--epochs`, `--local_ep`, `--local_bs` | communication rounds, local epochs per round, mini-batch size |
| `--lr`, `--lr_t` | step sizes for the model and for the threshold β |
| `--alpha` | CVaR level α |
| `--gamma` | weight on the plain loss (the CVaR term is weighted by 1 − `--gamma`) |
| `--mu` | FedProx proximal coefficient |

Every round, the run prints `user,eval_accuracy,eval_loss,test_accuracy,test_loss,epoch` for each
client. These are the accuracy and loss of the global model on that client's validation and test splits.

## Reproducing Table I and Figs. 4 and 6

`configs/table1.csv` lists the hyperparameters of every Table I configuration. These were read from
the headers of the original logs. Each configuration was run with seeds 0 to 4.

```bash
python scripts/run_table1.py --gpus 0 1 2 --out logs/        # 120 runs of 10,000 rounds
python scripts/summarize_logs.py --logs logs/*.txt --out results/
python scripts/make_table.py
python scripts/plot_figures.py
```

`results/` already contains the summaries of the original runs:

- `curves.csv.gz`: for every run and round, the mean and minimum test accuracy over clients
- `per_client.csv`: every client's accuracy, averaged over the last 10 rounds
- `runs.csv`: each run's configuration

You can therefore regenerate the table and figures without retraining. `make_table.py` reports the
mean ± standard deviation of the client-averaged test accuracy over the last 10 of 10,000 rounds,
pooled over the 5 seeds:

| algorithm | MNIST uniform | MNIST non-uniform | FMNIST uniform | FMNIST non-uniform | CIFAR-10 uniform | CIFAR-10 non-uniform |
|---|---|---|---|---|---|---|
| FedAvg | 81.27 ± 6.41 | 78.33 ± 10.20 | 74.80 ± 4.81 | 61.53 ± 9.08 | 51.58 ± 1.89 | 50.61 ± 1.30 |
| FedProx | 81.83 ± 7.06 | 78.04 ± 10.10 | 74.81 ± 4.81 | 61.53 ± 9.08 | 52.18 ± 0.77 | 50.27 ± 1.54 |
| SCAFFOLD | 96.95 ± 0.24 | 95.06 ± 0.55 | 83.93 ± 2.35 | 75.53 ± 2.48 | 51.18 ± 2.45 | 50.54 ± 1.28 |
| FedeRage | 96.10 ± 0.45 | 95.92 ± 0.39 | 84.51 ± 1.25 | 84.16 ± 1.02 | 56.84 ± 1.78 | 56.42 ± 0.99 |

## Acknowledgements

Built on [AshwinRJ/Federated-Learning-PyTorch](https://github.com/AshwinRJ/Federated-Learning-PyTorch) (MIT).
This work was supported by the US National Science Foundation under Grants 2242215 and 2431860.

## Citation

```bibtex
@article{rahimi_federage,
  title  = {{FedeRage}: Provably Convergent Agnostic Federated Learning under General Client Drift},
  author = {Rahimi, Herlock (SeyedAbolfazl) and Kalogerias, Dionysis},
}
```
