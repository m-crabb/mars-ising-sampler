# Critical Ising neural sampler

Submission repository for the MARS V prompt:

> Write a model that learns to approximately sample an N x N Ising lattice at
> critical temperature. Explain how to check whether its outputs are a
> reasonable sampler, and assign a numerical score.

This repo uses `N = 10`. The target is the square-lattice Ising model with
periodic boundary conditions. The code represents the lattice by the same
adjacency-matrix convention used during training:

```text
p(x) proportional to exp(sigma * x^T A x + bias * sum_i x_i),
x_i in {-1, +1}.
```

Here `A` is the symmetric nearest-neighbour adjacency matrix on the torus, so
`x^T A x` counts each edge twice. In textbook notation this is equivalent to
`exp(beta * sum_<ij> x_i x_j)` with `beta = 2 * sigma`. The trained
checkpoints use `sigma = 0.22305`, i.e. `beta = 0.44610`, the critical setting
used for the experiments here.

## Models

The main model is a locally equivariant transformer (`letf`). Two convolutional
models are included as ablations:

| model key | architecture | role |
|---|---|---|
| `letf` | hollow locally equivariant transformer | main sampler |
| `conv_global` | deep hollow convolution with leave-one-out global context | comparison model |
| `conv_local` | same conv model without global context | ablation |

Each model defines a time-dependent continuous-time Markov chain (CTMC). Starting
from random independent spins, the learned CTMC is simulated from time `t=0` to
`t=1` to produce approximate Ising samples.

## Numerical check

The primary score is effective sample size (ESS) from CTMC importance weights.
If all importance weights are equal, ESS equals the number of samples. If a
single trajectory dominates, ESS is near 1. We report `ESS / n_samples`.

The evaluation script also compares energy and magnetisation histograms against
a Wolff MCMC reference chain. I use Wolff rather than a single-spin Gibbs or
Metropolis-Hastings chain because local updates mix slowly near the critical
point; cluster flips give a more useful empirical reference at the same compute
budget.

Precomputed run summaries from the training run:

| model | ESS / 5000 | ESS fraction | internal energy per site |
|---|---:|---:|---:|
| `letf` | 3653.6 | 0.731 | -1.5134 |
| `conv_global` | 2416.9 | 0.483 | -1.4736 |
| `conv_local` | 1462.7 | 0.293 | -1.4935 |

## Usage

Install dependencies with Pixi:

```bash
pixi install
```

or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Evaluate one checkpoint:

```bash
python -m ising.evaluate --model letf --checkpoint checkpoints/letf_critical.pt
```

Evaluate all shipped checkpoints and create overlay plots:

```bash
pixi run evaluate
# or: python -m ising.evaluate --model all
```

Train from scratch, for example the transformer curriculum:

```bash
python -m ising.train --config letf --output-dir runs/letf
```

The full transformer curriculum is a long run. For code review, use
`pixi run train-letf-debug`, which deliberately shrinks the batch sizes and
Euler grid to exercise the training loop quickly. For the reported result, use
the shipped checkpoint and `results/summary_metrics.json`.

## Acknowledgements

This implementation is based on the discrete neural flow sampler idea from
Ou, Zhang and Li, with architectural inspiration from locally equivariant
networks and LEAPS-style convolutional samplers.
