"""Scalar diagnostics reported in the README and writeup."""

import torch
from torch import Tensor


def ess_from_log_weights(log_weights: Tensor) -> Tensor:
    """Effective sample size from log importance weights."""
    log_sum_w = torch.logsumexp(log_weights, dim=0)
    log_sum_w_sq = torch.logsumexp(2.0 * log_weights, dim=0)
    return torch.exp(2.0 * log_sum_w - log_sum_w_sq)


def free_energy_per_site(log_weights: Tensor, beta: float, n_sites: int) -> Tensor:
    """Free-energy lower-bound estimate per spin."""
    return -log_weights.mean() / (beta * n_sites)


def internal_energy_per_site(log_weights: Tensor, log_prob: Tensor, beta: float, n_sites: int) -> Tensor:
    """Self-normalised importance estimate of internal energy per spin."""
    weights = torch.softmax(log_weights, dim=0)
    return -(weights * log_prob).sum() / (beta * n_sites)


def entropy_per_site(f_per_site: Tensor, e_per_site: Tensor, beta: float) -> Tensor:
    """Entropy per spin from S/N = beta * (E/N - F/N)."""
    return beta * (e_per_site - f_per_site)
