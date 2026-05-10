"""CTMC simulation and training objective for the neural sampler.

The model outputs an antisymmetric local score G(tau, i | x, t). Positive
entries become transition rates for changing site i to token tau. The same
local scores also give the importance-weight integrand used for ESS.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def log_p_tilde_at_neighbours(x: Tensor, t: Tensor, target, vocab_size: int) -> Tensor:
    """Evaluate log p_tilde_t at every one-site replacement of each state."""
    batch, n_sites = x.shape
    spin_of_idx = 2.0 * torch.arange(vocab_size, device=x.device, dtype=x.dtype) - 1.0
    # Shape: (batch, site_to_replace, replacement_token, full_state).
    neighbours = x[:, None, None, :].expand(batch, n_sites, vocab_size, n_sites).clone()
    site_index = torch.arange(n_sites, device=x.device).view(1, n_sites, 1, 1)
    site_index = site_index.expand(batch, n_sites, vocab_size, 1)
    spin_at_site = spin_of_idx.view(1, 1, vocab_size, 1).expand(batch, n_sites, vocab_size, 1)
    neighbours.scatter_(-1, site_index, spin_at_site)
    flat = neighbours.reshape(batch * n_sites * vocab_size, n_sites)
    t_per = t.repeat_interleave(n_sites * vocab_size)
    return target.log_p_tilde_t(flat, t_per).reshape(batch, n_sites, vocab_size)


def compute_xi_t(state: Tensor, t: Tensor, model, target) -> Tensor:
    """Importance-weight integrand xi_t for locally equivariant models."""
    g_t = model(state, t)
    g_plus = F.relu(g_t)
    neg_g_plus = F.relu(-g_t)
    log_p_neighbours = log_p_tilde_at_neighbours(state, t, target, model.vocab_size)
    log_p_x = target.log_p_tilde_t(state, t)
    # This clamp follows the reference DNFS implementation and avoids very
    # large neighbour ratios dominating early training.
    log_ratio = (log_p_neighbours - log_p_x[:, None, None]).clamp(max=5.0)
    outflow = g_plus.sum(dim=(-2, -1))
    inflow = (neg_g_plus * log_ratio.exp()).sum(dim=(-2, -1))
    return target.dt_log_p_tilde_t(state, t) + outflow - inflow


def euler_step(model, state: Tensor, t: Tensor, step_dt: Tensor) -> Tensor:
    """One forward-Euler CTMC step with independent per-site categoricals."""
    g_t = model(state, t)
    rates = F.relu(g_t)
    step_probs = (rates * step_dt).clamp(0.0, 1.0)
    stay_prob = (1.0 - step_probs.sum(dim=-1)).clamp(0.0, 1.0)
    cat_probs = torch.cat([step_probs, stay_prob.unsqueeze(-1)], dim=-1)
    batch, n_sites = state.shape
    vocab_size = g_t.shape[-1]
    sampled_idx = torch.multinomial(
        cat_probs.reshape(batch * n_sites, vocab_size + 1),
        num_samples=1,
    ).reshape(batch, n_sites)
    spin_of_idx = 2.0 * torch.arange(vocab_size, device=state.device, dtype=state.dtype) - 1.0
    stay_mask = sampled_idx == vocab_size
    return torch.where(stay_mask, state, spin_of_idx[sampled_idx.clamp(max=vocab_size - 1)])


def sample_ctmc(
    model,
    x0: Tensor,
    ts: Tensor,
    *,
    target=None,
    return_log_weights: bool = False,
    return_all_states: bool = False,
):
    """Simulate the learned CTMC from x0 along time grid ts.

    If return_log_weights is true, the function also accumulates the log
    importance weights used by the ESS and free-energy diagnostics.
    """
    if return_log_weights and target is None:
        raise ValueError("target is required when return_log_weights=True")
    if return_log_weights and return_all_states:
        raise ValueError("return_log_weights and return_all_states are mutually exclusive")

    state = x0.clone()
    batch, n_sites = state.shape
    log_weights = torch.zeros(batch, dtype=state.dtype, device=state.device)
    if return_all_states:
        trajectory = torch.empty((len(ts), batch, n_sites), dtype=state.dtype, device=state.device)
        trajectory[0] = state

    for step in range(len(ts) - 1):
        t_curr = ts[step].expand(batch)
        step_dt = ts[step + 1] - ts[step]
        if return_log_weights:
            log_weights = log_weights + compute_xi_t(state, t_curr, model, target) * step_dt
        state = euler_step(model, state, t_curr, step_dt)
        if return_all_states:
            trajectory[step + 1] = state

    if return_log_weights:
        return state, log_weights
    if return_all_states:
        return trajectory
    return state


def kolmogorov_loss(x: Tensor, t: Tensor, c_t: Tensor, model, target) -> Tensor:
    """Squared Kolmogorov residual used to train the rate model."""
    residual = compute_xi_t(x, t, model, target) - c_t
    residual = residual.nan_to_num(posinf=1.0, neginf=-1.0, nan=0.0)
    return residual.pow(2).mean()


def compute_c_t_grid(t_grid: Tensor, x_traj: Tensor, target, model) -> tuple[Tensor, Tensor]:
    """Estimate the per-time normalising-flow control variate c_t."""
    n_grid, outer_batch, _ = x_traj.shape
    integrand = torch.empty((n_grid, outer_batch), dtype=x_traj.dtype, device=x_traj.device)
    with torch.no_grad():
        for k in range(n_grid):
            t_k = t_grid[k].expand(outer_batch)
            integrand[k] = compute_xi_t(x_traj[k], t_k, model, target)
    return integrand.mean(dim=-1), integrand
