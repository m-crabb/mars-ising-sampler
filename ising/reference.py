"""Reference observables for checking the neural sampler.

Wolff cluster updates are used for histograms because local Gibbs or
Metropolis-Hastings chains mix slowly near criticality. The neural sampler is
not trained against these samples; they are only a sanity-check reference for
energy and magnetisation distributions.
"""

import numpy as np


def wolff_samples(
    lattice_size: int,
    beta: float,
    n_samples: int,
    burn_in: int = 2000,
    thin: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Generate approximately independent Ising samples with Wolff updates."""
    rng = np.random.default_rng(seed)
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(lattice_size, lattice_size))
    p_add = 1.0 - np.exp(-2.0 * beta)
    out = []
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        _wolff_flip(spins, p_add, rng)
        if step >= burn_in and (step - burn_in) % thin == 0:
            out.append(spins.reshape(-1).copy())
    return np.stack(out, axis=0).astype(np.float32)


def _wolff_flip(spins: np.ndarray, p_add: float, rng: np.random.Generator) -> None:
    """Flip one Fortuin-Kasteleyn cluster in place."""
    n = spins.shape[0]
    r = int(rng.integers(0, n))
    c = int(rng.integers(0, n))
    spin_value = spins[r, c]
    in_cluster = np.zeros_like(spins, dtype=bool)
    in_cluster[r, c] = True
    stack = [(r, c)]
    while stack:
        rr, cc = stack.pop()
        for nr, nc in ((rr - 1) % n, cc), ((rr + 1) % n, cc), (rr, (cc - 1) % n), (rr, (cc + 1) % n):
            if not in_cluster[nr, nc] and spins[nr, nc] == spin_value and rng.random() < p_add:
                in_cluster[nr, nc] = True
                stack.append((nr, nc))
    spins[in_cluster] *= -1


def energy_per_site(samples: np.ndarray, lattice_size: int) -> np.ndarray:
    """Physics energy H/N = -sum_<ij> s_i s_j / N."""
    grid = samples.reshape(-1, lattice_size, lattice_size)
    edge_sum = (grid * np.roll(grid, -1, axis=2)).sum(axis=(1, 2))
    edge_sum += (grid * np.roll(grid, -1, axis=1)).sum(axis=(1, 2))
    return -edge_sum / (lattice_size * lattice_size)


def magnetisation(samples: np.ndarray) -> np.ndarray:
    """Mean spin per lattice site for each sample."""
    return samples.mean(axis=1)
