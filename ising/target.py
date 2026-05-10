"""Critical Ising target used by the learned CTMC samplers.

The target is written as log p(x) = sigma * x^T A x + bias * sum_i x_i, where A is the
symmetric nearest-neighbour adjacency matrix on the periodic square lattice. Because
x^T A x counts each undirected edge twice, the usual physics inverse temperature is beta = 2 * sigma.
"""

import math

import torch
from torch import Tensor


class IsingTarget:
    """Periodic-boundary D x D Ising model with a linear log-density path."""

    def __init__(
        self,
        D: int,
        sigma: float,
        bias: float = 0.0,
        device: torch.device | str = "cpu",
    ):
        self.D = D
        self.d = D * D
        self.sigma = sigma
        self.bias = bias
        self.device = torch.device(device)

        # Build the torus adjacency matrix with right/down edges, then
        # symmetrise to match the training convention.
        A = torch.zeros((self.d, self.d), device=self.device)

        for r in range(self.D):
            for c in range(self.D):
                i = r * self.D + c
                right = r * self.D + (c + 1) % self.D
                down = ((r + 1) % self.D) * self.D + c
                A[i, right] = 1.0
                A[i, down] = 1.0

        A = A + A.T
        self.A = A
        self.J = self.sigma * A

    def set_sigma(self, sigma: float) -> None:
        """Update the Ising coupling during the temperature curriculum."""
        self.sigma = sigma
        self.J = sigma * self.A

    def log_prob(self, x: Tensor) -> Tensor:
        """Unnormalised log-density at t=1 for spins in {-1, +1}."""
        return (x @ self.J * x).sum(dim=-1) + (self.bias * x.sum(dim=1))

    def log_p_tilde_t(self, x: Tensor, t: Tensor) -> Tensor:
        """Annealing path from the uniform distribution to the Ising target."""
        return (1 - t) * (-self.d * math.log(2)) + (t * self.log_prob(x))

    def dt_log_p_tilde_t(self, x: Tensor, t: Tensor) -> Tensor:
        """Time derivative of the unnormalised annealing log-density."""
        return self.log_prob(x) + (self.d * math.log(2))


def sample_uniform_spins(
    n_samples: int,
    n_sites: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw independent uniform spins in {-1, +1}."""
    return (
        torch.randint(0, 2, (n_samples, n_sites), generator=generator, device=device)
        .float()
        .mul(2.0)
        .sub(1.0)
    )
