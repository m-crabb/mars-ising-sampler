"""Small set of submission configs and checkpoint locations."""

from dataclasses import dataclass
from pathlib import Path

from .models import LeConvDeepRateMatrix, LeTFRateMatrix


SIGMA_CRITICAL = 0.22305
BETA_CRITICAL = 2.0 * SIGMA_CRITICAL


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters needed for evaluation and optional retraining."""

    key: str
    label: str
    checkpoint: Path
    n_euler_steps: int
    curriculum: tuple[tuple[int, float, float], ...]
    n_steps: int
    batch_size: int
    outer_batch_size: int
    replay_buffer_cycles: int
    lr: float
    grad_clip: float


MODEL_CONFIGS = {
    "letf": ModelConfig(
        key="letf",
        label="locally equivariant transformer",
        checkpoint=Path("checkpoints/letf_critical.pt"),
        n_euler_steps=64,
        curriculum=(
            (0, 0.100, 1e-3),
            (10_000, 0.140, 1e-3),
            (20_000, 0.170, 1e-3),
            (30_000, 0.190, 1e-3),
            (40_000, 0.205, 1e-3),
            (55_000, 0.215, 5e-4),
            (70_000, 0.220, 5e-4),
            (85_000, SIGMA_CRITICAL, 3e-4),
        ),
        n_steps=200_000,
        batch_size=128,
        outer_batch_size=256,
        replay_buffer_cycles=4,
        lr=1e-3,
        grad_clip=500.0,
    ),
    "conv_global": ModelConfig(
        key="conv_global",
        label="deep convolution with global context",
        checkpoint=Path("checkpoints/conv_global_critical.pt"),
        n_euler_steps=100,
        curriculum=(
            (0, 0.100, 1e-3),
            (5_000, 0.140, 1e-3),
            (10_000, 0.170, 1e-3),
            (15_000, 0.190, 1e-3),
            (20_000, 0.205, 3e-4),
            (25_000, 0.215, 3e-4),
            (30_000, SIGMA_CRITICAL, 3e-4),
        ),
        n_steps=50_000,
        batch_size=256,
        outer_batch_size=256,
        replay_buffer_cycles=4,
        lr=1e-3,
        grad_clip=500.0,
    ),
    "conv_local": ModelConfig(
        key="conv_local",
        label="deep convolution without global context",
        checkpoint=Path("checkpoints/conv_local_critical.pt"),
        n_euler_steps=100,
        curriculum=(
            (0, 0.100, 1e-3),
            (5_000, 0.140, 1e-3),
            (10_000, 0.170, 1e-3),
            (15_000, 0.190, 1e-3),
            (20_000, 0.205, 3e-4),
            (25_000, 0.215, 3e-4),
            (30_000, SIGMA_CRITICAL, 3e-4),
        ),
        n_steps=50_000,
        batch_size=256,
        outer_batch_size=256,
        replay_buffer_cycles=4,
        lr=1e-3,
        grad_clip=500.0,
    ),
}


def build_model(model_key: str):
    """Instantiate the architecture matching a shipped checkpoint."""
    if model_key == "letf":
        return LeTFRateMatrix(d=100, vocab_size=2, hidden_dim=128, n_layers=3, n_heads=4)
    if model_key == "conv_global":
        return LeConvDeepRateMatrix(
            lattice_size=10,
            vocab_size=2,
            kernel_schedule=(3, 5, 7, 9, 15),
            hidden_dim=64,
            use_global_context=True,
        )
    if model_key == "conv_local":
        return LeConvDeepRateMatrix(
            lattice_size=10,
            vocab_size=2,
            kernel_schedule=(3, 5, 7, 9, 15),
            hidden_dim=64,
            use_global_context=False,
        )
    raise KeyError(f"unknown model key {model_key!r}")
