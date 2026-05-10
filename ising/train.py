"""Optional training entrypoint for the submitted models.

The shipped checkpoints are used for the reported results. This script is kept
so the learning setup is reproducible; `pixi run train-letf-debug` runs a short
version, while the full transformer curriculum is intentionally long.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from .configs import MODEL_CONFIGS, SIGMA_CRITICAL, build_model
from .ctmc import compute_c_t_grid, kolmogorov_loss, sample_ctmc
from .target import IsingTarget, sample_uniform_spins


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set all optimizer parameter groups to a new learning rate."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def train(
    config_key: str,
    output_dir: Path,
    steps_override: int | None = None,
    seed: int = 42,
    batch_size_override: int | None = None,
    outer_batch_size_override: int | None = None,
    n_euler_steps_override: int | None = None,
) -> None:
    """Train one model with the sigma curriculum from configs.py."""
    cfg = MODEL_CONFIGS[config_key]
    n_steps = steps_override or cfg.n_steps
    batch_size = batch_size_override or cfg.batch_size
    outer_batch_size = outer_batch_size_override or cfg.outer_batch_size
    n_euler_steps = n_euler_steps_override or cfg.n_euler_steps
    inner_steps_per_outer = 100
    if n_steps % inner_steps_per_outer:
        raise ValueError("n_steps must be divisible by 100")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    target = IsingTarget(D=10, sigma=cfg.curriculum[0][1], device=device)
    model = build_model(config_key).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    t_grid = torch.linspace(0.0, 1.0, n_euler_steps, device=device)
    replay_x: list[torch.Tensor] = []
    replay_t: list[torch.Tensor] = []
    curriculum = [stage for stage in cfg.curriculum if stage[0] < n_steps]
    curriculum_idx = -1

    log_file = (output_dir / "training_log.csv").open("w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["step", "loss", "sigma", "lr"])

    step = 0
    for _outer in range(n_steps // inner_steps_per_outer):
        # Stage changes happen only on outer-loop boundaries, so replay-buffer
        # states and target sigma are always consistent.
        while curriculum_idx + 1 < len(curriculum) and step >= curriculum[curriculum_idx + 1][0]:
            curriculum_idx += 1
            _, sigma, lr = curriculum[curriculum_idx]
            target.set_sigma(sigma)
            set_lr(optimizer, lr)
            replay_x.clear()
            replay_t.clear()

        x0 = sample_uniform_spins(outer_batch_size, target.d, device=device)
        with torch.no_grad():
            x_traj = sample_ctmc(model, x0, t_grid, return_all_states=True)
            c_t_grid, _ = compute_c_t_grid(t_grid, x_traj, target, model)

        t_idx = torch.arange(n_euler_steps, device=device).repeat_interleave(outer_batch_size)
        replay_x.append(x_traj.reshape(-1, target.d).detach())
        replay_t.append(t_idx.detach())
        if len(replay_x) > cfg.replay_buffer_cycles:
            replay_x.pop(0)
            replay_t.pop(0)
        x_buffer = torch.cat(replay_x, dim=0)
        t_buffer = torch.cat(replay_t, dim=0)

        for _inner in range(inner_steps_per_outer):
            idx = torch.randint(x_buffer.shape[0], (batch_size,), device=device)
            x = x_buffer[idx]
            t_idx_sample = t_buffer[idx]
            t = t_grid[t_idx_sample]
            c_t = c_t_grid[t_idx_sample]
            loss = kolmogorov_loss(x, t, c_t, model, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            writer.writerow([step, float(loss.item()), target.sigma, optimizer.param_groups[0]["lr"]])
            if step % 500 == 0:
                log_file.flush()
                torch.save(model.state_dict(), output_dir / "checkpoints" / "latest.pt")
            step += 1

    torch.save(model.state_dict(), output_dir / "checkpoints" / "final.pt")
    (output_dir / "config.json").write_text(
        json.dumps(
            {"config": config_key, "sigma_final": SIGMA_CRITICAL, "n_steps": n_steps},
            indent=2,
        )
    )
    log_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=MODEL_CONFIGS.keys(), default="letf")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/debug"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--outer-batch-size", type=int)
    parser.add_argument("--n-euler-steps", type=int)
    args = parser.parse_args()
    train(
        args.config,
        args.output_dir,
        args.steps,
        args.seed,
        args.batch_size,
        args.outer_batch_size,
        args.n_euler_steps,
    )


if __name__ == "__main__":
    main()
