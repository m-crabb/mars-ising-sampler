"""Evaluate checkpoints and create observable histogram overlays."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from .configs import BETA_CRITICAL, MODEL_CONFIGS, SIGMA_CRITICAL, build_model
from .ctmc import sample_ctmc
from .metrics import (
    entropy_per_site,
    ess_from_log_weights,
    free_energy_per_site,
    internal_energy_per_site,
)
from .reference import energy_per_site as np_energy_per_site
from .reference import magnetisation as np_magnetisation
from .reference import wolff_samples
from .target import IsingTarget, sample_uniform_spins


def evaluate_model(model_key: str, checkpoint: Path, args, reference: dict) -> dict:
    """Sample one model, compute ESS/thermodynamic metrics, and save observables."""
    cfg = MODEL_CONFIGS[model_key]
    device = torch.device(args.device)
    target = IsingTarget(D=10, sigma=SIGMA_CRITICAL, device=device)
    model = build_model(model_key).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    x0 = sample_uniform_spins(
        args.n_samples,
        target.d,
        device=device,
        generator=generator,
    )
    ts = torch.linspace(0.0, 1.0, cfg.n_euler_steps, device=device)
    with torch.no_grad():
        samples, log_weights = sample_ctmc(model, x0, ts, target=target, return_log_weights=True)

    # ESS is the main numerical score: high ESS means the learned CTMC
    # importance weights are close to uniform.
    log_prob = target.log_prob(samples)
    ess = ess_from_log_weights(log_weights)
    beta = 2.0 * target.sigma
    f_site = free_energy_per_site(log_weights, beta, target.d)
    e_site = internal_energy_per_site(log_weights, log_prob, beta, target.d)
    s_site = entropy_per_site(f_site, e_site, beta)

    samples_np = samples.detach().cpu().numpy()
    energy = np_energy_per_site(samples_np, 10)
    mag = np_magnetisation(samples_np)
    np.savez(args.out_dir / f"{model_key}_observables.npz", energy=energy, magnetisation=mag)

    return {
        "model": model_key,
        "checkpoint": str(checkpoint),
        "n_samples": args.n_samples,
        "ess": float(ess.item()),
        "ess_fraction": float(ess.item() / args.n_samples),
        "free_energy_per_site": float(f_site.item()),
        "internal_energy_per_site": float(e_site.item()),
        "entropy_per_site": float(s_site.item()),
        "raw_energy_mean": float(energy.mean()),
        "raw_energy_std": float(energy.std()),
        "raw_abs_magnetisation_mean": float(np.abs(mag).mean()),
        "reference_energy_mean": float(reference["energy"].mean()),
        "reference_abs_magnetisation_mean": float(np.abs(reference["magnetisation"]).mean()),
    }


def make_reference(args) -> dict:
    """Load or generate the Wolff MCMC reference used for histogram overlays."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / "wolff_reference.npz"
    if cache.exists() and not args.regenerate_reference:
        loaded = np.load(cache)
        return {"samples": loaded["samples"], "energy": loaded["energy"], "magnetisation": loaded["magnetisation"]}
    samples = wolff_samples(
        lattice_size=10,
        beta=BETA_CRITICAL,
        n_samples=args.reference_samples,
        burn_in=args.reference_burn_in,
        thin=args.reference_thin,
        seed=args.seed + 17,
    )
    energy = np_energy_per_site(samples, 10)
    mag = np_magnetisation(samples)
    np.savez(cache, samples=samples, energy=energy, magnetisation=mag)
    return {"samples": samples, "energy": energy, "magnetisation": mag}


def plot_overlays(model_keys: list[str], reference: dict, out_dir: Path) -> None:
    """Write energy and magnetisation histogram overlays to out_dir."""
    labels = {
        "letf": "transformer",
        "conv_global": "conv + global",
        "conv_local": "conv local",
    }
    colors = {
        "letf": "#1f77b4",
        "conv_global": "#2ca02c",
        "conv_local": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(reference["energy"], bins=36, density=True, histtype="step", linewidth=2.5, color="black", label="Wolff MCMC")
    for key in model_keys:
        obs = np.load(out_dir / f"{key}_observables.npz")
        ax.hist(obs["energy"], bins=36, density=True, histtype="step", linewidth=1.8, color=colors[key], label=labels[key])
    ax.set_xlabel("energy per site")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_histograms.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(reference["magnetisation"], bins=36, density=True, histtype="step", linewidth=2.5, color="black", label="Wolff MCMC")
    for key in model_keys:
        obs = np.load(out_dir / f"{key}_observables.npz")
        ax.hist(obs["magnetisation"], bins=36, density=True, histtype="step", linewidth=1.8, color=colors[key], label=labels[key])
    ax.set_xlabel("magnetisation per site")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "magnetisation_histograms.png", dpi=200)
    plt.close(fig)


def parse_args():
    """Command-line interface for reproducing the reported diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", *MODEL_CONFIGS.keys()], default="all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--reference-samples", type=int, default=5000)
    parser.add_argument("--reference-burn-in", type=int, default=2000)
    parser.add_argument("--reference-thin", type=int, default=5)
    parser.add_argument("--regenerate-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = make_reference(args)
    model_keys = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    metrics = {}
    for key in model_keys:
        checkpoint = args.checkpoint if args.checkpoint else MODEL_CONFIGS[key].checkpoint
        metrics[key] = evaluate_model(key, checkpoint, args, reference)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_overlays(model_keys, reference, args.out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
