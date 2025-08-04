from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_progress_curve(progress: list[float], outpath: Path) -> None:
    """Save a semi-log plot of the objective values (progress) versus iteration index."""
    if len(progress) == 0:
        return
    plt.close()
    plt.plot(progress, marker=".")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Objective (log scale)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def tags_as_ticks(ax: plt.Axes, tags: list[str], **kwargs):
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=90, **kwargs)


def plot_energies(
    df: pd.DataFrame,
    output_folder: Path,
) -> None:
    tags = df["tag"]
    energy_ref = df["reference_energy"]
    energy_fit = df["last_energy"]
    n_atoms = df["n_atoms"]

    # Plot energies
    plt.close()
    ax = plt.gca()
    fig = plt.gcf()

    # Plot residuals
    residuals = np.array(np.abs(energy_ref - energy_fit) / n_atoms)

    mask = ~np.isnan(residuals)

    mean_resid = np.mean(residuals[mask])
    max_resid = np.max(residuals[mask])
    median_resid = np.median(residuals[mask])

    ax.set_title(
        f"Residuals: max {max_resid:.2e}, mean {mean_resid:.2e}, median {median_resid:.2e}"
    )
    ax.plot(
        energy_ref[mask] / n_atoms[mask], marker="o", color="black", label="reference"
    )
    ax.plot(energy_fit[mask] / n_atoms[mask], marker="x", label="fitted")

    ax.legend()
    ax.set_ylabel("energy [eV] / n_atoms")
    tags_as_ticks(ax, tags)
    fig.tight_layout()
    fig.savefig(output_folder / "plot_energy.png", dpi=300)
    plt.close()
