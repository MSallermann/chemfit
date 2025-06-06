import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List


def plot_progress_curve(progress: List[float], outpath: Path) -> None:
    """
    Save a semilog plot of the objective values (progress) versus iteration index.
    """
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


def plot_energies_and_residuals(
    df: pd.DataFrame,
    output_folder: Path,
    plot_initial: bool = True,
) -> None:
    """
    Given a DataFrame with columns:
      - 'tag'
      - 'energy_reference'
      - 'energy_initial'
      - 'energy_fitted'
      - 'n_atoms'
      - 'ob_value'
    Create two PNGs in output_folder:
      1. energy per atom vs. tag (reference, fitted, and optionally initial)
      2. residuals = |reference - fitted| / n_atoms vs. tag

    The files are saved as:
      output_folder / "plot_energy.png"
      output_folder / "plot_residuals.png"
      output_folder / "plot_objective.png"
    """
    tags = df["tag"]
    energy_ref = df["energy_reference"] / df["n_atoms"]
    energy_fit = df["energy_fitted"] / df["n_atoms"]
    energy_init = df["energy_initial"] / df["n_atoms"] if plot_initial else None
    n = len(tags)

    # Plot energies
    plt.close()
    ax = plt.gca()
    fig = plt.gcf()
    ax.plot(tags, energy_ref, marker="o", color="black", label="reference")
    ax.plot(tags, energy_fit, marker="x", label="fitted")
    if plot_initial:
        ax.plot(tags, energy_init, marker=".", label="initial")
    ax.set_xticks(range(n))
    ax.set_xticklabels(tags, rotation=90)
    ax.set_ylabel("energy [eV] / n_atoms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_folder / "plot_energy.png", dpi=300)
    plt.close()

    # Plot residuals
    residuals = np.abs(df["energy_reference"] - df["energy_fitted"]) / df["n_atoms"]
    avg_resid = np.mean(residuals)

    plt.close()
    ax = plt.gca()
    fig = plt.gcf()
    ax.plot(
        tags,
        residuals,
        marker="o",
        color="black",
        label=f"residuals (mean={avg_resid:.2e})",
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels(tags, rotation=90)
    ax.set_ylabel("|pred - target| [eV] / n_atoms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_folder / "plot_residuals.png", dpi=300)
    plt.close()

    # Plot objective function
    ob_values = np.abs(df["ob_value"])
    avg_ob = np.mean(ob_values)

    plt.close()
    ax = plt.gca()
    fig = plt.gcf()
    ax.plot(
        tags,
        ob_values,
        marker="o",
        color="black",
        label=f"objective_function (mean = {avg_ob:.2e})",
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels(tags, rotation=90)
    ax.set_ylabel("objective_function")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_folder / "plot_objective_function.png", dpi=300)
    plt.close()
