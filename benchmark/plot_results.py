import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def main(results_folder: Path, plot_folder: Path):
    fig, ax = plt.subplots()

    for i, p in enumerate(results_folder.glob("*.json")):
        with p.open("r") as f:
            res = json.load(f)

        time_taken = res["time_taken_list"]
        df = pl.from_records(time_taken)

        ax.plot(
            df["wait_time"],
            df["time_taken"],
            label=res["params"]["label"],
            marker="s",
            mec="black",
            ls="None",
            color=f"C{i}",
        )

        ax.set_xlabel("Time per task [s]")
        ax.set_ylabel("Total time [s]")
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.grid(visible=True)
    ax.legend()

    fig.savefig(plot_folder / "time_taken.png", dpi=600)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("-i", type=Path, required=True)
    cli.add_argument("-o", type=Path, required=True)

    args = cli.parse_args()
    input_folder = Path(args.i)
    output_folder = Path(args.o)
    output_folder.mkdir(exist_ok=True)

    main(results_folder=input_folder, plot_folder=output_folder)
