import argparse
from pathlib import Path

import numpy as np


def square_func(prefactor: float, x: float):
    return prefactor * (x - 2) ** 2


def output_y_values(outfile: Path, prefactor: float, xmin: 0.0, xmax: 100):
    x_values = np.linspace(xmin, xmax, num=100)

    y_values = [square_func(prefactor, x) for x in x_values]

    data = np.column_stack([y_values, x_values])
    np.savetxt(outfile, data, header="y x", fmt="%.6f")


def main():
    parser = argparse.ArgumentParser(
        description="Generate y = prefactor * (x - 2)^2 values and save to file."
    )
    parser.add_argument(
        "prefactor", type=float, help="Prefactor for the square function"
    )
    parser.add_argument("outfile", type=Path, help="Output file path")
    parser.add_argument(
        "--xmin", type=float, default=0.0, help="Minimum x value (default: 0.0)"
    )
    parser.add_argument(
        "--xmax", type=float, default=100.0, help="Maximum x value (default: 100.0)"
    )
    args = parser.parse_args()

    output_y_values(args.outfile, args.prefactor, args.xmin, args.xmax)


if __name__ == "__main__":
    main()
