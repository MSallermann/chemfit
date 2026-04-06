from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from chemfit.abstract_objective_function import (
    EvaluateContext,
    QuantityComputerObjectiveFunction,
)
from chemfit.file_based_computer import FileBasedQuantityComputer
from chemfit.fitter import Fitter


class MyOutputParser:
    def __call__(self, output_files: list[Path]) -> dict[str, Any]:
        """Parse the output files and retrieve the quantities."""

        res = {}
        for f in output_files:
            data = np.loadtxt(f)
            res = {"y": data[:, 0], "x": data[:, 1]}

        return res


def loss_function(quantities: dict[str, Any], ref_y: Iterable[float]) -> float:
    y_values = quantities["y"]
    errors = [(y - y_r) ** 2 for y, y_r in zip(y_values, ref_y)]
    return np.sum(errors)


def test_squares_file_based():
    test_dir = Path(__file__).parent

    ref_file = test_dir / Path("input/ref_data.txt")
    # Get the reference data for 2.0*(x-2)**2
    data = np.loadtxt(ref_file)
    ref_quantities = {"y": data[:, 0], "x": data[:, 1]}

    # Output file created by the FileBasedQuantityComputer
    output_file = Path("output/output_square_function.txt")

    # Script that creates the output file
    script_file = test_dir / Path("input/square_function.py")

    # Initial guess for the prefactor (the one parameter we will change)
    initial_guess = {"prefactor": 0.01}

    # Define the command that will be called to create the output file with given parameters
    def callable_cmd(
        parameters: dict[str, float],
        workdir: Path,
    ) -> list[str]:
        return f"python {script_file} {parameters['prefactor']} {workdir / output_file}".split()

    output_parser = MyOutputParser()

    file_based_computer = FileBasedQuantityComputer(
        output_files=[output_file],
        executable_cmd=callable_cmd,
        output_parsers=output_parser,
        poll_interval=0.5,
        base_working_directory=test_dir / ".filebased_workdir",
        subprocess_run_args={"capture_output": True},
        delete_temp_workdirs=True,
    )

    ob_func = QuantityComputerObjectiveFunction(
        loss_function=lambda q: loss_function(q, ref_y=ref_quantities["y"]),
        quantity_computer=file_based_computer,
    )

    ctx = EvaluateContext()
    ob_func(initial_guess, ctx)

    fitter = Fitter(ob_func, initial_params=initial_guess)

    opt_params = fitter.fit_scipy()

    assert np.isclose(opt_params["prefactor"], 2.0)


if __name__ == "__main__":
    import logging

    logging.basicConfig(filename="test_lj.log")

    test_squares_file_based()
