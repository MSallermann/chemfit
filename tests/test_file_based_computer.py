from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pytest

from chemfit.abstract_objective_function import (
    EvaluateContext,
)
from chemfit.file_based_computer import FileBasedQuantityComputer
from chemfit.fitter import Fitter

if TYPE_CHECKING:
    from collections.abc import Iterable


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
        script_file: Path,
        output_file: Path,
    ) -> list[str]:
        return [
            sys.executable,
            str(script_file),
            str(parameters["prefactor"]),
            str(workdir / output_file),
        ]

    output_parser = MyOutputParser()

    ob_func = (
        FileBasedQuantityComputer(
            output_files=[output_file],
            output_parsers=output_parser,
            poll_interval=0.5,
            base_working_directory=test_dir / ".filebased_workdir",
            subprocess_run_args={"capture_output": True},
            delete_temp_workdirs=True,
        )
        .with_cmd(callable_cmd, script_file=script_file, output_file=output_file)
        .with_loss(loss_function, ref_y=ref_quantities["y"])
    )

    ctx = EvaluateContext()
    ob_func(initial_guess, ctx)

    fitter = Fitter(ob_func, initial_params=initial_guess)

    opt_params = fitter.fit_scipy()

    assert np.isclose(opt_params["prefactor"], 2.0)


def test_try_parsing_after_subprocess_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    output_file = Path("result.txt")

    def failing_run(
        cmd: list[str], *, check: bool, cwd: Path | str, **_kwargs: Any
    ) -> NoReturn:
        assert check
        (Path(cwd) / output_file).write_text("42", encoding="utf-8")
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="partial output",
            stderr="expected failure",
        )

    def parse_output(output_files: list[Path]) -> dict[str, int]:
        return {"result": int(output_files[0].read_text(encoding="utf-8"))}

    monkeypatch.setattr(subprocess, "run", failing_run)
    computer = FileBasedQuantityComputer(
        output_files=[output_file],
        output_parsers=parse_output,
        base_working_directory=tmp_path,
        subprocess_run_args={},
        try_parsing_after_exception=True,
    ).with_cmd(lambda _parameters, _workdir: ["failing-command"])
    ctx = EvaluateContext()

    with caplog.at_level(logging.WARNING):
        result = computer({}, ctx)

    assert result == {"result": 42}
    assert not ctx.temp.workdir.exists()
    assert "Will attempt to parse output files." in caplog.text
    dump_files = list(tmp_path.glob("*.dump"))
    assert len(dump_files) == 1
    assert "expected failure" in dump_files[0].read_text(encoding="utf-8")


def test_subprocess_exception_does_not_parse_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_file = Path("result.txt")
    parser_called = False

    def failing_run(
        cmd: list[str], *, check: bool, cwd: Path | str, **_kwargs: Any
    ) -> NoReturn:
        assert check
        (Path(cwd) / output_file).write_text("42", encoding="utf-8")
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    def parse_output(output_files: list[Path]) -> dict[str, int]:
        nonlocal parser_called
        assert output_files
        parser_called = True
        return {"result": 42}

    monkeypatch.setattr(subprocess, "run", failing_run)
    computer = FileBasedQuantityComputer(
        output_files=[output_file],
        output_parsers=parse_output,
        base_working_directory=tmp_path,
        subprocess_run_args={},
        delete_temp_workdirs=True,
        keep_temp_workdir_after_crash=False,
        write_dump_file_after_crash=False,
    ).with_cmd(lambda _parameters, _workdir: ["failing-command"])
    ctx = EvaluateContext()

    with pytest.raises(Exception, match="Exception in `_compute`") as exc_info:
        computer({}, ctx)

    subprocess_exception = exc_info.value.__cause__.__cause__
    assert isinstance(subprocess_exception, subprocess.CalledProcessError)
    assert not parser_called
    assert not ctx.temp.workdir.exists()


if __name__ == "__main__":
    logging.basicConfig(filename="test_file_based.log")

    test_squares_file_based()
