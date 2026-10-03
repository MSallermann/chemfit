from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pytest

from chemfit.abstract_objective_function import (
    EvaluateContext,
)
from chemfit.external_computer import ExternalQuantityComputer
from chemfit.fitter import Fitter

if TYPE_CHECKING:
    from collections.abc import Iterable


class MyParser:
    def __call__(self, output_file: Path) -> dict[str, Any]:
        """Parse an output file and retrieve its quantities."""
        data = np.loadtxt(output_file)
        return {"y": data[:, 0], "x": data[:, 1]}


def loss_function(quantities: dict[str, Any], ref_y: Iterable[float]) -> float:
    y_values = quantities["y"]
    errors = [(y - y_r) ** 2 for y, y_r in zip(y_values, ref_y, strict=False)]
    return np.sum(errors)


def test_squares_external():
    test_dir = Path(__file__).parent

    ref_file = test_dir / Path("input/ref_data.txt")
    # Get the reference data for 2.0*(x-2)**2
    data = np.loadtxt(ref_file)
    ref_quantities = {"y": data[:, 0], "x": data[:, 1]}

    # Output file created by the ExternalQuantityComputer
    output_file = Path("output/output_square_function.txt")

    # Script that creates the output file
    script_file = test_dir / Path("input/square_function.py")

    # Initial guess for the prefactor (the one parameter we will change)
    initial_guess = {"prefactor": 0.01}

    # Define the command that will be called to create the output file with given parameters
    def callable_cmd(
        parameters: dict[str, float],
        workdir: Path,
        _ctx: EvaluateContext,
        *,
        script_file: Path,
        output_file: Path,
    ) -> list[str]:
        return [
            sys.executable,
            str(script_file),
            str(parameters["prefactor"]),
            str(workdir / output_file),
        ]

    output_parser = MyParser()

    ob_func = (
        ExternalQuantityComputer(
            poll_interval=0.5,
            base_working_directory=test_dir / ".external_workdir",
            subprocess_run_args={"capture_output": True},
            delete_temp_workdirs=True,
        )
        .with_cmd(callable_cmd, script_file=script_file, output_file=output_file)
        .with_parser(output_parser, output_file)
        .with_loss(loss_function, ref_y=ref_quantities["y"])
    )

    ctx = EvaluateContext()
    ob_func(initial_guess, ctx)

    fitter = Fitter(ob_func, initial_params=initial_guess)

    opt_params = fitter.fit_scipy()

    assert np.isclose(opt_params["prefactor"], 2.0)


def test_parser_bindings_and_completion_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parser_calls: list[tuple[str, ...]] = []

    def fake_run(
        _cmd: list[str], *, check: bool, cwd: Path | str, **_kwargs: Any
    ) -> None:
        assert check
        workdir = Path(cwd)
        for name, contents in {
            "a.dat": "1",
            "b.dat": "2",
            "c.dat": "3",
            "task.done": "",
        }.items():
            (workdir / name).write_text(contents, encoding="utf-8")

    def parse_a(output: Path) -> dict[str, int]:
        parser_calls.append((output.name,))
        return {"a": int(output.read_text(encoding="utf-8"))}

    def parse_bc(first: Path, second: Path) -> dict[str, int]:
        parser_calls.append((first.name, second.name))
        return {
            "b": int(first.read_text(encoding="utf-8")),
            "c": int(second.read_text(encoding="utf-8")),
        }

    monkeypatch.setattr(subprocess, "run", fake_run)
    original = ExternalQuantityComputer(base_working_directory=tmp_path)
    with_one_parser = original.with_parser(parse_a, "a.dat")
    with_parsers = with_one_parser.with_parser(parse_bc, "b.dat", "c.dat")
    computer = with_parsers.wait_for("task.done", "a.dat", "task.done").with_cmd(
        lambda _parameters, _workdir, _ctx: ["simulation"]
    )

    assert original._parser_bindings == ()  # noqa: SLF001
    assert len(with_one_parser._parser_bindings) == 1  # noqa: SLF001
    assert len(with_parsers._parser_bindings) == 2  # noqa: SLF001
    assert with_parsers._completion_files == ()  # noqa: SLF001
    assert computer._completion_files == (  # noqa: SLF001
        Path("task.done"),
        Path("a.dat"),
    )

    ctx = EvaluateContext()
    assert computer({}, ctx) == {"a": 1, "b": 2, "c": 3}
    assert parser_calls == [("a.dat",), ("b.dat", "c.dat")]
    assert [path.name for path in ctx.temp.output_files] == [
        "a.dat",
        "b.dat",
        "c.dat",
        "task.done",
    ]


def test_execution_steps_run_in_registration_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []
    step_contexts: list[EvaluateContext] = []

    def hook(
        parameters: dict[str, int],
        _workdir: Path,
        ctx: EvaluateContext,
        *,
        name: str,
    ) -> None:
        assert parameters == {"value": 7}
        step_contexts.append(ctx)
        if name == "input":
            ctx.temp.prepared_value = parameters["value"]
        else:
            assert ctx.temp.prepared_value == 7
        events.append(f"hook:{name}")

    def command(
        parameters: dict[str, int],
        _workdir: Path,
        ctx: EvaluateContext,
        *,
        executable: str,
    ) -> list[str]:
        step_contexts.append(ctx)
        assert ctx.temp.prepared_value == 7
        events.append(f"build:{executable}")
        return [executable, str(parameters["value"])]

    def run_command(
        cmd: list[str], *, check: bool, cwd: Path | str, **_kwargs: Any
    ) -> None:
        assert check
        events.append(f"run:{cmd[0]}")
        if cmd[0] == "convert":
            workdir = Path(cwd)
            (workdir / "result.txt").write_text(cmd[1], encoding="utf-8")
            (workdir / "task.done").touch()

    def parse_result(output: Path) -> dict[str, int]:
        events.append("parse")
        return {"result": int(output.read_text(encoding="utf-8"))}

    monkeypatch.setattr(subprocess, "run", run_command)
    base = ExternalQuantityComputer(base_working_directory=tmp_path)
    pipeline = (
        base.with_hook(hook, name="input")
        .with_cmd(command, executable="preprocess")
        .with_hook(hook, name="modify")
        .with_cmd(command, executable="simulate")
        .with_hook(hook, name="postprocess")
        .with_cmd(command, executable="convert")
        .with_parser(parse_result, "result.txt")
        .wait_for("task.done")
    )
    other_branch = base.with_cmd(command, executable="other")

    assert base._steps == ()  # noqa: SLF001
    assert len(pipeline._steps) == 6  # noqa: SLF001
    assert len(other_branch._steps) == 1  # noqa: SLF001

    ctx = EvaluateContext()
    assert pipeline({"value": 7}, ctx) == {"result": 7}
    assert events == [
        "hook:input",
        "build:preprocess",
        "run:preprocess",
        "hook:modify",
        "build:simulate",
        "run:simulate",
        "hook:postprocess",
        "build:convert",
        "run:convert",
        "parse",
    ]
    assert ctx.temp.commands == [
        ["preprocess", "7"],
        ["simulate", "7"],
        ["convert", "7"],
    ]
    assert step_contexts == [ctx] * 6


def test_hook_failure_stops_execution_pipeline(tmp_path: Path):
    events: list[str] = []

    def fail(
        _parameters: dict[str, float],
        _workdir: Path,
        _ctx: EvaluateContext,
    ) -> None:
        events.append("failed hook")
        msg = "hook failed"
        raise ValueError(msg)

    def later_command(
        _parameters: dict[str, float],
        _workdir: Path,
        _ctx: EvaluateContext,
    ) -> list[str]:
        events.append("command")
        return ["command"]

    computer = (
        ExternalQuantityComputer(base_working_directory=tmp_path)
        .with_hook(lambda _parameters, _workdir, _ctx: events.append("first hook"))
        .with_hook(fail)
        .with_cmd(later_command)
    )
    ctx = EvaluateContext()

    with pytest.raises(Exception, match="Exception in `_compute`") as exc_info:
        computer({}, ctx)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert events == ["first hook", "failed hook"]
    assert ctx.temp.commands == []
    dump_files = list(tmp_path.glob("*.dump"))
    assert len(dump_files) == 1
    dump = dump_files[0].read_text(encoding="utf-8")
    assert "Exception type: ValueError" in dump
    assert "hook failed" in dump
    assert "'current_step_index': 1" in dump
    assert "'current_step_type': 'hook'" in dump


def test_waits_synchronously_for_delayed_completion_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def submit_job(_cmd: list[str], *, cwd: Path | str, **_kwargs: Any) -> None:
        workdir = Path(cwd)
        (workdir / "result.txt").write_text("42", encoding="utf-8")

        def finish_job() -> None:
            time.sleep(0.02)
            (workdir / "task.done").touch()

        threading.Thread(target=finish_job, daemon=True).start()

    def parse_result(output: Path) -> dict[str, int]:
        assert (output.parent / "task.done").exists()
        return {"result": int(output.read_text(encoding="utf-8"))}

    monkeypatch.setattr(subprocess, "run", submit_job)
    computer = (
        ExternalQuantityComputer(
            base_working_directory=tmp_path,
            poll_interval=0.001,
            wait_timeout=1,
        )
        .with_cmd(lambda _parameters, _workdir, _ctx: ["submit-job"])
        .with_parser(parse_result, "result.txt")
        .wait_for("task.done")
    )

    assert computer({}, EvaluateContext()) == {"result": 42}


def test_output_wait_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    parser_called = False

    def parse_result(_output: Path) -> dict[str, int]:
        nonlocal parser_called
        parser_called = True
        return {"result": 42}

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: None)
    computer = (
        ExternalQuantityComputer(
            base_working_directory=tmp_path,
            poll_interval=0.001,
            wait_timeout=0.01,
            keep_temp_workdir_after_crash=False,
        )
        .with_cmd(lambda _parameters, _workdir, _ctx: ["submit-job"])
        .with_parser(parse_result, "missing.txt")
    )
    ctx = EvaluateContext()

    with pytest.raises(Exception, match="Exception in `_compute`") as exc_info:
        computer({}, ctx)

    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert "Timed out waiting" in str(exc_info.value.__cause__)
    assert not parser_called
    assert not ctx.temp.workdir.exists()


def test_parser_retries_are_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    attempts = 0

    def fake_run(_cmd: list[str], *, cwd: Path | str, **_kwargs: Any) -> None:
        (Path(cwd) / "result.txt").write_text("42", encoding="utf-8")

    def flaky_parser(output: Path) -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            msg = "not ready"
            raise ValueError(msg)
        return {"result": int(output.read_text(encoding="utf-8"))}

    monkeypatch.setattr(subprocess, "run", fake_run)
    computer = (
        ExternalQuantityComputer(base_working_directory=tmp_path)
        .with_cmd(lambda _parameters, _workdir, _ctx: ["simulation"])
        .with_parser(flaky_parser, "result.txt")
    )

    assert computer({}, EvaluateContext()) == {"result": 42}
    assert attempts == 2


def test_parser_and_completion_file_validation(tmp_path: Path):
    computer = ExternalQuantityComputer(base_working_directory=tmp_path)

    with pytest.raises(TypeError, match="callable"):
        computer.with_parser(None, "result.txt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        computer.with_parser(lambda _path: {})
    with pytest.raises(ValueError, match="relative"):
        computer.with_parser(lambda _path: {}, tmp_path / "result.txt")
    with pytest.raises(ValueError, match="at least one"):
        computer.wait_for()
    with pytest.raises(ValueError, match="relative"):
        computer.wait_for(tmp_path / "task.done")


def test_try_parsing_after_subprocess_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    output_file = Path("result.txt")
    later_step_ran = False

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

    def parse_output(output_file: Path) -> dict[str, int]:
        return {"result": int(output_file.read_text(encoding="utf-8"))}

    def later_hook(
        _parameters: dict[str, Any],
        _workdir: Path,
        _ctx: EvaluateContext,
    ) -> None:
        nonlocal later_step_ran
        later_step_ran = True

    monkeypatch.setattr(subprocess, "run", failing_run)
    computer = (
        ExternalQuantityComputer(
            base_working_directory=tmp_path,
            subprocess_run_args={},
            try_parsing_after_exception=True,
        )
        .with_cmd(lambda _parameters, _workdir, _ctx: ["failing-command"])
        .with_hook(later_hook)
        .with_parser(parse_output, output_file)
    )
    ctx = EvaluateContext()

    with caplog.at_level(logging.WARNING):
        result = computer({}, ctx)

    assert result == {"result": 42}
    assert not later_step_ran
    assert ctx.temp.commands == [["failing-command"]]
    assert not ctx.temp.workdir.exists()
    assert "Will attempt to parse output files." in caplog.text
    assert ctx.temp.command_failure["exception_type"] == "CalledProcessError"
    assert ctx.temp.command_failure["stderr"] == "expected failure"
    assert list(tmp_path.glob("*.dump")) == []


def test_later_failure_dumps_recovered_command_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def failing_run(cmd: list[str], **_kwargs: Any) -> NoReturn:
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=cmd,
            stderr="recoverable command failure",
        )

    monkeypatch.setattr(subprocess, "run", failing_run)
    computer = (
        ExternalQuantityComputer(
            base_working_directory=tmp_path,
            poll_interval=0.001,
            wait_timeout=0.01,
            try_parsing_after_exception=True,
        )
        .with_cmd(lambda _parameters, _workdir, _ctx: ["failing-command"])
        .with_parser(lambda _output: {"result": 42}, "missing.txt")
    )

    with pytest.raises(Exception, match="Exception in `_compute`"):
        computer({}, EvaluateContext())

    dump_files = list(tmp_path.glob("*.dump"))
    assert len(dump_files) == 1
    dump = dump_files[0].read_text(encoding="utf-8")
    assert "Exception type: TimeoutError" in dump
    assert "CalledProcessError" in dump
    assert "recoverable command failure" in dump


def test_subprocess_exception_does_not_parse_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_file = Path("result.txt")
    parser_called = False
    later_step_ran = False

    def failing_run(
        cmd: list[str], *, check: bool, cwd: Path | str, **_kwargs: Any
    ) -> NoReturn:
        assert check
        (Path(cwd) / output_file).write_text("42", encoding="utf-8")
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="partial stdout",
            stderr="fatal stderr",
        )

    def parse_output(output_file: Path) -> dict[str, int]:
        nonlocal parser_called
        assert output_file.name == "result.txt"
        parser_called = True
        return {"result": 42}

    def later_hook(
        _parameters: dict[str, Any],
        _workdir: Path,
        _ctx: EvaluateContext,
    ) -> None:
        nonlocal later_step_ran
        later_step_ran = True

    monkeypatch.setattr(subprocess, "run", failing_run)
    computer = (
        ExternalQuantityComputer(
            base_working_directory=tmp_path,
            subprocess_run_args={},
            delete_temp_workdirs=True,
            keep_temp_workdir_after_crash=False,
        )
        .with_cmd(lambda _parameters, _workdir, _ctx: ["failing-command"])
        .with_hook(later_hook)
        .with_parser(parse_output, output_file)
    )
    ctx = EvaluateContext()

    with pytest.raises(Exception, match="Exception in `_compute`") as exc_info:
        computer({}, ctx)

    assert isinstance(exc_info.value.__cause__, subprocess.CalledProcessError)
    assert not parser_called
    assert not later_step_ran
    assert ctx.temp.commands == [["failing-command"]]
    assert not ctx.temp.workdir.exists()
    dump_files = list(tmp_path.glob("*.dump"))
    assert len(dump_files) == 1
    dump = dump_files[0].read_text(encoding="utf-8")
    assert "Exception type: CalledProcessError" in dump
    assert "returncode: 1" in dump
    assert "partial stdout" in dump
    assert "fatal stderr" in dump


if __name__ == "__main__":
    logging.basicConfig(filename="test_external.log")

    test_squares_external()
