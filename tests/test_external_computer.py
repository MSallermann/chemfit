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

import chemfit
from chemfit.abstract_objective_function import (
    EvaluateContext,
)
from chemfit.evaluation_working_directory import EvaluationWorkingDirectory
from chemfit.external_computer import ExternalQuantityComputer
from chemfit.fitter import Fitter

if TYPE_CHECKING:
    from collections.abc import Iterable


def _external_computer(
    base_working_directory: Path | str,
    wait_timeout: float | None = 500.0,
    poll_interval: float = 1.0,
    delete_temp_workdirs: bool = True,
    write_dump_file_after_crash: bool = True,
    keep_temp_workdir_after_crash: bool = True,
    try_parsing_after_exception: bool = False,
) -> ExternalQuantityComputer[dict[str, Any], dict[str, Any]]:
    """Construct the low-level computer from its working-directory object."""
    return ExternalQuantityComputer(
        EvaluationWorkingDirectory(
            base_working_directory,
            delete_temp_workdirs=delete_temp_workdirs,
            keep_temp_workdir_after_crash=keep_temp_workdir_after_crash,
        ),
        wait_timeout=wait_timeout,
        poll_interval=poll_interval,
        write_dump_file_after_crash=write_dump_file_after_crash,
        try_parsing_after_exception=try_parsing_after_exception,
    )


class MyParser:
    def __call__(self, output_file: Path) -> dict[str, Any]:
        """Parse an output file and retrieve its quantities."""
        data = np.loadtxt(output_file)
        return {"y": data[:, 0], "x": data[:, 1]}


def test_external_computer_accepts_working_directory_configuration(tmp_path: Path):
    workdir = EvaluationWorkingDirectory(
        tmp_path,
        delete_temp_workdirs=False,
        keep_temp_workdir_after_crash=False,
    )
    computer = ExternalQuantityComputer(workdir)

    assert computer.evaluation_working_directory is workdir


def test_external_quantity_constructs_working_directory_configuration(
    tmp_path: Path,
):
    computer = chemfit.external_quantity(
        workdir=tmp_path,
        delete_temp_workdirs=False,
        keep_temp_workdir_after_crash=False,
    )

    workdir = computer.evaluation_working_directory
    assert workdir.base_working_directory == tmp_path
    assert not workdir.delete_temp_workdirs
    assert not workdir.keep_temp_workdir_after_crash


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
        _external_computer(
            poll_interval=0.5,
            base_working_directory=test_dir / ".external_workdir",
            delete_temp_workdirs=True,
        )
        .with_cmd(
            callable_cmd,
            subprocess_run_args={"capture_output": True},
            script_file=script_file,
            output_file=output_file,
        )
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
    original = _external_computer(base_working_directory=tmp_path)
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


def test_output_parsers_reject_duplicate_quantity_keys(tmp_path: Path):
    def write_output(
        _parameters: dict[str, Any],
        workdir: Path,
        _ctx: EvaluateContext,
    ) -> None:
        (workdir / "result.txt").touch()

    def make_computer(
        *, allow_overwrites: bool
    ) -> ExternalQuantityComputer[dict[str, Any], dict[str, Any]]:
        return (
            chemfit.external_quantity(
                tmp_path,
                allow_quantity_overwrites=allow_overwrites,
            )
            .with_hook(write_output)
            .with_parser(lambda _path: {"energy": 1.0}, "result.txt")
            .with_parser(lambda _path: {"energy": 2.0}, "result.txt")
        )

    with pytest.raises(
        ValueError,
        match=r"output parser 1 returned duplicate quantity keys: 'energy'",
    ):
        make_computer(allow_overwrites=False)({})

    assert make_computer(allow_overwrites=True)({}) == {"energy": 2.0}


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
    base = _external_computer(base_working_directory=tmp_path)
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


def test_subprocess_output_streams_to_workdir_by_default(tmp_path: Path):
    computer = _external_computer(
        base_working_directory=tmp_path,
        delete_temp_workdirs=False,
    ).with_cmd(
        lambda _parameters, _workdir, _ctx: [
            sys.executable,
            "-c",
            "import sys; print('standard output'); print('standard error', file=sys.stderr)",
        ]
    )
    ctx = EvaluateContext()

    assert computer({}, ctx) == {}

    stdout_file = ctx.temp.workdir / "command-0.stdout"
    stderr_file = ctx.temp.workdir / "command-0.stderr"
    assert stdout_file.read_text(encoding="utf-8") == "standard output\n"
    assert stderr_file.read_text(encoding="utf-8") == "standard error\n"
    assert ctx.temp.command_output_files == [
        {
            "step_index": 0,
            "stdout": stdout_file,
            "stderr": stderr_file,
        }
    ]


def test_named_command_output_can_be_used_as_parser_input(tmp_path: Path):
    def parse_command_output(
        stdout_file: Path,
        stderr_file: Path,
    ) -> dict[str, Any]:
        return {
            "result": int(stdout_file.read_text(encoding="utf-8")),
            "diagnostic": stderr_file.read_text(encoding="utf-8").strip(),
        }

    computer = (
        _external_computer(
            base_working_directory=tmp_path,
            delete_temp_workdirs=False,
        )
        .with_cmd(
            lambda _parameters, _workdir, _ctx: [
                sys.executable,
                "-c",
                "import sys; print(42); print('diagnostic', file=sys.stderr)",
            ],
            stdout_file="logs/result.txt",
            stderr_file="logs/simulation.err",
        )
        .with_parser(
            parse_command_output,
            "logs/result.txt",
            "logs/simulation.err",
        )
    )
    ctx = EvaluateContext()

    assert computer({}, ctx) == {"result": 42, "diagnostic": "diagnostic"}
    assert ctx.temp.command_output_files == [
        {
            "step_index": 0,
            "stdout": ctx.temp.workdir / "logs/result.txt",
            "stderr": ctx.temp.workdir / "logs/simulation.err",
        }
    ]


def test_failed_subprocess_keeps_streamed_output_for_diagnostics(tmp_path: Path):
    computer = _external_computer(base_working_directory=tmp_path).with_cmd(
        lambda _parameters, _workdir, _ctx: [
            sys.executable,
            "-c",
            "import sys; print('before failure'); print('failure detail', file=sys.stderr); raise SystemExit(3)",
        ]
    )
    ctx = EvaluateContext()

    with pytest.raises(subprocess.CalledProcessError) as exception_info:
        computer({}, ctx)

    assert exception_info.value.returncode == 3
    stdout_file = ctx.temp.command_failure["stdout_file"]
    stderr_file = ctx.temp.command_failure["stderr_file"]
    assert stdout_file.read_text(encoding="utf-8") == "before failure\n"
    assert stderr_file.read_text(encoding="utf-8") == "failure detail\n"
    assert ctx.temp.workdir.exists()


def test_explicit_subprocess_output_handling_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_run(_cmd: list[str], **kwargs: Any) -> None:
        assert kwargs["capture_output"] is True
        assert "stdout" not in kwargs
        assert "stderr" not in kwargs

    monkeypatch.setattr(subprocess, "run", fake_run)
    computer = _external_computer(
        base_working_directory=tmp_path,
        delete_temp_workdirs=False,
    ).with_cmd(
        lambda _parameters, _workdir, _ctx: ["simulation"],
        subprocess_run_args={"capture_output": True},
    )
    ctx = EvaluateContext()

    assert computer({}, ctx) == {}
    assert ctx.temp.command_output_files == []


def test_named_command_output_rejects_subprocess_output_configuration(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot be combined"):
        _external_computer(base_working_directory=tmp_path).with_cmd(
            lambda _parameters, _workdir, _ctx: ["simulation"],
            stdout_file="simulation.out",
            subprocess_run_args={"capture_output": True},
        )


def test_subprocess_run_args_are_command_specific(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    observed_timeouts: list[float] = []

    def fake_run(_cmd: list[str], **kwargs: Any) -> None:
        observed_timeouts.append(kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    computer = (
        _external_computer(base_working_directory=tmp_path)
        .with_cmd(
            lambda _parameters, _workdir, _ctx: ["first"],
            subprocess_run_args={"timeout": 1.0},
        )
        .with_cmd(
            lambda _parameters, _workdir, _ctx: ["second"],
            subprocess_run_args={"timeout": 2.0},
        )
    )

    assert computer({}, EvaluateContext()) == {}
    assert observed_timeouts == [1.0, 2.0]


@pytest.mark.parametrize("reserved_arg", ["check", "cwd"])
def test_subprocess_run_args_reject_chemfit_managed_arguments(
    tmp_path: Path,
    reserved_arg: str,
):
    with pytest.raises(ValueError, match="ChemFit manages"):
        _external_computer(base_working_directory=tmp_path).with_cmd(
            lambda _parameters, _workdir, _ctx: ["simulation"],
            subprocess_run_args={reserved_arg: True},
        )


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
        _external_computer(base_working_directory=tmp_path)
        .with_hook(lambda _parameters, _workdir, _ctx: events.append("first hook"))
        .with_hook(fail)
        .with_cmd(later_command)
    )
    ctx = EvaluateContext()

    with pytest.raises(ValueError):  # noqa: PT011
        computer({}, ctx)

    assert events == ["first hook", "failed hook"]
    assert ctx.temp.commands == []
    dump_files = list(tmp_path.glob("*.dump"))
    assert len(dump_files) == 1
    dump = dump_files[0].read_text(encoding="utf-8")
    assert "Exception type: ValueError" in dump
    assert "hook failed" in dump
    assert "'current_step_index': 1" in dump
    assert "'current_step_type': 'hook'" in dump


def test_cleanup_failure_does_not_mask_evaluation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    def fail_evaluation(
        _parameters: dict[str, Any],
        _workdir: Path,
        _ctx: EvaluateContext,
    ) -> None:
        msg = "evaluation failed"
        raise ValueError(msg)

    def fail_cleanup(_workdir: Path) -> None:
        msg = "cleanup failed"
        raise OSError(msg)

    monkeypatch.setattr(
        "chemfit.evaluation_working_directory.shutil.rmtree",
        fail_cleanup,
    )
    computer = _external_computer(
        base_working_directory=tmp_path,
        delete_temp_workdirs=True,
        keep_temp_workdir_after_crash=False,
        write_dump_file_after_crash=False,
    ).with_hook(fail_evaluation)

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(ValueError, match="evaluation failed") as exception_info,
    ):
        computer({}, EvaluateContext())

    assert "cleanup failed" in caplog.text
    if hasattr(exception_info.value, "__notes__"):
        assert "cleanup failed" in "\n".join(exception_info.value.__notes__)


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
        _external_computer(
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
        _external_computer(
            base_working_directory=tmp_path,
            poll_interval=0.001,
            wait_timeout=0.01,
            keep_temp_workdir_after_crash=False,
        )
        .with_cmd(lambda _parameters, _workdir, _ctx: ["submit-job"])
        .with_parser(parse_result, "missing.txt")
    )
    ctx = EvaluateContext()

    with pytest.raises(TimeoutError):
        computer({}, ctx)

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
        _external_computer(base_working_directory=tmp_path)
        .with_cmd(lambda _parameters, _workdir, _ctx: ["simulation"])
        .with_parser(flaky_parser, "result.txt")
    )

    assert computer({}, EvaluateContext()) == {"result": 42}
    assert attempts == 2


def test_parser_and_completion_file_validation(tmp_path: Path):
    computer = _external_computer(base_working_directory=tmp_path)

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
        _external_computer(
            base_working_directory=tmp_path,
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
        _external_computer(
            base_working_directory=tmp_path,
            poll_interval=0.001,
            wait_timeout=0.01,
            try_parsing_after_exception=True,
        )
        .with_cmd(lambda _parameters, _workdir, _ctx: ["failing-command"])
        .with_parser(lambda _output: {"result": 42}, "missing.txt")
    )

    # this test should time out since `try_parsing_after_exception=True`
    with pytest.raises(TimeoutError):
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
        _external_computer(
            base_working_directory=tmp_path,
            delete_temp_workdirs=True,
            keep_temp_workdir_after_crash=False,
        )
        .with_cmd(lambda _parameters, _workdir, _ctx: ["failing-command"])
        .with_hook(later_hook)
        .with_parser(parse_output, output_file)
    )
    ctx = EvaluateContext()

    with pytest.raises(subprocess.CalledProcessError):
        computer({}, ctx)

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


def test_unsafe_relative_paths():
    unsafe_path = "../bla/../../out.txt"

    with pytest.raises(ValueError, match="Relative path"):
        _external_computer(base_working_directory="workdir").wait_for(unsafe_path)

    with pytest.raises(ValueError, match="Relative path"):
        _external_computer(base_working_directory="workdir").with_parser(
            MyParser(), unsafe_path
        )

    with pytest.raises(ValueError, match="Relative path"):
        _external_computer(base_working_directory="workdir").with_cmd(
            lambda _parameters, _workdir, _ctx: ["simulation"],
            stdout_file=unsafe_path,
        )


if __name__ == "__main__":
    logging.basicConfig(filename="test_external.log")

    test_squares_external()
