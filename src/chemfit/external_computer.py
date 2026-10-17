from __future__ import annotations

import copy
import functools
import logging
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import (
    Any,
    Concatenate,
    Generic,
    TypeVar,
    cast,
)

from typing_extensions import Self

from chemfit.abstract_objective_function import EvaluateContext, QuantityComputer
from chemfit.evaluation_working_directory import EvaluationWorkingDirectory
from chemfit.utils import merge_quantities

logger = logging.getLogger(__name__)

ParametersT = TypeVar("ParametersT", bound=Mapping[str, object])
QuantitiesT = TypeVar("QuantitiesT", bound=dict[str, Any])


def _subprocess_output_to_text(output: bytes | str) -> str:
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


CommandType = Callable[[ParametersT, Path, EvaluateContext], list[str]]
HookType = Callable[[ParametersT, Path, EvaluateContext], None]


@dataclass(frozen=True)
class _CommandStep(Generic[ParametersT]):
    """Build and execute one command in an external evaluation."""

    command: CommandType[ParametersT]
    stdout_file: Path | None = None
    stderr_file: Path | None = None
    subprocess_run_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _HookStep(Generic[ParametersT]):
    """Run one Python hook in an external evaluation."""

    hook: HookType[ParametersT]


@dataclass(frozen=True)
class _ParserBinding(Generic[QuantitiesT]):
    """Associate an output parser with its ordered input files."""

    parser: Callable[..., QuantitiesT]
    files: tuple[Path, ...]


def _relative_paths(output_files: tuple[Path | str, ...]) -> tuple[Path, ...]:
    """Convert str to Path and make sure that they are relative to the working directory."""
    return EvaluationWorkingDirectory.relative_paths(output_files)


class ExternalQuantityComputer(QuantityComputer[ParametersT, QuantitiesT]):
    def __init__(
        self,
        evaluation_working_directory: EvaluationWorkingDirectory,
        wait_timeout: float | None = 500.0,
        poll_interval: float = 1,
        write_dump_file_after_crash: bool = True,
        try_parsing_after_exception: bool = False,
        allow_quantity_overwrites: bool = False,
    ):
        """
        Initialize an external quantity computer.

        This quantity computer evaluates parameters by creating a temporary
        working directory, executing an ordered sequence of hooks and commands,
        waiting for the expected output files to appear, and parsing those
        files into a quantity dictionary.

        Args:
            evaluation_working_directory: Configuration and lifecycle policy
                for per-evaluation working directories.
            wait_timeout (float, optional):
                Maximum time in seconds to wait for all output files to
                appear. Defaults to 500.0 seconds.
            poll_interval (float, optional):
                Interval in seconds between checks for output file creation.
                Defaults to 1 second.
            write_dump_file_after_crash: Whether to write an evaluation-level
                diagnostic dump when execution, output waiting, or parsing fails.
            try_parsing_after_exception: Whether to continue waiting for and parsing
                output files when ``subprocess.run`` raises
                ``subprocess.CalledProcessError``. Defaults to False.
            allow_quantity_overwrites: Whether later output parsers may replace
                keys returned by earlier parsers. Defaults to ``False``.

        """

        super().__init__()

        if not isinstance(evaluation_working_directory, EvaluationWorkingDirectory):
            msg = "evaluation_working_directory must be an EvaluationWorkingDirectory"
            raise TypeError(msg)

        self._steps: tuple[_CommandStep[ParametersT] | _HookStep[ParametersT], ...] = ()
        self._parser_bindings: tuple[_ParserBinding[QuantitiesT], ...] = ()
        self._completion_files: tuple[Path, ...] = ()
        self.evaluation_working_directory = evaluation_working_directory
        self.write_dump_file_after_crash = write_dump_file_after_crash
        self.try_parsing_after_exception = try_parsing_after_exception
        self.allow_quantity_overwrites = allow_quantity_overwrites

        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.retries_output_parsing = 1

    def create_temp_workdir(self) -> Path:
        """
        Create and return a fresh temporary working directory.

        Returns:
            Path to the newly created working directory.

        """

        return self.evaluation_working_directory.create()

    def with_parser(
        self,
        parser: Callable[..., QuantitiesT],
        *output_files: Path | str,
    ) -> Self:
        """
        Return a copy with a parser bound to its input files.

        The parser receives the resolved output paths as positional arguments
        in the same order as ``output_files``. Parser input files are also
        treated as required outputs and are watched before parsing begins.

        Args:
            parser: Callable that accepts the resolved output paths and returns
                a quantity dictionary.
            *output_files: Relative paths consumed by ``parser``.

        Returns:
            A new computer containing the additional parser binding.

        Raises:
            TypeError: If ``parser`` is not callable.
            ValueError: If no files are supplied, a path is absolute, or a
                relative path escapes the evaluation working directory.

        """
        if not callable(parser):
            msg = "The output parser must be callable."
            raise TypeError(msg)
        if not output_files:
            msg = "with_parser() requires at least one output file."
            raise ValueError(msg)

        files = _relative_paths(output_files)
        new = copy.copy(self)
        new._parser_bindings = (  # noqa: SLF001
            *self._parser_bindings,
            _ParserBinding(parser, files),
        )
        return new

    def wait_for(self, *output_files: Path | str) -> Self:
        """
        Return a copy that also waits for the supplied completion files.

        Completion files are not passed to parsers. They are useful for tools
        that create result files before the external computation is complete.

        Args:
            *output_files: Relative paths that must exist before parsing.

        Returns:
            A new computer containing the additional completion files.

        Raises:
            ValueError: If no files are supplied, a path is absolute, or a
                relative path escapes the evaluation working directory.

        """
        if not output_files:
            msg = "wait_for() requires at least one output file."
            raise ValueError(msg)

        files = _relative_paths(output_files)
        new = copy.copy(self)
        new._completion_files = tuple(  # noqa: SLF001
            dict.fromkeys((*self._completion_files, *files))
        )
        return new

    def _watched_files(self) -> tuple[Path, ...]:
        """Return each parser input and completion file once, in registration order."""
        parser_files = (
            output_file
            for binding in self._parser_bindings
            for output_file in binding.files
        )
        return tuple(dict.fromkeys((*parser_files, *self._completion_files)))

    def _wait_for_outputs(self, output_files: Iterable[Path]) -> None:
        """Wait synchronously until every required output file exists."""
        output_files = tuple(output_files)
        if all(output_file.exists() for output_file in output_files):
            return

        start = time.monotonic()
        while not all(output_file.exists() for output_file in output_files):
            if (
                self.wait_timeout is not None
                and time.monotonic() - start >= self.wait_timeout
            ):
                msg = f"Timed out waiting for {list(output_files)}"
                raise TimeoutError(msg)
            time.sleep(self.poll_interval)

    def _execute_steps(self, parameters: ParametersT, ctx: EvaluateContext) -> None:
        """Execute hooks and commands sequentially in registration order."""
        ctx.temp.commands = []
        ctx.temp.command_output_files = []
        ctx.temp.command_failure = None
        ctx.temp.current_step_index = None
        ctx.temp.current_step_type = None

        for step_index, step in enumerate(self._steps):
            ctx.temp.current_step_index = step_index

            if isinstance(step, _HookStep):
                ctx.temp.current_step_type = "hook"
                step.hook(parameters, ctx.temp.workdir, ctx)
                continue

            ctx.temp.current_step_type = "command"
            run_args = dict(step.subprocess_run_args)
            output_paths: dict[str, Path] | None = None
            manages_subprocess_output = any(
                option in run_args for option in ("capture_output", "stdout", "stderr")
            )
            if manages_subprocess_output and (
                step.stdout_file is not None or step.stderr_file is not None
            ):
                msg = (
                    "Command-specific stdout_file or stderr_file cannot be combined "
                    "with capture_output, stdout, or stderr in subprocess_run_args."
                )
                raise ValueError(msg)

            cmd = step.command(parameters, ctx.temp.workdir, ctx)
            ctx.temp.commands.append(cmd)

            try:
                with ExitStack() as output_files:
                    if not manages_subprocess_output:
                        stdout_file = step.stdout_file or Path(
                            f"command-{step_index}.stdout"
                        )
                        stderr_file = step.stderr_file or Path(
                            f"command-{step_index}.stderr"
                        )
                        output_paths = {
                            "stdout": ctx.temp.workdir / stdout_file,
                            "stderr": ctx.temp.workdir / stderr_file,
                        }
                        for output_path in set(output_paths.values()):
                            output_path.parent.mkdir(parents=True, exist_ok=True)

                        run_args["stdout"] = output_files.enter_context(
                            output_paths["stdout"].open("wb")
                        )
                        if output_paths["stderr"] == output_paths["stdout"]:
                            run_args["stderr"] = run_args["stdout"]
                        else:
                            run_args["stderr"] = output_files.enter_context(
                                output_paths["stderr"].open("wb")
                            )
                        ctx.temp.command_output_files.append(
                            {
                                "step_index": step_index,
                                **output_paths,
                            }
                        )

                    subprocess.run(  # noqa: S603
                        cmd,  # type: ignore
                        check=True,
                        cwd=ctx.temp.workdir,
                        **run_args,
                    )  # type: ignore
            except subprocess.CalledProcessError as exception:
                ctx.temp.command_failure = {
                    "step_index": step_index,
                    "exception_type": type(exception).__name__,
                    "message": str(exception),
                    "cmd": exception.cmd,
                    "returncode": exception.returncode,
                    "stdout_file": (
                        None if output_paths is None else output_paths["stdout"]
                    ),
                    "stderr_file": (
                        None if output_paths is None else output_paths["stderr"]
                    ),
                    "stdout": (
                        None
                        if exception.stdout is None
                        else _subprocess_output_to_text(exception.stdout)
                    ),
                    "stderr": (
                        None
                        if exception.stderr is None
                        else _subprocess_output_to_text(exception.stderr)
                    ),
                }
                if not self.try_parsing_after_exception:
                    raise

                logger.warning(
                    "Command step %d failed with %s: %s. "
                    "Will attempt to parse output files.",
                    step_index,
                    type(exception).__name__,
                    exception,
                )
                break

        ctx.temp.current_step_index = None
        ctx.temp.current_step_type = None

    def _write_crash_dump(self, ctx: EvaluateContext, exception: Exception) -> Path:
        """Write evaluation-level failure details and return the dump path."""
        dump_path = (
            self.evaluation_working_directory.base_working_directory
            / ctx.temp.workdir.name
        ).with_suffix(".dump")
        with dump_path.open("w") as dump_file:
            dump_file.write(f"Exception type: {type(exception).__name__}\n")
            dump_file.write(f"Exception: {exception}\n")

            if isinstance(exception, subprocess.CalledProcessError):
                dump_file.write("Subprocess failure:\n")
                dump_file.write(f"  cmd: {exception.cmd!r}\n")
                dump_file.write(f"  returncode: {exception.returncode}\n")
                if exception.stdout is not None:
                    stdout = _subprocess_output_to_text(exception.stdout)
                    dump_file.write(f"  stdout: {stdout}\n")
                if exception.stderr is not None:
                    stderr = _subprocess_output_to_text(exception.stderr)
                    dump_file.write(f"  stderr: {stderr}\n")

            dump_file.write("ctx.temp:\n")
            dump_file.write(pformat(vars(ctx.temp)))
            dump_file.write("\n")
        return dump_path

    def with_hook(
        self,
        hook: Callable[Concatenate[ParametersT, Path, EvaluateContext, ...], None],
        /,
        **kwargs: Any,
    ) -> Self:
        """
        Return a copy with a hook appended to the execution pipeline.

        The provided ``hook`` callable may accept additional keyword arguments
        beyond ``(parameters, workdir, ctx)``. These are bound via ``kwargs``
        and the resulting callable is executed at this position in the
        pipeline.

        This is a convenience wrapper around ``functools.partial`` that avoids
        requiring users to manually construct partial functions.

        Args:
            hook: Callable executed at this position in the pipeline. Must accept
                ``(parameters: Mapping[str, object], workdir: Path,
                ctx: EvaluateContext, ...)`` where any additional arguments
                are keyword-only.
            **kwargs: Keyword arguments to bind to ``hook``.

        Returns:
            A new computer containing the additional hook step.

        Example:
            >>> from chemfit.evaluation_working_directory import (
            ...     EvaluationWorkingDirectory,
            ... )
            >>> from chemfit.external_computer import ExternalQuantityComputer
            >>> workdir = EvaluationWorkingDirectory("workdir")
            >>> computer = ExternalQuantityComputer(workdir)
            >>> def write_input(parameters, workdir, ctx, *, template_path):
            ...     ...
            >>> computer2 = computer.with_hook(
            ...     write_input,
            ...     template_path="INCAR.template",
            ... )

        Note:
            Additional arguments must be keyword-only in ``hook``.

        """
        bound_hook = cast("HookType[ParametersT]", functools.partial(hook, **kwargs))
        new = copy.copy(self)
        new._steps = (  # noqa: SLF001
            *self._steps,
            _HookStep(bound_hook),
        )
        return new

    def with_cmd(
        self,
        command: Callable[
            Concatenate[ParametersT, Path, EvaluateContext, ...], list[str]
        ],
        /,
        *,
        stdout_file: Path | str | None = None,
        stderr_file: Path | str | None = None,
        subprocess_run_args: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Return a copy with a command appended to the execution pipeline.

        The provided ``command`` may accept additional keyword arguments
        beyond ``(parameters, workdir, ctx)``. These are bound via ``kwargs``
        and the resulting callable is executed at this position in the
        pipeline.

        This is a convenience wrapper around ``functools.partial`` that avoids
        requiring users to manually construct partial functions.

        Args:
            command: Callable used to construct the command. Must accept
                ``(parameters: Mapping[str, object], workdir: Path,
                ctx: EvaluateContext, ...)`` where any additional arguments
                are keyword-only.
            stdout_file: Relative path within the evaluation working directory
                to which this command's standard output is streamed. Defaults
                to ``command-N.stdout``, where ``N`` is the step index.
            stderr_file: Relative path within the evaluation working directory
                to which this command's standard error is streamed. Defaults
                to ``command-N.stderr``, where ``N`` is the step index.
            subprocess_run_args: Additional keyword arguments forwarded to
                ``subprocess.run`` for this command. ``cwd`` and ``check`` are
                managed by ChemFit and cannot be supplied. Explicit output
                handling cannot be combined with ``stdout_file`` or
                ``stderr_file``.
            **kwargs: Keyword arguments to bind to ``command``.

        Returns:
            A new computer containing the additional command step.

        Example:
            >>> from chemfit.evaluation_working_directory import (
            ...     EvaluationWorkingDirectory,
            ... )
            >>> from chemfit.external_computer import ExternalQuantityComputer
            >>> workdir = EvaluationWorkingDirectory("workdir")
            >>> computer = ExternalQuantityComputer(workdir)
            >>> def command(parameters, workdir, ctx, *, executable):
            ...     return [executable, "input.dat"]
            >>> computer2 = computer.with_cmd(
            ...     command,
            ...     stdout_file="simulation.out",
            ...     stderr_file="simulation.err",
            ...     subprocess_run_args={"timeout": 3600},
            ...     executable="simulation",
            ... )

        Note:
            Additional arguments must be keyword-only in ``command``.

        """

        bound_command = cast(
            "CommandType[ParametersT]", functools.partial(command, **kwargs)
        )
        stdout_path = (
            None if stdout_file is None else _relative_paths((stdout_file,))[0]
        )
        stderr_path = (
            None if stderr_file is None else _relative_paths((stderr_file,))[0]
        )
        run_args = {} if subprocess_run_args is None else dict(subprocess_run_args)
        reserved_args = {"check", "cwd"}.intersection(run_args)
        if reserved_args:
            names = ", ".join(sorted(reserved_args))
            msg = f"ChemFit manages these subprocess arguments: {names}."
            raise ValueError(msg)
        manages_subprocess_output = any(
            option in run_args for option in ("capture_output", "stdout", "stderr")
        )
        if manages_subprocess_output and (
            stdout_path is not None or stderr_path is not None
        ):
            msg = (
                "Command-specific stdout_file or stderr_file cannot be combined "
                "with capture_output, stdout, or stderr in subprocess_run_args."
            )
            raise ValueError(msg)

        new = copy.copy(self)
        new._steps = (  # noqa: SLF001
            *self._steps,
            _CommandStep(bound_command, stdout_path, stderr_path, run_args),
        )
        return new

    def _run(
        self,
        parameters: ParametersT,
        ctx: EvaluateContext,
    ) -> QuantitiesT:
        """
        Execute an external workflow and parse its output files.

        This method implements the core logic:

        1. Create a temporary working directory.
        2. Execute each registered hook or command in order.
        3. Wait until all configured output files exist (or timeout).
        4. Run the configured output parsers and merge the resulting
           quantity dictionaries.
        5. Optionally delete the temporary working directory.

        Args:
            parameters: Parameter dictionary for this evaluation.
            ctx: Evaluation context for this call. The temporary working
                directory, resolved output file paths, and executed commands
                are stored in ``ctx.temp``.

        Returns:
            Dictionary of parsed quantities.

        Side Effects:
            - Creates and stores ``ctx.temp.workdir``.
            - Stores the resolved output file paths in
            ``ctx.temp.output_files``.
            - Stores executed commands in ``ctx.temp.commands``.
            - Creates and optionally deletes a temporary working directory.
            - Runs an external subprocess.

        Raises:
            TimeoutError: If the configured output files do not appear
                before ``wait_timeout`` expires.
            Exception: If subprocess execution fails, output parsing fails,
                or temporary working-directory management fails.

        ----------------------
        IMPORTANT WARNINGS
        ----------------------

        **1. Use with sbatch / job schedulers**

        Commands like ``sbatch`` (SLURM), ``qsub`` (Torque/PBS), ``bsub`` (LSF),
        or any *queueing* submission command typically return **immediately** from
        ``subprocess.run``. The actual compute job may start minutes or hours later.

        - Later execution steps run once the submission command itself returns;
          ChemFit does not wait for the submitted job between steps.
        - Output polling and `wait_timeout` begin after the complete execution
          pipeline finishes, *not* when a submitted job begins executing.
        - This almost always causes a timeout if users submit through a scheduler.

        **Recommended workaround:**
        - Modify your job script to create a **completion flag file**, e.g.:

            .. code-block:: bash

                # at end of your SLURM job script
                touch task.done

        - Then configure ``wait_for("task.done")`` in addition to the files
        registered with output parsers.

        This ensures `ExternalQuantityComputer` waits for job completion rather than
        the output file prematurely appearing or remaining absent.

        **2. Timeout awareness**

        The `wait_timeout` applies to the *combined* waiting time after all
        execution steps finish.

        - Use very large timeouts (or None) or a reliable completion flag for scheduler-based workloads.
        - If timeout is too small, you will get a `TimeoutError`.

        **3. Programs that stream output**

        Many scientific programs create output files early and append to them as
        they run. Because this class only checks **existence**, not completeness:

        - An output file may exist while the job is still running.
        - Parsers may read incomplete or partially written files.

        **Solutions:**
        - Use a completion marker file (as above).
        - Or implement a parser that verifies file completeness (checksum, fixed-size,
        closing footer, etc.).

        **4. Crash diagnostics and dump files**

        If any execution step, output wait, or parser fails, this class can
        optionally write a diagnostic dump file to help with debugging.

        When ``write_dump_file_after_crash=True``, a dump file is written to the
        base working directory using the name of the temporary work directory
        with the suffix ``.dump``.

        For example:

            base_working_directory/
                7f4d9c1a-8cbb-4b6f-b88c-8a1c53eae6c3/
                7f4d9c1a-8cbb-4b6f-b88c-8a1c53eae6c3.dump

        The dump file contains diagnostic information including:

        - the final exception;
        - the current execution step and commands executed so far;
        - paths to the streamed subprocess output files;
        - captured subprocess diagnostics, when output capture was explicitly
          enabled and the final exception is a ``subprocess.CalledProcessError``;
        - any recoverable command failure recorded before a later failure;
        - the contents of ``ctx.temp`` at the time of failure

        This information often provides enough context to diagnose failures
        without needing to reproduce the run manually.

        The dump file is especially useful when:

        - evaluations run inside large optimization loops
        - runs are executed on remote clusters
        - temporary working directories are automatically deleted

        If ``keep_temp_workdir_after_crash=True``, the temporary working
        directory is preserved for manual inspection. Otherwise it may be
        deleted depending on the configuration.

        Keeping both the dump file and the working directory can make it much
        easier to reproduce and debug failed evaluations.

        """

        with self.evaluation_working_directory.isolated_workdir() as workdir:
            ctx.temp.workdir = workdir
            try:
                watched_files = self._watched_files()
                ctx.temp.output_files = [
                    workdir / output_file for output_file in watched_files
                ]
                self._execute_steps(parameters, ctx)

                self._wait_for_outputs(ctx.temp.output_files)

                res: dict[str, Any] = {}
                for index, binding in enumerate(self._parser_bindings):
                    parser_inputs = tuple(
                        workdir / output_file for output_file in binding.files
                    )
                    # First we perform the retries while silencing all exceptions
                    for _ in range(self.retries_output_parsing):
                        try:
                            parsed_quantities = binding.parser(*parser_inputs)
                            break
                        except Exception:  # noqa: S112
                            continue
                    else:
                        # The final attempt propagates its exception so the
                        # diagnostic handler records the actual parser failure.
                        parsed_quantities = binding.parser(*parser_inputs)

                    merge_quantities(
                        res,
                        parsed_quantities,
                        source=f"output parser {index}",
                        allow_overwrites=self.allow_quantity_overwrites,
                    )

            except Exception as e:
                msg = (
                    "Exception in `_run` of ExternalQuantityComputer.\n"
                    f"  ctx.temp = {ctx.temp}"
                )

                if self.write_dump_file_after_crash:
                    try:
                        dump_path = self._write_crash_dump(ctx, e)
                        msg += f"\nWrote dump file to `{dump_path}`."
                    except Exception as dump_exception:
                        msg += (
                            "\nCould not write a crash dump because of "
                            f"{dump_exception}."
                        )

                if hasattr(e, "add_note"):  # we dont have `add_note` in 3.10
                    e.add_note(msg)

                raise

        return cast("QuantitiesT", res)
