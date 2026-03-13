from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from chemfit.abstract_objective_function import EvaluateContext, QuantityComputer
from chemfit.utils import check_protocol

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class OutputParser(Protocol):
    """Protocol for parsing output files into a quantity dictionary."""

    def __call__(self, output_files: list[Path]) -> dict[str, Any]:
        """
        Parse the output files and retrieve the quantities.

        Args:
            output_files (list[Path]): List of paths to output files. These
                are typically located in the working directory of a single
                evaluation.

        Returns:
            dict[str, Any]: Dictionary of parsed quantities.

        """
        ...


@runtime_checkable
class PreSubmitHook(Protocol):
    """Protocol for running things before the command is submitted."""

    def __call__(self, parameters: dict[str, Any], workdir: Path) -> None:
        """
        Run pre-submit actions.

        Args:
            parameters (dict[str, Any]): Parameter dictionary for the evaluation.
            workdir (Path): Temporary working directory for this evaluation.

        """


@runtime_checkable
class PostSubmitHook(Protocol):
    """Protocol for running things after the command has run."""

    def __call__(self, parameters: dict[str, Any], workdir: Path) -> None:
        """
        Protocol for running things after the command has run.

        Args:
            parameters (dict[str, Any]): Parameter dictionary for the evaluation.
            workdir (Path): Temporary working directory for this evaluation.

        """
        ...


class FileBasedQuantityComputer(QuantityComputer):
    def __init__(
        self,
        output_files: list[Path],
        executable_cmd: Callable[[dict[str, Any], Path], list[str]],
        output_parsers: list[OutputParser] | OutputParser,
        base_working_directory: Path,
        presubmit_hook: PreSubmitHook | None = None,
        wait_timeout: float | None = 500.0,
        poll_interval: float = 1,
        subprocess_run_args: dict | None = None,
        delete_temp_workdirs: bool = True,
        write_dump_file_after_crash: bool = True,
        keep_temp_workdir_after_crash: bool = True,
    ):
        """
        Initialize a file-based quantity computer.

        This quantity computer evaluates parameters by creating a temporary
        working directory, executing an external command, waiting for the
        expected output files to appear, and parsing those files into a
        quantity dictionary.

        Args:
            output_files (list[Path]):
                Paths to output files that are expected to be created by
                the external command. These paths must be **relative** to
                the working directory; absolute paths are not allowed.
            executable_cmd (Callable[[dict[str, Any], Path], list[str]]):
                Callable that constructs the command to execute. It receives
                the parameter dictionary and the temporary working directory,
                and must return a list of strings suitable for
                ``subprocess.run``.
            output_parsers (list[OutputParser] | OutputParser):
                One or more output parsers called after the external command
                completes and the output files exist. Each parser receives
                the list of output file paths and returns a dictionary of
                quantities. The results of all parsers are merged.
            base_working_directory (Path):
                Base directory under which temporary working directories
                will be created, one per evaluation.
            presubmit_hook (PreSubmitHook | None, optional):
                Optional hook executed before the external command is run.
                It can be used to prepare input files, templates, etc.
            wait_timeout (float, optional):
                Maximum time in seconds to wait for all output files to
                appear. Defaults to 500.0 seconds.
            poll_interval (float, optional):
                Interval in seconds between checks for output file creation.
                Defaults to 1 second.
            subprocess_run_args (dict | None, optional):
                Additional keyword arguments forwarded to
                ``subprocess.run`` (e.g. ``capture_output=True``).
                Defaults to None.
            delete_temp_workdirs (bool, optional):
                Whether to delete temporary working directories after each
                evaluation. Defaults to True.
            write_dump_file_after_crash: Whether to write a dump file with
                subprocess output when command execution fails.
            keep_temp_workdir_after_crash: Whether to keep the temporary
                working directory for inspection after a failed evaluation.


        Raises:
            Exception: If any path in `output_files` is absolute rather
                than relative.

        """

        super().__init__()

        self.output_files = output_files
        self.base_working_directory = base_working_directory
        self.write_dump_file_after_crash = write_dump_file_after_crash
        self.keep_temp_workdir_after_crash = keep_temp_workdir_after_crash

        # We need to make sure none of the output files is absolute.
        # The reason for this is that, to facilitate multiple concurrent evaluations,
        # we may have to create temporary working directories so that concurrent runs do not mess with each others outputs.
        # Then the relative paths are used to place the output files relative ot the temporary working directory.
        if any(of.is_absolute() for of in self.output_files):
            msg = "One of the output files is an absolute path. All output paths need to be relative to the working directory."
            raise Exception(msg)

        if subprocess_run_args is None:
            self.subprocess_run_args = {}
        else:
            self.subprocess_run_args = subprocess_run_args

        self.executable_cmd: Callable[[dict[str, Any], Path], list[str]] = (
            executable_cmd
        )

        # Make sure that, if a single OutputParser has been passed, we turn it into a list with on element
        if isinstance(output_parsers, OutputParser):
            self.output_parsers = [output_parsers]
        else:
            self.output_parsers = output_parsers

        for o in self.output_parsers:
            check_protocol(o, OutputParser)

        self.presubmit_hook = presubmit_hook
        check_protocol(self.presubmit_hook, PreSubmitHook)

        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.delete_temp_workdirs = delete_temp_workdirs
        self.retries_output_parsing = 1

    def create_temp_workdir(self) -> Path:
        """
        Create and return a fresh temporary working directory.

        Returns:
            Path to the newly created working directory.

        """

        name = str(uuid.uuid4())
        temp_workdir = self.base_working_directory / name
        temp_workdir.mkdir(exist_ok=False, parents=True)
        return temp_workdir

    def build_cmd(self, parameters: dict[str, Any], ctx: EvaluateContext) -> list[str]:
        """
        Build the external command for the current evaluation.

        Args:
            parameters: Parameter dictionary for the current evaluation.
            ctx: Evaluation context whose temporary working directory is
                used when constructing the command.

        Returns:
            Command to execute, formatted for ``subprocess.run``.

        """
        return self.executable_cmd(parameters, ctx.temp.workdir)

    def _compute(  # noqa: PLR0912, PLR0915
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,
    ) -> dict[str, Any]:
        """
        Execute external command and parse parse its output files.

        This method implements the core logic:

        1. Create a temporary working directory.
        2. Optionally run a pre-submit hook.
        3. Build and execute the external command via ``subprocess.run``.
        4. Wait until all configured output files exist (or timeout).
        5. Run the configured output parsers and merge the resulting
           quantity dictionaries.
        6. Optionally delete the temporary working directory.

        Args:
            parameters: Parameter dictionary for this evaluation.
            ctx: Evaluation context for this call. The temporary working
                directory, resolved output file paths, and executed command
                are stored in ``ctx.temp``.

        Returns:
            Dictionary of parsed quantities.

        Side Effects:
            - Creates and stores ``ctx.temp.workdir``.
            - Stores the resolved output file paths in
            ``ctx.temp.output_files``.
            - Stores the executed command in ``ctx.temp.cmd``.
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

        - The `wait_timeout` starts counting **as soon as `subprocess.run` returns**,
        *not* when the job begins executing.
        - This almost always causes a timeout if users submit through a scheduler.

        **Recommended workaround:**
        - Modify your job script to create a **completion flag file**, e.g.:

            .. code-block:: bash

                # at end of your SLURM job script
                touch task.done

        - Then configure `output_files=[Path("task.done")]` **in addition to** your
        real output files.

        This ensures `FileBasedQuantityComputer` waits for job completion rather than
        the output file prematurely appearing or remaining absent.

        **2. Timeout awareness**

        The `wait_timeout` applies to the *combined* waiting time after the external
        command returns.

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

        If the external command fails (i.e. ``subprocess.run`` raises
        ``subprocess.CalledProcessError``), this class can optionally write a
        diagnostic dump file to help with debugging.

        When ``write_dump_file_after_crash=True``, a dump file is written to the
        base working directory using the name of the temporary work directory
        with the suffix ``.dump``.

        For example:

            base_working_directory/
                7f4d9c1a-8cbb-4b6f-b88c-8a1c53eae6c3/
                7f4d9c1a-8cbb-4b6f-b88c-8a1c53eae6c3.dump

        The dump file contains diagnostic information including:

        - the captured ``stderr`` of the external program (if available)
        - the captured ``stdout`` of the external program (if available)
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

        # Create a temporary working directory
        ctx.temp.workdir = self.create_temp_workdir()

        try:
            ctx.temp.output_files = [ctx.temp.workdir / o for o in self.output_files]

            if self.presubmit_hook is not None:
                self.presubmit_hook(parameters, ctx.temp.workdir)

            cmd = self.build_cmd(parameters, ctx)

            ctx.temp.cmd = cmd

            # Spin up the watcher BEFORE running the command to avoid race conditions
            ready = threading.Event()
            stop = threading.Event()
            watcher = threading.Thread(
                target=self._file_watch_loop,
                args=(ctx.temp.output_files, ready, stop),
                daemon=True,
            )
            watcher.start()

            try:
                # Run the external program (raises on non-zero exit)
                subprocess.run(  # noqa: S603
                    cmd,  # type: ignore
                    check=True,
                    cwd=ctx.temp.workdir,
                    **self.subprocess_run_args,
                )  # type: ignore
            except subprocess.CalledProcessError as e:
                msg = (
                    f"Exception in `subprocess.run` of FileBasedQuantityComputer.\n"
                    f"  stderr (if captured) = {e.stderr.decode('utf-8')}\n"
                    f"  ctx.temp = {ctx.temp}"
                )

                # Try to write a dump file
                if self.write_dump_file_after_crash:
                    dump_path = (
                        self.base_working_directory / ctx.temp.workdir.name
                    ).with_suffix(".dump")
                    try:
                        with dump_path.open("w") as f:
                            f.write("Stderr:\n")
                            f.write(e.stderr.decode("utf-8"))
                            f.write("Stdout:\n")
                            f.write(e.stdout.decode("utf-8"))
                            f.write("ctx.temp:\n")
                            f.write(f"{ctx.temp}")
                        msg += f"\nWrote dump file to `{dump_path}`."
                    except Exception as exc_dump:
                        msg += f"\nCould not write dump file to `{dump_path}`, because of {exc_dump}."
                raise Exception(msg) from e

            # Block here until file appears (or timeout)
            # The main reason to implement this extra check is to eventually support remote execution, e.g. on clusters
            # A script submitted with `sbatch` for example would immediately return from `subprocess.run`, but the necessary output files
            # would not be present until the submitted script has actually run on one of the compute nodes.
            # Therefore, waiting until the output files are actually present is a valid strategy.
            # Of course, we might still run into problems in the case of output files which get continuously appended to.
            # These could be present already, but not complete and thus fool us into thinking that the script has completed it's run.
            if all(o.exists() for o in ctx.temp.output_files):
                # We do one immediate check on the main thread
                stop.set()
            else:
                ok = ready.wait(timeout=self.wait_timeout)
                stop.set()
                watcher.join(timeout=1)

                if not ok:
                    err_message = f"Timed out waiting for {ctx.temp.output_files}"
                    raise TimeoutError(err_message)

            res = {}
            for o in self.output_parsers:
                success = False
                # First we perform the retries while silencing all exceptions
                for _ in range(self.retries_output_parsing):
                    try:
                        res.update(o(ctx.temp.output_files))
                        success = True
                        break
                    except Exception as e:  # noqa: F841, S112
                        continue

                # If we have not succeeded so far, the (retries + 1)th (aka the last)
                # attempt is made without a try block, so that we get to handle the actual exception
                if not success:
                    res.update(o(ctx.temp.output_files))

        except Exception as e:
            msg = (
                "Exception in `_compute` of FileBasedQuantityComputer.\n"
                f"  ctx.temp = {ctx.temp}"
            )

            if self.delete_temp_workdirs and not self.keep_temp_workdir_after_crash:
                shutil.rmtree(ctx.temp.workdir)
            else:
                msg += (
                    f"\nKeeping temporary workdir '{ctx.temp.workdir}' for inspection"
                )

            raise Exception(msg) from e
        else:
            if self.delete_temp_workdirs:
                shutil.rmtree(ctx.temp.workdir)

        return res

    def _file_watch_loop(
        self,
        output_files: Iterable[Path],
        ready: threading.Event,
        stop: threading.Event,
    ) -> None:
        # check if files are there
        while not stop.is_set():
            files_created = all(o.exists() for o in output_files)
            if files_created:
                ready.set()
                return
            time.sleep(self.poll_interval)
