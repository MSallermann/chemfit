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
    """Protocol for a function parses an output file and obtains quantities."""

    def __call__(self, output_files: list[Path]) -> dict[str, Any]:
        """Parse the output files and retrieve the quantities."""
        ...


@runtime_checkable
class PreSubmitHook(Protocol):
    """Protocol for running things before the command is submitted."""

    def __call__(self, parameters: dict[str, Any], workdir: Path) -> None:
        """Pre-submit hook."""
        ...


@runtime_checkable
class PostSubmitHook(Protocol):
    """Protocol for running things after the command has run."""

    def __call__(self, parameters: dict[str, Any], workdir: Path) -> None:
        """Post-submit hook."""
        ...


class FileBasedQuantityComputer(QuantityComputer):
    def __init__(
        self,
        output_files: list[Path],
        executable_cmd: Callable[[dict[str, Any], Path], list[str]],
        output_parsers: list[OutputParser] | OutputParser,
        base_working_directory: Path,
        presubmit_hook: PreSubmitHook | None = None,
        wait_timeout: float = 500.0,
        poll_interval: float = 1,
        subprocess_run_args: dict | None = None,
        delete_temp_workdirs: bool = True,
    ):
        """
        Initialize a Computer that can create files and parse quantities from files.

        Args:
            output_file (Path): Path to output file. These need to be file paths **relative** to the working directory.
            executable_cmd (str): Command that will be executed to create the file
            wait_timeout: in seconds
            poll_interval: check for file (in seconds)

        """
        super().__init__()
        self.output_files = output_files
        self.base_working_directory = base_working_directory

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

    def create_temp_workdir(self) -> Path:
        name = str(uuid.uuid4())
        temp_workdir = self.base_working_directory / name
        temp_workdir.mkdir(exist_ok=False, parents=True)
        return temp_workdir

    def _compute(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,
    ) -> dict[str, Any]:
        # Create a temporary working directory
        ctx.temp.workdir = self.create_temp_workdir()

        try:
            ctx.temp.output_files = [ctx.temp.workdir / o for o in self.output_files]

            if self.presubmit_hook is not None:
                self.presubmit_hook(parameters, ctx.temp.workdir)

            cmd = self.executable_cmd(parameters, ctx.temp.workdir)

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
                    f"Exception in `subprocess.run` of FileBasedComputer."
                    f"  stderr (if captured) = {e.stderr}"
                    f"  stdout (if captured) = {e.stdout}"
                )
                logger.exception(msg)
                raise e

            # Block here until file appears (or timeout)
            # The main reason to implement this extra check is to eventually support remote execution, e.g. on clusters
            # A script submitted with `sbatch` for example would immediately return from `subprocess.run`, but the necessary output files
            # would not be present until the submitted script has actually run on one of the compute nodes.
            # Therefore, waiting until the output files are actually present is a valid strategy.
            # Of course, we might still run into problems in the case of output files which get continuously appended to.
            # These could be present already, but not complete and thus fool us into thinking that the script has completed it's run.
            if all(o.exists() for o in self.output_files):
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
                res.update(o(ctx.temp.output_files))

            return res
        finally:
            if self.delete_temp_workdirs:
                shutil.rmtree(ctx.temp.workdir)

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
