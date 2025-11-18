from __future__ import annotations

import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from chemfit.abstract_objective_function import QuantityComputer
from chemfit.utils import check_protocol

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class OutputParser(Protocol):
    """Protocol for a function parses an output file and obtains quantities."""

    def __call__(self, output_files: list[Path]) -> dict[str, Any]:
        """Parse the output files and retrieve the quantities."""
        ...


@runtime_checkable
class PreSubmitHook(Protocol):
    """Protocol for running things before the command is submitted."""

    def __call__(self, parameters: dict[str, Any]) -> None:
        """Pre-submit hook."""
        ...


@runtime_checkable
class PostSubmitHook(Protocol):
    """Protocol for running things after the command has run."""

    def __call__(self, parameters: dict[str, Any]) -> None:
        """Post-submit hook."""
        ...


class FileBasedQuantityComputer(QuantityComputer):
    def __init__(
        self,
        output_files: list[Path],
        executable_cmd: str | Callable[[dict[str, Any]], str],
        output_parser: OutputParser,
        presubmit_hook: PreSubmitHook | None = None,
        working_directory: Path | None = None,
        wait_timeout: float = 500.0,
        poll_interval: float = 60,
    ):
        """
        Initialize a Computer that can create files and quantities from files.

        Args:
            output_file (Path): Path to output file
            executable_cmd (str): Command that will be executed to create the file
            wait_timeout: in seconds
            poll_interval: check for file (in seconds)

        """
        super().__init__()
        self.output_files = output_files

        if isinstance(executable_cmd, str):
            self.executable_cmd = lambda _: executable_cmd
        else:
            self.executable_cmd = executable_cmd

        self.output_parser = output_parser
        check_protocol(self.output_parser, OutputParser)
        self.presubmit_hook = presubmit_hook
        check_protocol(self.presubmit_hook, PreSubmitHook)
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.working_directory = working_directory

    def _compute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        if self.presubmit_hook is not None:
            self.presubmit_hook(parameters)

        cmd = self.executable_cmd(parameters)

        # Spin up the watcher BEFORE running the command to avoid race conditions
        ready = threading.Event()
        stop = threading.Event()
        watcher = threading.Thread(
            target=self._file_watch_loop, args=(ready, stop), daemon=True
        )
        watcher.start()

        # Run the external program (raises on non-zero exit)
        proc = subprocess.run(cmd, check=True, shell=True, cwd=self.working_directory)  # noqa: S602

        # Block here until file appears (or timeout)
        # The main reason to implement this extra check is to eventually support remote execution, e.g. on clusters
        # A script submitted with sbatch for example would immediately return from `subprocess.run`, but the necessary output files
        # would not be present until the submitted script has actually run on one of the compute nodes.
        # Therefore, waiting until the output files are actually present is a valid strategy.
        # Of course, we might still run into problems in the case of output files wich get continousely appended to.
        # These could be present already, but not complete and thus fool us into thinking that the script has completed it's run.
        ok = ready.wait(timeout=self.wait_timeout)
        stop.set()
        watcher.join(timeout=1)

        if not ok:
            err_message = f"Timed out waiting for {self.output_files}"
            raise TimeoutError(err_message)

        return self.output_parser(self.output_files)

    def _file_watch_loop(self, ready: threading.Event, stop: threading.Event) -> None:
        # check if files are there
        while not stop.is_set():
            files_created = all(o.exists() for o in self.output_files)
            if files_created:
                ready.set()
                return
            time.sleep(self.poll_interval)
