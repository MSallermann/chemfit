from __future__ import annotations

import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from chemfit.abstract_objective_function import QuantityComputer

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class OutputParser(Protocol):
    """Protocol for a function parses an output file and obtains quantities."""

    def __call__(self, output_files: list[Path]) -> dict[str, Any]:
        """Parse the output files and retrieve the quantities."""
        ...


@runtime_checkable
class PreCommitHook(Protocol):
    """Protocol for running things before the command is submitted."""

    def __call__(self, parameters: dict[str, Any]) -> None:
        """Pre-commit hook."""
        ...


class FileBasedQuantityComputer(QuantityComputer):
    def __init__(
        self,
        output_files: list[Path],
        executable_cmd: str | Callable[[dict[str, Any]], str],
        output_parser: OutputParser,
        presubmit_hook: PreCommitHook | None = None,
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
        self.presubmit_hook = presubmit_hook
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval

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
        subprocess.run(cmd, check=True, shell=True)  # noqa: S602

        # Block here until file appears (or timeout)
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
