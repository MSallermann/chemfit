from __future__ import annotations

import abc
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from chemfit.abstract_objective_function import QuantityComputer

if TYPE_CHECKING:
    from pathlib import Path


class FileBasedQuantityComputer(QuantityComputer):
    def __init__(
        self,
        output_file: Path,
        executable_cmd: str | Callable[[dict[str, Any], str]],
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
        self.output_file = output_file

        if isinstance(executable_cmd, str):
            self.executable_cmd = lambda _: executable_cmd
        else:
            self.executable_cmd = executable_cmd

        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval

    def _compute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        self.pre_submit_hook(parameters)

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
            err_message = f"Timed out waiting for {self.output_file}"
            raise TimeoutError(err_message)

        return self.parse_outputs()

    def pre_submit_hook(self, parameters: dict[str, Any]):
        """Modifies the executable command using the parameters."""

    def _file_watch_loop(self, ready: threading.Event, stop: threading.Event) -> None:
        # The output file has been created
        if self.output_file.exists():
            ready.set()
            return

        # timeout
        while not stop.is_set():
            if self.output_file.exists():
                ready.set()
                return
            time.sleep(self.poll_interval)

    @abc.abstractmethod
    def parse_outputs(self) -> dict[str, Any]:
        """Compute dictionary of quantities after parsing an output file."""
        ...
