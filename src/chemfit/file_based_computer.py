from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from chemfit.abstract_objective_function import EvaluateContext, QuantityComputer
from chemfit.utils import check_protocol

if TYPE_CHECKING:
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
        executable_cmd: Callable[[dict[str, Any]], Iterable[str]] | Iterable[str] | str,
        output_parsers: list[OutputParser] | OutputParser,
        presubmit_hook: PreSubmitHook | None = None,
        working_directory: Path | None = None,
        wait_timeout: float = 500.0,
        poll_interval: float = 1,
        subprocess_run_args: dict | None = None,
        check_if_output_exits: bool = True,
        clear_output_before_compute: bool = False,
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

        self.check_if_output_exists = check_if_output_exits
        self.clear_output_before_compute = clear_output_before_compute

        if subprocess_run_args is None:
            self.subprocess_run_args = {}
        else:
            self.subprocess_run_args = subprocess_run_args

        if isinstance(executable_cmd, str):
            if len(executable_cmd.split()) != 1:
                msg = "You passed a string with whitespace for the executable command, presumably a program with command line arguments. Arguments need to be passed as separate list items. e.g exectuable_cmd = ['myprogram','-f', 'arg2'], **not** exectuable_cmd = 'myprogram -f arg2'"
                raise Exception(msg)
            executable_cmd = [executable_cmd]
        elif isinstance(executable_cmd, Iterable):
            self.executable_cmd = lambda _: executable_cmd
        else:
            self.executable_cmd = executable_cmd

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
        self.working_directory = working_directory

    def _compute(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        if self.presubmit_hook is not None:
            self.presubmit_hook(parameters)

        if self.clear_output_before_compute:
            for o in self.output_files:
                if o.exists():
                    o.unlink()

        if self.check_if_output_exists:
            for o in self.output_files:
                if o.exists():
                    logger.warning(
                        f"The outputfiles {o} exist already. This may lead to unforeseen behaviour. To disable these warnings either set `check_if_output_exists` to False or `clear_output_before_compute` to True"
                    )

        cmd = self.executable_cmd(parameters)

        # Spin up the watcher BEFORE running the command to avoid race conditions
        ready = threading.Event()
        stop = threading.Event()
        watcher = threading.Thread(
            target=self._file_watch_loop, args=(ready, stop), daemon=True
        )
        watcher.start()

        # Run the external program (raises on non-zero exit)
        subprocess.run(  # noqa: S603
            cmd,  # type: ignore
            check=True,
            cwd=self.working_directory,
            **self.subprocess_run_args,
        )  # type: ignore

        # Block here until file appears (or timeout)
        # The main reason to implement this extra check is to eventually support remote execution, e.g. on clusters
        # A script submitted with sbatch for example would immediately return from `subprocess.run`, but the necessary output files
        # would not be present until the submitted script has actually run on one of the compute nodes.
        # Therefore, waiting until the output files are actually present is a valid strategy.
        # Of course, we might still run into problems in the case of output files wich get continousely appended to.
        # These could be present already, but not complete and thus fool us into thinking that the script has completed it's run.
        if all(o.exists() for o in self.output_files):
            # We do one immediate check on the main thread
            stop.set()
        else:
            ok = ready.wait(timeout=self.wait_timeout)
            stop.set()
            watcher.join(timeout=1)

            if not ok:
                err_message = f"Timed out waiting for {self.output_files}"
                raise TimeoutError(err_message)

        res = {}
        for o in self.output_parsers:
            res.update(o(self.output_files))

        return res

    def _file_watch_loop(self, ready: threading.Event, stop: threading.Event) -> None:
        # check if files are there
        while not stop.is_set():
            files_created = all(o.exists() for o in self.output_files)
            if files_created:
                ready.set()
                return
            time.sleep(self.poll_interval)
