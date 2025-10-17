from __future__ import annotations

import abc
import subprocess
from typing import TYPE_CHECKING, Any, Callable

from chemfit.abstract_objective_function import QuantityComputer

if TYPE_CHECKING:
    from pathlib import Path


class FileBasedQuantityComputer(QuantityComputer):
    def __init__(
        self, output_file: Path, executable_cmd: str | Callable[[dict[str, Any], str]]
    ):
        """
        Initialize a Computer that can create files and quantities from files.

        Args:
            output_file (Path): Path to output file
            executable_cmd (str): Command that will be executed to create the file

        """
        super().__init__()
        self.output_file = output_file

        if isinstance(executable_cmd, str):
            self.executable_cmd = lambda _: executable_cmd
        else:
            self.executable_cmd = executable_cmd

    def _compute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        self.pre_submit_hook(parameters)

        cmd = self.executable_cmd(parameters)

        # Submit the command
        subprocess.run(cmd, check=True, shell=True)  # noqa: S602
        # ... wait for output file

        # return quantities
        return self.parse_outputs()

    def pre_submit_hook(self, parameters: dict[str, Any]): ...

    @abc.abstractmethod
    def parse_outputs(self) -> dict[str, Any]:
        """Compute dictionary of quantities after parsing an output file."""
        ...
