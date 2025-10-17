import abc
from pathlib import Path
from typing import Any

from chemfit.abstract_objective_function import QuantityComputer


class FileBasedQuantityComputer(QuantityComputer):
    def __init__(self, output_file: Path, executable_cmd: str):
        """
        Initialize a Computer that can create files and quantities from files.

        Args:
            output_file (Path): Path to output file
            executable_cmd (str): Command that will be executed to create the file

        """
        super().__init__()
        self.output_file = output_file
        self.executable_cmd = executable_cmd

    def _compute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        self.pre_submit_hook(parameters)

        # submit cmd
        # ... wait for output file

        # return quantities
        return self.parse_outputs()

    def pre_submit_hook(self, parameters: dict[str, Any]): ...

    @abc.abstractmethod
    def parse_outputs(self) -> dict[str, Any]:
        """Compute dictionary of quantities after parsing an output file."""
        ...
