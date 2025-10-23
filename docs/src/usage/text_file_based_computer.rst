=====================================
Text-File-Based Quantity Computers
=====================================

Often, one would like to fit simulation data (generated
from a dictionary of parameters) to reference data,
such as the density. Many simulation engines rely on
text-file-based inputs .Therefore, the :class:`FileBasedQuantityComputer`
is designed to handle situations where you want, for instance, to run
simulations which use parameters, which are subsequently postprocessed to
generate quantities.

This is not limited to simulations, but any use-case where files
can be generated as outputs from parameters.

Consider a simple example, where we have a script that has a command-line interface, and which outputs the values of :math:`A (x-2)^2`
in an output file, for a predefined range of :math:`x`. The prefactor :math:`A` will be our fitting parameter.
The full script can be found at `<https://github.com/MSallermann/chemfit/tests/input/square_function.py>`_.

An illustration of how to call this script and use it
with the :class:`FileBasedQuantityComputer` is described in the following.
We need to define an :class:`OutputParser` class that can go from the generate output file(s)
to quantities. For our example, we could define such a parser like so:

.. code-block:: python

    class MyOutputParser:
        def __call__(self, output_files: list[Path]) -> dict[str, Any]:
            """Parse the output files and retrieve the quantities."""
            res = {}
            for f in output_files:
                data = np.loadtxt(f)
                res = {"y": data[:, 0], "x": data[:, 1]}

            return res

Optionally, it is possible to define an optional :class:`PreSubmitHook` class that uses
the dictionary of parameters to perform some task before actually submitting the command to create the output files.
In this example, we don't need one.

Therefore, the entire sequence of logic (from parameters to quantities in a `FileBasedQuantityComputer`) would follow these steps:

1. Optionally do something with parameters (via the :class:`PreSubmitHook` class)
2. Run a command to generate output file(s), which in this example, would correspond to running the aforementioned script generating output data.
3. Parse the output file(s) using a user-defined :class:`OutputParser` to obtain quantities.

We would also require a loss function to build the objective function, for instance:

.. code-block:: python

    def loss_function(quantities: dict[str, Any], ref_y: Iterable[float]) -> float:
        y_values = quantities["y"]
        errors = [(y - y_r) ** 2 for y, y_r in zip(y_values, ref_y)]
        return np.sum(errors)

And the code that would use all of this would look something like this:

.. code-block:: python

    from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
    from chemfit.file_based_computer import FileBasedQuantityComputer
    from chemfit.fitter import Fitter

    ref_file = "input/ref_data.txt"
    data = np.loadtxt(ref_file)
    ref_quantities = {"y": data[:, 0], "x": data[:, 1]}
    # Output file created by the FileBasedQuantityComputer
    output_file = test_dir / Path("output/output_square_function.txt")
    # Script that creates the output file
    script_file = test_dir / Path("input/square_function.py")

    # Initial guess for the prefactor (the one parameter we will change)
    initial_guess = {"prefactor": 0.01}

    # Define the command that will be called to create the output file with given parameters
    def callable_cmd(
        parameters: dict[str, float], script_file: Path, output_file: Path
    ) -> str:
        return f"python {script_file} {parameters['prefactor']} {output_file}"

    output_parser = MyOutputParser()
    file_based_computer = FileBasedQuantityComputer(
        [output_file],
        lambda p: callable_cmd(p, script_file, output_file),
        output_parser,
        poll_interval=0.5,
    )

    ob_func = QuantityComputerObjectiveFunction(
        loss_function=lambda q: loss_function(q, ref_y=ref_quantities["y"]),
        quantity_computer=file_based_computer,
    )

    fitter = Fitter(ob_func, initial_params=initial_guess)

    opt_params = fitter.fit_scipy()

The entire example can be found in the tests.
