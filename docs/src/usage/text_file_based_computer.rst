.. _file_based:

===============================
File-Based Quantity Computers
===============================

The :py:class:`~chemfit.file_based_computer.FileBasedQuantityComputer`
runs an external command in a temporary working directory and parses the
resulting output files into a quantity dictionary.

This is the standard way to integrate external simulation codes into ChemFit.

A file-based computer is constructed from three pieces:

- a function that builds the command
- a list of expected output files
- a function that parses those files

Minimal example
-----------------

Consider an external script with a command-line interface, which does the following:

1. Accept an input :math:`A`
2. Compute :math:`y_i = A (x_i-2)^2` for a predefined range of :math:`x_i \in \left[ x_\text{min}, x_\text{max} \right]`
3. Write the resulting arrays :math:`y_i` and the corresponding :math:`x_i` to a file

.. note::

    The full script can be found in the unit tests at `<https://github.com/MSallermann/chemfit/tests/input/square_function.py>`_.

In this example we will use the :py:class:`~chemfit.file_based_computer.FileBasedQuantityComputer` to determine the
pre-factor :math:`A`.

Before we can start we should define how our external command can be called.
For maximum flexibility, the command is provided as a function that accepts
the parameter dictionary and the temporary working directory. Each evaluation
runs in its own isolated working directory.

All files created by the external command should be written relative to this
working directory. The paths specified in ``output_files`` are interpreted
relative to it as well.

.. note::

    The extra arguments, ``script_file`` and ``output_file``, need to be bound. In the end the computer will accept only a function
    whose only free arguments are the parameters and the working directory. In this example we will use the :py:meth:`~chemfit.file_based_computer.FileBasedQuantityComputer.with_cmd`
    utility method to help us out with this.

.. code-block:: python

    # Define the command that will be called to create the output file with given parameters
    def callable_cmd(
        parameters: dict[str, float], workdir: Path, script_file: Path, output_file: Path
    ) -> list[str]:
        return f"python {script_file} {parameters['prefactor']} {output_file}".split()

Next, we need to define a parser that converts the generated output file(s)
into quantities.

For our example, we could define such a parser like so:

.. code-block:: python

    import numpy as np

    def my_output_parser(output_files: list[Path]) -> dict[str, Any]:
        """Parse the output files and retrieve the quantities."""
        f = output_files[0]
        data = np.loadtxt(f)
        return {"y": data[:, 0], "x": data[:, 1]}

.. note::

    As you can see :py:func:`my_output_parser` *has* to accept a list of output files.
    In this simple example, we do not have to worry about this, since we know there will only ever be one output file.

    The reason for the list is that the :py:class:`~chemfit.file_based_computer.FileBasedQuantityComputer` may specify multiple output files and, in fact, multiple parsers.
    All output files are passed to all parsers, and their outputs are merged.

We will also need the following loss function

.. code-block:: python

    def loss_function(quantities: dict[str, Any], ref_y: Iterable[float]) -> float:
        y_values = quantities["y"]
        errors = [(y - y_r) ** 2 for y, y_r in zip(y_values, ref_y)]
        return np.sum(errors)

Now we're ready to wire everything up:

.. code-block:: python

    ob = (
        FileBasedQuantityComputer(
            output_files=["output.txt"],
            output_parsers=[my_output_parser],
            base_working_directory=".",
            delete_temp_workdirs=True,
        )
        .with_cmd(callable_cmd, script_file=script_file, output_file="output.txt")
        .with_loss(loss_function, ref_y=ref_quantities["y"])
    )

    initial_guess = {"prefactor": 0.01}
    fitter = Fitter(ob, initial_params=initial_guess)
    opt_params = fitter.fit_scipy()

The entire example can be found in the tests.

What happens during evaluation
------------------------------

Each evaluation runs in an isolated working directory.

A single call performs the following steps:

1. create a temporary working directory
2. run ``presubmit_hook`` (if provided)
3. build the command via ``executable_cmd``
4. execute it using :py:func:`subprocess.run`
5. wait until all expected output files exist
6. parse them using ``output_parsers``
7. return the resulting quantity dictionary

The working directory is removed after evaluation unless configured otherwise.


Customization points
--------------------

The behavior is controlled entirely through callables.


Command construction
^^^^^^^^^^^^^^^^^^^^

``executable_cmd`` receives the parameter dictionary and the current workdir and must
return a command (list of strings):

.. code-block:: python

   def executable_cmd(parameters : dict[str,Any], workdir : Path):
       return ["my_program", "--x", str(parameters["x"])]

This function is called for every evaluation.


Output files
^^^^^^^^^^^^

``output_files`` defines which files must exist before parsing begins.

.. code-block:: python

   output_files = [Path("energy.txt"), Path("forces.txt")]

All paths must be **relative to the working directory**.


Output parsing
^^^^^^^^^^^^^^

``output_parsers`` receives the list of output file paths and returns a
dictionary of quantities:

.. code-block:: python

   def output_parsers(paths):
       energy = float(paths[0].read_text())
       return {"energy": energy}


Presubmit hook
--------------

If input files need to be written before execution, use ``presubmit_hook``:

.. code-block:: python

   def presubmit_hook(parameters:dict[str,Any], workdir:Path):
       with open("input.txt", "w") as f:
           f.write(str(parameters["x"]))

This runs inside the working directory before the command is executed.


Important rules
---------------

Output files must be relative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All paths in ``output_files`` must be relative to the working directory.

Using absolute paths breaks isolation and can lead to incorrect results
when running in parallel.


Existence vs completeness
^^^^^^^^^^^^^^^^^^^^^^^^^

An output file is considered ready as soon as it exists.

The framework does not check whether the file is fully written.

If your program writes files incrementally, ensure that files only appear
once complete, or use a separate completion flag file.


Scheduler caveat
^^^^^^^^^^^^^^^^

Some commands (e.g. ``srun`` or ``sbatch``) return before the computation
has finished.

In that case, output files may appear before the job is done.

A common solution is to write a ``done`` file and include it in
``output_files``.


Debugging and failure handling
------------------------------

Temporary working directories are deleted after successful execution.

For debugging, you can keep them:

.. code-block:: python

   computer = FileBasedQuantityComputer(
       ...,
       keep_temp_workdir_after_crash=True,
   )

To inspect failures, you can also enable dump files:

.. code-block:: python

   computer = FileBasedQuantityComputer(
       ...,
       write_dump_file_after_crash=True,
   )

During execution, useful information is stored in the context, including:

- the working directory
- the executed command
- the output files


Execution options
-----------------

The constructor exposes additional options:

- ``base_working_directory`` - where temporary directories are created
- ``wait_timeout`` - maximum time to wait for output files
- ``poll_interval`` - how often file existence is checked
- ``subprocess_run_args`` - arguments passed to ``subprocess.run``
- ``delete_temp_workdirs`` - whether to remove directories after success


Subclassing
-----------

In most cases, constructing a
:py:class:`~chemfit.file_based_computer.FileBasedQuantityComputer`
with callables is sufficient.

Subclassing is useful when the execution flow itself needs to change.

A typical example is adding a scheduler wrapper such as ``srun``:

.. code-block:: python

   class SrunComputer(FileBasedQuantityComputer):
       def build_cmd(self, parameters, ctx):
           base_cmd = super().build_cmd(parameters, ctx)
           return ["srun", *base_cmd]

This pattern is used when command construction depends on runtime context.


Summary
-------

:py:class:`~chemfit.file_based_computer.FileBasedQuantityComputer`
provides a structured way to:

- run external programs
- isolate executions in temporary directories
- collect results as dictionaries

It is the main integration point for external simulation workflows in ChemFit.
