.. _external_computer:

===========================
External Quantity Computers
===========================

The :py:class:`~chemfit.external_computer.ExternalQuantityComputer`
runs an external command in a temporary working directory and parses the
resulting output files into a quantity dictionary.

This is the standard way to integrate external simulation codes into ChemFit.

An external computer is constructed from three pieces:

- a function that builds the command
- one or more parser functions
- the output file or files consumed by each parser

Minimal example
-----------------

Consider an external script with a command-line interface, which does the following:

1. Accept an input :math:`A`
2. Compute :math:`y_i = A (x_i-2)^2` for a predefined range of :math:`x_i \in \left[ x_\text{min}, x_\text{max} \right]`
3. Write the resulting arrays :math:`y_i` and the corresponding :math:`x_i` to a file

.. note::

    The full script can be found in the unit tests at `<https://github.com/MSallermann/chemfit/tests/input/square_function.py>`_.

In this example we will use the :py:class:`~chemfit.external_computer.ExternalQuantityComputer` to determine the
pre-factor :math:`A`.

Before we can start we should define how our external command can be called.
For maximum flexibility, the command is provided as a function that accepts
the parameter dictionary and the temporary working directory. Each evaluation
runs in its own isolated working directory.

All files created by the external command should be written relative to this
working directory. Paths registered through ``with_parser()`` or ``wait_for()``
are interpreted relative to it as well.

.. note::

    The extra arguments, ``script_file`` and ``output_file``, need to be bound. In the end the computer will accept only a function
    whose only free arguments are the parameters and the working directory. In this example we will use the :py:meth:`~chemfit.external_computer.ExternalQuantityComputer.with_cmd`
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

    def my_output_parser(output_file: Path) -> dict[str, Any]:
        """Parse the output file and retrieve its quantities."""
        data = np.loadtxt(output_file)
        return {"y": data[:, 0], "x": data[:, 1]}

.. note::

    A parser receives only the files registered with it. For a parser that
    consumes multiple files, pass each filename to ``with_parser()`` in the
    same order as the parser's positional arguments.

We will also need the following loss function

.. code-block:: python

    def loss_function(quantities: dict[str, Any], ref_y: Iterable[float]) -> float:
        y_values = quantities["y"]
        errors = [(y - y_r) ** 2 for y, y_r in zip(y_values, ref_y)]
        return np.sum(errors)

Now we're ready to wire everything up:

.. code-block:: python

    from chemfit.external_computer import ExternalQuantityComputer

    ob = (
        ExternalQuantityComputer(base_working_directory=".")
        .with_cmd(callable_cmd, script_file=script_file, output_file="output.txt")
        .with_parser(my_output_parser, "output.txt")
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
2. execute every hook and command in registration order
3. wait until all parser inputs and completion files exist
4. invoke each registered parser with its resolved input paths
5. return the resulting quantity dictionary

The working directory is removed after evaluation unless configured otherwise.


Customization points
--------------------

The behavior is controlled entirely through callables.


Commands and hooks
^^^^^^^^^^^^^^^^^^

A command callable receives the parameter dictionary and current working
directory and returns a command as a list of strings:

.. code-block:: python

   def run_simulation(parameters: dict[str, Any], workdir: Path):
       return ["my_program", "--x", str(parameters["x"])]

Register commands with ``with_cmd()``. Register ordinary Python setup or
file-processing functions with ``with_hook()``. Calls to these fluent methods
define the exact execution order:

.. code-block:: python

   computer = (
       computer
       .with_hook(write_input, template="input.template")
       .with_cmd(preprocess)
       .with_hook(modify_preprocessed_input)
       .with_cmd(run_simulation)
       .with_hook(postprocess_files)
       .with_cmd(convert_output)
   )

Each step completes before the next begins. Additional keyword arguments passed
to ``with_hook()`` or ``with_cmd()`` are bound to that callable.


Parser inputs and completion files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Register a parser together with the relative paths it consumes:

.. code-block:: python

   computer = computer.with_parser(parse_energy, "energy.txt")
   computer = computer.with_parser(parse_forces, "forces.txt", "stress.txt")

The parser input files are automatically watched before parsing begins. Add a
completion marker that is not consumed by a parser with ``wait_for()``:

.. code-block:: python

   computer = computer.wait_for("task.done")

All registered paths must be **relative to the working directory**.


Output parsing
^^^^^^^^^^^^^^

Each parser receives its resolved output paths as positional arguments and
returns a dictionary of quantities:

.. code-block:: python

   def parse_results(log: Path, forces: Path):
       return {
           "energy": read_energy(log),
           "forces": read_forces(forces),
       }

   computer = computer.with_parser(parse_results, "run.log", "forces.dat")

Parser results are merged in registration order.


Hooks
-----

Hooks are useful whenever Python code must run between external commands. For
example, an input-writing hook can run before the first command:

.. code-block:: python

   def write_input(parameters: dict[str, Any], workdir: Path):
       (workdir / "input.txt").write_text(str(parameters["x"]))

The hook receives the evaluation parameters and temporary working directory.

Example: generating an input file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common use of ``with_hook()`` is to generate input files from a template.

.. code-block:: python

   def write_input(
       parameters: dict[str, Any],
       workdir: Path,
       *,
       template_path: Path,
       output_name: str,
   ):
       template = template_path.read_text()

       content = template.replace("{{A}}", str(parameters["prefactor"]))

       output_path = workdir / output_name
       output_path.write_text(content)

This can then be attached to the computer:

.. code-block:: python

   computer = (
       ExternalQuantityComputer(base_working_directory=".")
       .with_hook(
           write_input,
           template_path=Path("template.in"),
           output_name="input.in",
       )
       .with_cmd(callable_cmd, script_file="square.py", output_file="output.txt")
       .with_parser(my_output_parser, "output.txt")
   )

The hook completes before the following command begins. Hooks can also be
inserted between commands to inspect or modify intermediate files.

.. hint::

    Using a template engine such as ``Jinja`` to generate input files can be
    a very powerful option in an input-writing hook, especially when many
    files need to be configured or share common structure.

Important rules
---------------

Output files must be relative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All paths passed to ``with_parser()`` and ``wait_for()`` must be relative to
the working directory.

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

In that case, the next pipeline step begins after the submission command
returns; ChemFit does not wait for the remote job between steps. Express remote
dependencies through the external scheduler or a wrapper script. Output polling
begins after every registered execution step has run.

A common solution is to write a ``done`` file and register it with
``wait_for("done")``.


Debugging and failure handling
------------------------------

Temporary working directories are deleted after successful execution.

For debugging, you can keep them:

.. code-block:: python

   computer = ExternalQuantityComputer(
       base_working_directory="runs",
       keep_temp_workdir_after_crash=True,
   )

To inspect failures, enable evaluation-level dump files:

.. code-block:: python

   computer = ExternalQuantityComputer(
       base_working_directory="runs",
       write_dump_file_after_crash=True,
   )

A dump records the final exception, current execution step, commands executed
so far, and the other temporary context state. When the final exception is a
:py:class:`subprocess.CalledProcessError`, its command, return code, standard
output, and standard error are included when available. Hook, output-waiting,
and parser failures therefore produce diagnostics as well as command failures.

Parsing output from a failed command
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some external programs return a non-zero exit status after writing usable
output files. Set ``try_parsing_after_exception=True`` to let the computer
continue waiting for its configured output files and run the output parsers
after :py:func:`subprocess.run` raises
:py:class:`subprocess.CalledProcessError`:

.. code-block:: python

   computer = ExternalQuantityComputer(
       base_working_directory="runs",
       try_parsing_after_exception=True,
   )

The default is ``False``, so a non-zero exit status normally fails the
evaluation without parsing. When enabled, the evaluation succeeds only if all
configured output files appear within ``wait_timeout`` and every parser
succeeds. The subprocess failure is logged and recorded in ``ctx.temp``, but a
successful evaluation does not produce a crash dump. If output waiting or
parsing subsequently fails, the single final dump also includes the earlier
command-failure record.

Enable this only when a non-zero exit status is known to leave complete,
trustworthy output. It can otherwise turn a failed calculation into an
apparently successful result based on partial files.

During execution, useful information is stored in the context, including:

- the working directory
- all executed commands, in ``ctx.temp.commands``
- the current execution step during a failure
- a recoverable command failure, when applicable
- the output files


Execution options
-----------------

The constructor exposes additional options:

- ``base_working_directory`` - where temporary directories are created
- ``wait_timeout`` - maximum time to wait for output files
- ``poll_interval`` - how often file existence is checked
- ``subprocess_run_args`` - arguments passed to ``subprocess.run``
- ``delete_temp_workdirs`` - whether to remove directories after success
- ``write_dump_file_after_crash`` - whether to write evaluation diagnostics
- ``keep_temp_workdir_after_crash`` - whether to retain failed work directories
- ``try_parsing_after_exception`` - whether to parse output after a non-zero exit


Command wrappers
----------------

In most cases, constructing a
:py:class:`~chemfit.external_computer.ExternalQuantityComputer`
with callables is sufficient.

Commands can be wrapped without subclassing. For example, a command builder can
add ``srun`` to another command:

.. code-block:: python

   def with_srun(parameters, workdir, *, command):
       return ["srun", *command(parameters, workdir)]

   computer = computer.with_cmd(with_srun, command=run_simulation)


Summary
-------

:py:class:`~chemfit.external_computer.ExternalQuantityComputer`
provides a structured way to:

- run external programs
- isolate executions in temporary directories
- collect results as dictionaries

It is the main integration point for external simulation workflows in ChemFit.
