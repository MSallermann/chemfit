==================================
Asynchronous Objective Evaluation
==================================

The :py:class:`~chemfit.async_wrapper_cob.AsyncWrapperCOB` class enables
**concurrent evaluation** of a
:py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
using Python's :mod:`asyncio` framework.

It is particularly useful when objective terms internally perform **blocking
operations**, for example:

- :func:`subprocess.run` calls,
- text-file-based simulations,
- external command-line tools (e.g., LAMMPS, GROMACS),
- other I/O-bound or external-code workloads.

This makes it an excellent companion to
:class:`~chemfit.file_based_computer.FileBasedQuantityComputer` and related
file-based workflows, where each term may trigger an external simulation.

Basic concept
-------------

A :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
combines multiple sub-objectives into a single scalar loss. Evaluating all
terms serially can be slow if each term runs a separate simulation or
command-line tool.

``AsyncWrapperCOB`` addresses this by:

- wrapping a ``CombinedObjectiveFunction``,
- scheduling **one async task per term**,
- running the blocking work for each term in a thread pool,
- gathering the results concurrently,
- summing all contributions into a single scalar loss.

Because it uses threads, no pickling is required, and it works well with
file-based quantity computers and external simulators.

Example
-------

Basic usage with a combined objective:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor
    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit.async_wrapper_cob import AsyncWrapperCOB
    from chemfit.fitter import Fitter

    # Suppose each term internally calls subprocess.run via file-based computers
    combined = CombinedObjectiveFunction(objective_functions=terms)

    # Optional: restrict concurrency via a custom thread pool
    executor = ThreadPoolExecutor(max_workers=4)

    # Wrap for async parallel evaluation
    wrapper = AsyncWrapperCOB(combined, executor=executor)

    fitter = Fitter(wrapper, initial_params={"x": 0.0})
    opt_params = fitter.fit_scipy()

    print(opt_params)

Integration with file-based quantity computers and external tools
-----------------------------------------------------------------

A possible pattern in ChemFit is to use
:class:`~chemfit.file_based_computer.FileBasedQuantityComputer` to drive an
external simulation code (such as LAMMPS) via text-based input and output
files:

1. Parameters are mapped to an input file (e.g., a LAMMPS input script).
2. A command is executed (often using :func:`subprocess.run`) to run the
   simulation.
3. Output files are parsed into quantities through a user-defined parser.
4. A loss function converts these quantities into a scalar objective value.

When many such file-based objectives are combined into a single
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`,
``AsyncWrapperCOB`` allows these external simulations to be launched **in
parallel** within a single Python process.

Sketch of usage with a LAMMPS-based workflow:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor
    from chemfit.file_based_computer import FileBasedQuantityComputer
    from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit.async_wrapper_cob import AsyncWrapperCOB
    from chemfit.fitter import Fitter

    # Build several file-based computers, e.g. each running a different LAMMPS setup
    energy_computer = FileBasedQuantityComputer(...)
    density_computer = FileBasedQuantityComputer(...)
    rdf_computer = FileBasedQuantityComputer(...)

    energy_ob = QuantityComputerObjectiveFunction(
        loss_function=energy_loss_fn,
        quantity_computer=energy_computer,
    )

    density_ob = QuantityComputerObjectiveFunction(
        loss_function=density_loss_fn,
        quantity_computer=density_computer,
    )

    rdf_ob = QuantityComputerObjectiveFunction(
        loss_function=rdf_loss_fn,
        quantity_computer=rdf_computer,
    )

    combined = CombinedObjectiveFunction(
        objective_functions=[energy_ob, density_ob, rdf_ob],
        weights=[1.0, 0.5, 0.2],
    )

    # Use asyncio-based parallelism to run the LAMMPS-based terms concurrently
    executor = ThreadPoolExecutor(max_workers=4)
    async_wrapper = AsyncWrapperCOB(combined, executor=executor)

    fitter = Fitter(async_wrapper, initial_params=initial_params)
    opt_params = fitter.fit_scipy()

    executor.shutdown(wait=True)

In this setting, each objective term may launch an independent LAMMPS run via
a file-based quantity computer. ``AsyncWrapperCOB`` overlaps these runs in
time, reducing total wall-clock time while keeping the API compatible with
:class:`~chemfit.fitter.Fitter`.

Comparison with MPI-based wrapper
---------------------------------

``AsyncWrapperCOB``:

- Uses threads on a **single machine**.
- Requires no MPI installation.
- Works naturally with file-based quantity computers and external tools like
  LAMMPS.
- Ideal when you have a moderate number of independent, blocking terms.

:py:class:`~chemfit.mpi_wrapper_cob.MPIWrapperCOB`:

- Uses MPI across **multiple processes or nodes**.
- Best suited for heavy simulations in distributed environments.
- Requires MPI and :mod:`mpi4py`.

Choose the async wrapper when:

- You are running on a single workstation or node.
- Your objective terms are independent and blocking (e.g. LAMMPS runs).
- You prefer a simpler, MPI-free parallelization layer.

Common pitfalls
---------------

- Calling ``AsyncWrapperCOB.__call__`` from within an already running event
  loop (e.g., some notebook or async frameworks) is not supported; in such
  contexts, use :meth:`~chemfit.async_wrapper_cob.AsyncWrapperCOB.async_call`
  directly and ``await`` it.
- Launching too many concurrent simulations may overload a single machine;
  use a custom :class:`~concurrent.futures.ThreadPoolExecutor` with a suitable
  ``max_workers`` limit.
- Ensure that file-based workflows and external tools (e.g., LAMMPS input/output
  directories) are configured to avoid conflicting file names when running
  multiple simulations in parallel.

Summary
-------

- :py:class:`~chemfit.async_wrapper_cob.AsyncWrapperCOB` provides async/threaded
  parallelism for objective-term evaluation.
- It complements file-based quantity computers and is well suited to external
  tools like LAMMPS.
- It is a drop-in wrapper around
  :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.
- It offers a simple, MPI-free way to reduce wall-clock time when fitting
  against multiple, independent external simulations.
