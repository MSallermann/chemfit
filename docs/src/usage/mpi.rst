.. _mpi:

##################
Running with MPI
##################

.. warning::

    This section is a work in progress!

.. note::

    To run ``ChemFit`` with MPI, you need a working MPI installation -- duh! If you're on a cluster, consult its documentation. You can of course also install MPI on your local machine.

    Further, you need to have ``mpi4py``, which is an optional dependency installed with ``pip install chemfit[mpi]``.

The gist of it is that you can use MPI to parallelize the evaluation of the individual terms in a :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.

In principle the fitting code looks not terribly different from the single threaded case.

Within the python script, the required steps are:

1. **All ranks** construct the :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
2. **All ranks** enter the context provided by :py:class:`~chemfit.mpi_wrapper_cob.MPIWrapperCOB`
3. **The main rank** calls the fitting routines, **all other ranks** enter the :py:class:`~chemfit.mpi_wrapper_cob.MPIWrapperCOB.worker_loop`

.. note::

    Thanks to the **lazy-loading** mechanism, constructing the objective function on all ranks is not actually wasteful.
    Since only the ranks which actually compute a certain term of the objective function actually construct the ``Atoms`` and the ``Calculator``.

**Then:**
Call the script with ``mpirun -n $NRANKS python script.py`` (or something equivalent)

A more schematic example:

.. code-block:: python

    from chemfit.mpi_wrapper_cob import MPIWrapperCOB
    from chemfit import CombinedObjectiveFunction

    # ...
    # construct the combined objective function on **all ranks**
    # ...

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    with MPIWrapperCOB(ob) as ob_mpi:
        # The optimization needs to run on the first rank only
        if ob_mpi.rank == 0:
            fitter = Fitter(ob_mpi, initial_params=initial_params, bounds=bounds)
            opt_params = fitter.fit_nevergrad(budget=100)

            #...
            meta_data_list = ob_mpi.gather_meta_data()
            # write output etc.
            #...
        else:
            ob_mpi.worker_loop()

.. tip::

    :py:meth:`~chemfit.mpi_wrapper_cob.MPIWrapperCOB.gather_meta_data` gathers the meta_data from all worker ranks on the main rank

Concrete Example:
********************

This is the same Lennard Jones example as in the :ref:`quickstart`, but this time with MPI. Yay!

.. code-block:: python

    from chemfit.multi_energy_objective_function import create_multi_energy_objective_function
    from chemfit.fitter import Fitter
    from chemfit.mpi_wrapper_cob import MPIWrapperCOB
    from ase.calculators.lj import LennardJones

    def e_lj(r, eps, sigma):
        return 4.0 * eps * ((sigma / r) ** 6 - 1.0) * (sigma / r) ** 6

    class LJAtomsFactory:
        def __init__(self, r: float):
            p0 = np.zeros(3)
            p1 = np.array([r, 0.0, 0.0])
            self.atoms = Atoms(positions=[p0, p1])

        def __call__(self):
            return self.atoms

    def construct_lj(atoms: Atoms):
        atoms.calc = LennardJones(rc=2000)

    def apply_params_lj(atoms: Atoms, params: dict[str, float]):
        atoms.calc.parameters.sigma = params["sigma"]
        atoms.calc.parameters.epsilon = params["epsilon"]

    ### Construct the objective function on *all* ranks
    eps = 1.0
    sigma = 1.0

    r_min = 2 ** (1/6) * sigma
    r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

    ob = create_multi_energy_objective_function(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        tag_list=[f"lj_{r:.2f}" for r in r_list],
        reference_energy_list=[e_lj(r, eps, sigma) for r in r_list],
        path_or_factory_list=[LJAtomsFactory(r) for r in r_list],
    )

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    with MPIWrapperCOB(ob) as ob_mpi:
        # The optimization needs to run on the first rank only
        if ob_mpi.rank == 0:
            initial_params = {"epsilon": 2.0, "sigma": 1.5}
            bounds = {"epsilon": (0.1, 10), "sigma": (0.5, 3.0)}
            fitter = Fitter(ob_mpi, initial_params=initial_params, bounds=bounds)

            opt_params = fitter.fit_scipy()

            assert np.isclose(opt_params["epsilon"], eps)
            assert np.isclose(opt_params["sigma"], sigma)
        else:
            ob_mpi.worker_loop()
