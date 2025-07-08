try:
    import mpi4py

    HAS_MPI = True
except ImportError:
    HAS_MPI = False
