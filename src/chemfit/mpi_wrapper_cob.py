from mpi4py import MPI
from typing import Optional, Any
import math
import logging
from numbers import Real
from chemfit.exceptions import FactoryException
from chemfit.debug_utils import log_all_methods


logger = logging.getLogger(__name__)


def slice_up_range(N: int, n_ranks: int):
    chunk_size = math.ceil(N / n_ranks)

    for rank in range(n_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, N)
        yield (start, end)


class MPIWrapperCOB:
    def __init__(
        self, cob: Any, comm: Optional[Any] = None, mpi_debug_log: bool = False
    ):
        self.cob = cob
        if comm is None:
            self.comm = MPI.COMM_WORLD.Dup()
        else:
            self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if mpi_debug_log:
            self.comm = log_all_methods(
                self.comm, lambda msg: logger.warning(f"[Rank {self.rank}] {msg}")
            )

    def __enter__(self):
        return self

    def worker_loop(self):
        if self.size > 1 and self.rank != 0:
            start, end = list(slice_up_range(self.cob.n_terms(), self.size))[self.rank]

            # Worker loop: wait for params, compute slice+reduce, repeat
            while True:

                params = self.comm.bcast(None, root=0)

                if params is None:
                    break

                # In the usual use-case the worker loop will be the top-level context for the worker ranks.
                # Therefore, the error handling needs to be slightly different,
                # and we try to suppress general exceptions instead of re-raising them
                # We do not suppress `FactoryException`s, however, because, it makes no sense to continue execution.
                # The reason that it makes no sense is that these  exceptions are connected to being unable to construct internals
                # of the objective functions.
                # (remember due to lazy evaluation such constructions can happen inside `__call__`)
                try:
                    # First we try to obtain a value the normal way
                    local_total = self.cob(params, idx_slice=slice(start, end))

                    # if we don't get a real number, we convert it to a NaN
                    if not isinstance(local_total, Real):
                        logger.debug(
                            f"Index ({start},{end}) did not return a number. It returned `{local_total}` of type {type(local_total)}."
                        )
                        local_total = float("NaN")
                except FactoryException as e:
                    # If we catch a factory exception we should just crash the code
                    local_total = float("NaN")
                    logger.exception(e, stack_info=True, stacklevel=2)
                    raise e  # <-- from here we enter the __exit__ method, the worker rank will crash and consequently all processes are stopped
                except Exception as e:
                    # We assume all other exceptions stem from going into bad parameter regions
                    # In such a case we dont propagate the exception, but instead set the local_total to "Nan"
                    # We only log this at the debug level otherwise we might create *huge* log files when the objective function is called in a loop
                    logger.debug(
                        e,
                        stack_info=True,
                        stacklevel=2,
                    )
                    local_total = float("NaN")
                finally:
                    # Finally, we have to run the reduce. This must always happen since, otherwise, we might cause deadlocks
                    # Sum up all local_totals into a global_total on the master rank
                    _ = self.comm.reduce(local_total, op=MPI.SUM, root=0)

    def __call__(self, params: dict) -> float:
        # Function to evaluate the objective function, to be called from rank 0

        # Ensure only rank 0 can call this
        if self.rank != 0:
            raise RuntimeError("__call__ can only be used on rank 0")

        self.comm.bcast(params, root=0)

        start, end = list(slice_up_range(self.cob.n_terms(), self.size))[self.rank]

        try:
            local_total = self.cob(params, idx_slice=slice(start, end))
        except Exception as e:
            # If an exception occurs on the master rank, we set the local total to "NaN"
            # (so that the later reduce works fine and gives NaN)
            # then we simply re-raise and the exception can be handled higher up (or not)
            local_total = float("NaN")
            raise e
        finally:
            # Finally, we have to run the reduce. This must always happen since, otherwise, we might cause deadlocks
            # Sum up all local_totals into a global_total on every rank
            global_total = self.comm.reduce(local_total, op=MPI.SUM, root=0)
        return global_total

    def __exit__(self, exc_type, exc, tb):

        # Only rank 0 needs to shut down workers
        if self.rank == 0 and self.size > 1:
            # send the poison‐pill (None) so workers break out
            self.comm.bcast(None, root=0)
