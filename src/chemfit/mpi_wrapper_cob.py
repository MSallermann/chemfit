from mpi4py import MPI
from typing import Optional, Any
import math
import logging
from numbers import Real

logger = logging.getLogger(__name__)


def slice_up_range(N: int, n_ranks: int):
    chunk_size = math.ceil(N / n_ranks)

    for rank in range(n_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, N)
        yield (start, end)


class MPIWrapperCOB:
    def __init__(self, cob: Any, comm: Optional[Any] = None, finalize_mpi: bool = True):
        self.cob = cob
        if comm is None:
            self.comm = MPI.COMM_WORLD.Dup()
        else:
            self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.finalize_mpi = finalize_mpi

    def print(self, msg: str):
        print(f"[Rank {self.rank}] {msg}")

    def __enter__(self):
        # Attach comm info to the object for call_mpi
        self.cob.comm = self.comm
        self.cob.rank = self.rank
        self.cob.size = self.size

        start, end = list(slice_up_range(self.cob.n_terms(), self.size))[self.rank]

        if self.size > 1 and self.rank != 0:
            # Worker loop: wait for params, compute slice+reduce, repeat
            while True:
                params = self.comm.bcast(None, root=0)

                if params is None:
                    break

                try:
                    local_total = self.cob(params, idx_slice=slice(start, end))
                except Exception as e:
                    logging.debug(
                        f"Caught exception while evaluating ({start},{end}). Returning Nan."
                    )

                    local_total = float("NaN")
                finally:
                    if not isinstance(local_total, Real):
                        logging.debug(
                            f"Index ({start},{end}) did not return a number. It returned `{local_total}` of type {type(local_total)}."
                        )
                        local_total = float("NaN")

                    # Sum up all local_totals into a global_total on the master rank
                    _ = self.comm.reduce(local_total, op=MPI.SUM, root=0)

        return self

    def __call__(self, params: dict) -> float:
        # Function to evaluate the objective function, to be called from rank 0

        self.comm.bcast(params, root=0)
        start, end = list(slice_up_range(self.cob.n_terms(), self.size))[self.rank]

        try:
            local_total = self.cob(params, idx_slice=slice(start, end))
        except Exception as e:
            local_total = float("NaN")
            raise e
        finally:
            # Sum up all local_totals into a global_total on every rank
            global_total = self.comm.reduce(local_total, op=MPI.SUM, root=0)

        return global_total

    def __exit__(self, exc_type, exc, tb):

        # Only rank 0 needs to shut down workers
        if self.rank == 0 and self.size > 1:
            # send the poison‐pill (None) so workers break out
            self.comm.bcast(None, root=0)

        # ensure everyone leaves together
        self.comm.Barrier()

        if self.finalize_mpi:
            MPI.Finalize()
