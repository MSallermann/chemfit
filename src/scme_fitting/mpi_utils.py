from mpi4py import MPI
from typing import Optional, Any
import math


def slice_up_range(N: int, n_ranks: int):
    chunk_size = math.ceil(N / n_ranks)

    for rank in range(n_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, N)
        yield (start, end)


class MPIContext:
    def __init__(self, cob: Any, comm: Optional[Any] = None):
        self.cob = cob
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def __enter__(self):
        # Attach comm info to the object for call_mpi
        self.cob.comm = self.comm
        self.cob.rank = self.rank
        self.cob.size = self.size

        if self.size > 1 and self.rank != 0:
            # Worker loop: wait for params, compute slice+reduce, repeat
            while True:
                params = self.comm.bcast(None, root=0)

                if params is None:
                    break

                start, end = list(slice_up_range(self.cob.n_terms(), self.size))[
                    self.rank
                ]
                local_total = self.cob(params, idx_slice=slice(start, end))
                # Sum up all local_totals into a global_total on the master rank
                _ = self.comm.reduce(local_total, op=MPI.SUM, root=0)

        # Rank 0: return the MPI‐aware evaluate function
        def mpi_evaluate(params: dict):
            self.comm.bcast(params, root=0)
            start, end = list(slice_up_range(self.cob.n_terms(), self.size))[self.rank]
            local_total = self.cob(params, idx_slice=slice(start, end))
            # Sum up all local_totals into a global_total on every rank
            global_total = self.comm.reduce(local_total, op=MPI.SUM, root=0)
            return global_total

        return mpi_evaluate

    def __exit__(self, exc_type, exc, tb):
        # Only rank 0 needs to shut down workers
        if self.rank == 0 and self.size > 1:
            # send the poison‐pill (None) so workers break out
            self.comm.bcast(None, root=0)

        # ensure everyone leaves together
        self.comm.Barrier()
