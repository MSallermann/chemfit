from __future__ import annotations

import logging
import math
from enum import Enum
from numbers import Real
from typing import Any

from mpi4py import MPI

from chemfit.abstract_objective_function import ObjectiveFunctor
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.debug_utils import log_all_methods
from chemfit.exceptions import FactoryException

logger = logging.getLogger(__name__)


def slice_up_range(n: int, n_ranks: int):
    chunk_size = math.ceil(n / n_ranks)

    for rank in range(n_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, n)
        yield (start, end)


class Signal(Enum):
    ABORT = -1
    GATHER_META_DATA = 0


class MPIWrapperCOB(ObjectiveFunctor):
    def __init__(
        self,
        cob: CombinedObjectiveFunction,
        comm: Any | None = None,
        mpi_debug_log: bool = False,
    ) -> None:
        """Initialize wrapper for combined objective function."""

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

        self.start, self.end = list(slice_up_range(self.cob.n_terms(), self.size))[
            self.rank
        ]

    def __enter__(self):
        return self

    def worker_process_params(self, params: dict[str, Any]):
        # In the usual use-case the worker loop will be the top-level context for the worker ranks.
        # Therefore, the error handling needs to be slightly different,
        # and we try to suppress general exceptions instead of re-raising them
        # We do not suppress `FactoryException`s, however, because, it makes no sense to continue execution.
        # The reason that it makes no sense is that these  exceptions are connected to being unable to construct internals
        # of the objective functions.
        # (remember due to lazy evaluation such constructions can happen inside `__call__`)
        local_total = float("Nan")
        try:
            # First we try to obtain a value the normal way
            local_total = self.cob(params, idx_slice=slice(self.start, self.end))

            # if we don't get a real number, we convert it to a NaN
            if not isinstance(local_total, Real):
                logger.debug(
                    f"Index ({self.start},{self.end}) did not return a number. It returned `{local_total}` of type {type(local_total)}."
                )
                local_total = float("NaN")
        except FactoryException as e:
            # If we catch a factory exception we should just crash the code
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
        finally:
            # Finally, we have to run the reduce. This must always happen since, otherwise, we might cause deadlocks
            # Sum up all local_totals into a global_total on the master rank
            _ = self.comm.reduce(local_total, op=MPI.SUM, root=0)

    def worker_gather_meta_data(self):
        local_meta_data = self.cob.gather_meta_data(
            idx_slice=slice(self.start, self.end)
        )
        self.comm.gather(local_meta_data, root=0)

    def worker_loop(self):
        if self.size > 1 and self.rank != 0:
            # Worker loop: wait for params, compute slice+reduce, repeat
            while True:
                signal = self.comm.bcast(None, root=0)

                if signal == Signal.ABORT:
                    break
                if signal == Signal.GATHER_META_DATA:
                    self.worker_gather_meta_data()
                elif isinstance(signal, dict):
                    params: dict[str, Any] = signal
                    self.worker_process_params(params)

    def gather_meta_data(self) -> list[dict[str, Any] | None]:
        # Ensure only rank 0 can call this
        if self.rank != 0:
            msg = "`gather_meta_data` can only be used on rank 0"
            raise RuntimeError(msg)

        self.comm.bcast(Signal.GATHER_META_DATA, root=0)

        local_meta_data = self.cob.gather_meta_data(
            idx_slice=slice(self.start, self.end)
        )
        gathered = self.comm.gather(local_meta_data)

        # Since gathered will now be a list of list, we unpack it
        total_meta_data: list[dict[str, Any] | None] = []

        if gathered is not None:
            [total_meta_data.extend(m) for m in gathered]

        return total_meta_data

    def get_meta_data(self) -> dict[str, Any]:
        d = self.cob.get_meta_data()
        d["type"] = type(self).__name__
        return d

    def __call__(self, params: dict[str, Any]) -> float:
        # Function to evaluate the objective function, to be called from rank 0

        # Ensure only rank 0 can call this
        if self.rank != 0:
            msg = "`__call__` can only be used on rank 0"
            raise RuntimeError(msg)

        self.comm.bcast(params, root=0)

        local_total = float("NaN")
        try:
            local_total = self.cob(params, idx_slice=slice(self.start, self.end))
        except FactoryException as e:
            # If we catch a factory exception we should just crash the code
            raise e
        except Exception as e:
            # If an exception occurs on the master rank, we set the local total to "NaN"
            # (so that the later reduce works fine and gives NaN)
            # then we simply re-raise and the exception can be handled higher up (or not)
            raise e
        finally:
            # Finally, we have to run the reduce. This must always happen since, otherwise, we might cause deadlocks
            # Sum up all local_totals into a global_total on every rank
            global_total = self.comm.reduce(local_total, op=MPI.SUM, root=0)
            if global_total is None:
                global_total = float("NaN")

        return global_total

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ):
        # Only rank 0 needs to shut down workers
        if self.rank == 0 and self.size > 1:
            # send the poison-pill (None) so workers break out
            self.comm.bcast(Signal.ABORT, root=0)
