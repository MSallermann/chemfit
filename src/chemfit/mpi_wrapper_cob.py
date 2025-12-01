from __future__ import annotations

import logging
import math
from enum import Enum
from numbers import Real
from typing import Any

from mpi4py import MPI

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.debug_utils import log_all_methods

logger = logging.getLogger(__name__)


def slice_up_range(n: int, n_ranks: int):
    chunk_size = math.ceil(n / n_ranks)

    for rank in range(n_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, n)
        yield (start, end)


class Signal(Enum):
    ABORT = -1


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

    def worker_process_params(self, params: dict[str, Any], ctx: EvaluateContext):
        # In the usual use-case the worker loop will be the top-level context for the worker ranks.
        # Therefore, the error handling is slightly different than on rank 0 and we log the exception here before re-raising
        local_total = float("Nan")
        try:
            # First we try to obtain a value the normal way
            local_total = self.cob(
                params, ctx=ctx, idx_slice=slice(self.start, self.end)
            )

            # if we don't get a real number, we convert it to a NaN
            if not isinstance(local_total, Real):
                logger.debug(
                    f"Index ({self.start},{self.end}) did not return a number. It returned `{local_total}` of type {type(local_total)}."
                )
                local_total = float("NaN")
        except Exception as e:
            # If we catch an exception we should just crash the code
            logger.exception(e, stack_info=True, stacklevel=2)
            raise e  # <-- from here we enter the __exit__ method, the worker rank will crash and consequently all processes are stopped
        finally:
            # Finally, we have to run the reduce.
            # This must always happen, otherwise, we might cause deadlocks because other ranks might wait on a reduce.
            # Sum up all local_totals into a global_total on the master rank
            _ = self.comm.reduce(local_total, op=MPI.SUM, root=0)

    def worker_gather_meta_data(self, ctx: EvaluateContext):
        # The local meta data should already be in the context
        assert "children" in ctx.meta
        local_meta_data = ctx.meta["children"]
        self.comm.gather(local_meta_data, root=0)

    def worker_loop(self):
        # Ensure only rank 0 can call this
        if self.rank == 0:
            msg = "`worker_loop` cannot be used on rank 0"
            raise RuntimeError(msg)

        # Worker loop: wait for params, compute slice+reduce, repeat
        while True:
            # receive a signal from rank 0
            signal = self.comm.bcast(None, root=0)

            if signal == Signal.ABORT:
                break

            if isinstance(signal, dict):
                params: dict[str, Any] = signal
                ctx = EvaluateContext()
                self.worker_process_params(params, ctx)
                self.worker_gather_meta_data(ctx)

    def gather_meta_data(self, ctx: EvaluateContext):
        # Ensure only rank 0 can call this
        if self.rank != 0:
            msg = "`gather_meta_data` can only be used on rank 0"
            raise RuntimeError(msg)

        # The local meta data should already be in the context
        assert "children" in ctx.meta
        local_meta_data = ctx.meta["children"]

        # Broadcast the signal
        gathered = self.comm.gather(local_meta_data)

        # Since gathered will now be a list of list, we unpack it
        total_meta_data: list[dict[str, Any] | None] = []

        if gathered is not None:
            [total_meta_data.extend(m) for m in gathered]

        # TODO(MS): think about this some more. It is a bit hacky because now the `ctx._children` variable only has the child ctxs for rank 0, but the meta_data from all the other ranks too  # noqa: TD003
        ctx.meta["children"] = total_meta_data

    def __call__(
        self, params: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> float:
        # Function to evaluate the objective function, to be called from rank 0

        if ctx is None:
            ctx = EvaluateContext()

        # Ensure only rank 0 can call this
        if self.rank != 0:
            msg = "`__call__` can only be used on rank 0"
            raise RuntimeError(msg)

        # Broadcast the params to the worker ranks
        self.comm.bcast(params, root=0)

        local_total = float("NaN")  # So we get NaN in case the local compute fails
        try:
            # Compute one slice of the objective function on the main rank
            local_total = self.cob(
                params, ctx=ctx, idx_slice=slice(self.start, self.end)
            )
        finally:
            # Finally, we have to run the reduce. This must always happen since, otherwise, we might cause deadlocks
            # Sum up all local_totals into a global_total on every rank
            global_total = self.comm.reduce(local_total, op=MPI.SUM, root=0)
            if global_total is None:
                global_total = float("NaN")

            self.gather_meta_data(ctx)

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
