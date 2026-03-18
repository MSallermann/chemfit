from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Any

from mpi4py import MPI

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.debug_utils import log_all_methods

logger = logging.getLogger(__name__)


def slice_up_range(n: int, n_ranks: int):
    """
    Split a range of length ``n`` into contiguous rank-local chunks.

    The chunks are distributed as evenly as possible across ``n_ranks``
    by using a ceiling-based chunk size.

    Args:
        n: Total number of items to distribute.
        n_ranks: Number of MPI ranks.

    Yields:
        Tuples ``(start, end)`` defining half-open index ranges for
        each rank.

    """

    chunk_size = math.ceil(n / n_ranks)

    for rank in range(n_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, n)
        yield (start, end)


class Signal(Enum):
    ABORT = -1


class MPIWrapperCOB(ObjectiveFunctor):
    """
    MPI-based wrapper for ``CombinedObjectiveFunction``.

    This wrapper distributes the terms of a combined objective across
    MPI ranks. Rank 0 acts as the driver rank: it broadcasts the
    evaluation context to all worker ranks, evaluates its own local
    slice of terms, gathers the worker results, re-raises any worker
    exceptions, collects child metadata, and applies the wrapped
    combined objective's reduction.

    Worker ranks do not call ``__call__`` directly. Instead, they run
    ``worker_loop()``, which waits for broadcast evaluation requests
    from rank 0 and processes the local slice assigned to that rank.
    """

    def __init__(
        self,
        cob: CombinedObjectiveFunction,
        comm: Any | None = None,
        mpi_debug_log: bool = False,
    ) -> None:
        """
        Initialize an MPI wrapper for a combined objective.

        Args:
            cob: Combined objective function whose terms are distributed
                across MPI ranks.
            comm: MPI communicator to use. If ``None``, a duplicate of
                ``MPI.COMM_WORLD`` is created.
            mpi_debug_log: If ``True``, wrap the communicator so that MPI
                method calls are logged for debugging.

        Notes:
            Each rank is assigned a contiguous slice of objective terms at
            initialization time.

        """

        self.cob = cob
        if comm is None:
            self.comm = MPI.COMM_WORLD.Dup()
        else:
            self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if mpi_debug_log:
            self.comm = log_all_methods(
                self.comm,
                log_func=self._log_func,
                log_args=True,
                log_res=True,
            )

        self.start, self.end = list(slice_up_range(self.cob.n_terms(), self.size))[
            self.rank
        ]

    def _log_func(self, msg: str):
        logger.warning(f"[Rank {self.rank}] {msg}")

    def __enter__(self):
        return self

    def shifted_child_context_configurator(
        self,
        idx_child_ctx: int,
        child_ctx: EvaluateContext,
        num_children: int,  # noqa: ARG002
        parent_ctx: EvaluateContext,
    ):
        """
        Invoke the parents child context configurator, while accounting for the slicing.

        The goal of this function is to lead to the same behaviour as on the original combined objective function.
        This means the child_context_configurator has to "see" the idx of the current child not within the current slice, but
        the absolute index.
        Fort the same reason we overwrite the number of children.

        Args:
            idx_child_ctx (int): The index of the current child context *within* the slice
            child_ctx (EvaluateContext): The child context
            num_children (int): The number of children *within* the current slice
            parent_ctx (EvaluateContext): The parent context.

        """

        if self.cob.child_context_configurator is not None:
            self.cob.child_context_configurator(
                idx_child_ctx=idx_child_ctx + self.start,
                child_ctx=child_ctx,
                num_children=self.cob.n_terms(),
                parent_ctx=parent_ctx,
            )

    def evaluate_slice(
        self, params: dict[str, Any], ctx: EvaluateContext
    ) -> list[float | None]:
        idx_slice = slice(self.start, self.end)
        selected_indices = range(self.cob.n_terms())[idx_slice]

        local_terms = []

        with ctx.child_contexts(
            len(selected_indices), configurator=self.shifted_child_context_configurator
        ) as contexts:
            for idx, ctx_term in zip(selected_indices, contexts, strict=True):
                try:
                    local_terms.append(self.cob.evaluate_term(params, idx, ctx_term))
                except Exception as e:  # noqa: PERF203
                    # If we catch an exception we log it and append it to the local terms
                    # It will be sent to master and raised there
                    logger.exception(e)
                    local_terms.append(e)

        return local_terms

    def worker_process_params(self, params: dict[str, Any], ctx: EvaluateContext):
        # In the usual use-case the worker loop will be the top-level context for the worker ranks.

        local_terms = self.evaluate_slice(params, ctx)

        # Finally, we have to run the gather
        # This must always happen, otherwise, we might cause deadlocks because other ranks might wait on a reduce.
        # Sum up all local_totals into a global_total on the master rank
        _ = self.comm.gather(local_terms, root=0)

    def worker_gather_meta_data(self, ctx: EvaluateContext):
        ctx.collect_child_meta_data()
        if "children" in ctx.meta:
            local_meta_data = ctx.meta["children"]
            self.comm.gather(local_meta_data, root=0)
        else:
            self.comm.gather([], root=0)

    def worker_loop(self):
        """
        Run the worker-side MPI evaluation loop.

        This method must be called only on nonzero ranks. Each worker rank
        waits for broadcast messages from rank 0. On receiving an
        ``EvaluateContext``, it evaluates its assigned slice of the
        combined objective and gathers both term values and child metadata
        back to rank 0. On receiving ``Signal.ABORT``, the loop exits.

        Raises:
            RuntimeError: If called on rank 0.

        """

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

            if isinstance(signal, EvaluateContext):
                assert signal.parameters is not None
                params: dict[str, Any] = signal.parameters
                ctx = signal
                self.worker_process_params(params, ctx)
                self.worker_gather_meta_data(ctx)

    def gather_meta_data(self, ctx: EvaluateContext):
        """
        Collect child metadata from all ranks into the parent context.

        This method must be called only on rank 0. It collects the local
        child metadata from rank 0 together with the metadata gathered from
        worker ranks and stores the flattened result in
        ``ctx.meta["children"]``.

        Args:
            ctx: Parent evaluation context on rank 0.

        Raises:
            RuntimeError: If called on a nonzero rank.

        Notes:
            The collected metadata is flattened across ranks. The resulting
            metadata may therefore contain child entries for ranks whose
            original child contexts are not present in ``ctx._children`` on
            rank 0.

        """

        # Ensure only rank 0 can call this
        if self.rank != 0:
            msg = "`gather_meta_data` can only be used on rank 0"
            raise RuntimeError(msg)

        # The local meta data should already be in the context
        assert "children" in ctx.meta
        local_meta_data = ctx.meta["children"]

        # Broadcast the signal
        gathered = self.comm.gather(local_meta_data, root=0)

        # Since gathered will now be a list of list, we unpack it
        total_meta_data: list[dict[str, Any] | None] = []

        if gathered is not None:
            [total_meta_data.extend(m) for m in gathered]

        # TODO(MS): think about this some more. It is a bit hacky because now the `ctx._children` variable only has the child ctxs for rank 0, but the meta_data from all the other ranks too  # noqa: TD003
        ctx.meta["children"] = total_meta_data

    def __call__(
        self, params: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> float:
        """
        Evaluate the combined objective on rank 0 using MPI.

        Rank 0 broadcasts the evaluation context to all worker ranks,
        evaluates its own assigned slice locally, gathers the worker term
        values, re-raises any worker exceptions, gathers child metadata
        from all ranks, and reduces the full list of term values using the
        wrapped combined objective's reduction function.

        Args:
            params: Parameter dictionary for the current evaluation.
            ctx: Optional parent evaluation context. If ``None``, a new
                ``EvaluateContext`` is created.

        Returns:
            Reduced scalar loss value.

        Raises:
            RuntimeError: If called on a nonzero rank.
            Exception: Re-raises any exception returned from a worker rank.

        Side Effects:
            - Stores ``params`` in ``ctx.parameters``.
            - Broadcasts the evaluation context to all worker ranks.
            - Collects child metadata into ``ctx.meta["children"]``.
            - Stores the final reduced loss in ``ctx.loss``.

        """

        # Function to evaluate the objective function, to be called from rank 0

        if ctx is None:
            ctx = EvaluateContext()

        ctx.parameters = params

        # Ensure only rank 0 can call this
        if self.rank != 0:
            msg = "`__call__` can only be used on rank 0"
            raise RuntimeError(msg)

        # Broadcast the params to the worker ranks
        self.comm.bcast(ctx, root=0)

        local_terms: list[
            float | None
        ] = []  # So we get NaN in case the local compute fails
        try:
            # Compute one slice of the objective function on the main rank
            local_terms = self.evaluate_slice(params, ctx=ctx)
        finally:
            # Finally, we have to run the reduce. This must always happen since, otherwise, we might cause deadlocks
            # Sum up all local_totals into a global_total on every rank
            gathered_terms = self.comm.gather(local_terms, root=0)

        self.gather_meta_data(ctx)

        # Since gathered will now be a list of list, we unpack it
        terms: list[float | None] = []

        if gathered_terms is not None:
            [terms.extend(m) for m in gathered_terms]

        # If any exceptions from worker loops were sent to us we re-raise it
        for t in terms:
            if isinstance(t, Exception):
                raise t

        filtered_terms = self.cob.filter_terms(terms, ctx)
        ctx.loss = self.cob.reduction(filtered_terms)
        return ctx.loss

    def release_workers(self):
        # Only rank 0 needs to shut down workers
        if self.rank == 0 and self.size > 1:
            # send the poison-pill (None) so workers break out
            self.comm.bcast(Signal.ABORT, root=0)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ):
        self.release_workers()
