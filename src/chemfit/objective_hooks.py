"""Built-in hooks for objective-function evaluations."""

from __future__ import annotations

import time
import uuid

from chemfit.abstract_objective_function import EvaluateContext


class UUIDHook:
    """
    Assign a unique identifier to each objective evaluation.

    The UUID is stored as a string so that it can be serialized in context
    metadata and propagated across executor and MPI process boundaries.
    Register this one-sided hook with::

        objective.register_eval_hook(UUIDHook())
    """

    def __init__(self, meta_key: str = "evaluation_id") -> None:
        """
        Initialize the UUID hook.

        Args:
            meta_key: Key under ``ctx.meta`` where the UUID is stored.

        Raises:
            ValueError: If ``meta_key`` is empty.

        """
        if not meta_key:
            msg = "meta_key must not be empty"
            raise ValueError(msg)
        self.meta_key = meta_key

    def pre_eval(self, ctx: EvaluateContext) -> None:
        """Generate and store an evaluation UUID."""
        ctx.meta[self.meta_key] = str(uuid.uuid4())


class TimingHook:
    """
    Record the elapsed real time of an objective evaluation.

    The monotonic start time is kept in ``ctx.temp`` and is therefore local to
    the current evaluation. The completed duration is written to
    ``ctx.meta[meta_key]["elapsed_seconds"]``, which allows executor and MPI
    wrappers to propagate it back from worker processes.

    Register both halves of the hook on an objective with::

        objective.register_eval_hook(TimingHook())

    Timing is also recorded when the objective raises, because post-evaluation
    hooks run during exception unwinding.
    """

    def __init__(self, meta_key: str = "timing") -> None:
        """
        Initialize the timing hook.

        Args:
            meta_key: Key under ``ctx.meta`` where timing data is stored.

        Raises:
            ValueError: If ``meta_key`` is empty.

        """
        if not meta_key:
            msg = "meta_key must not be empty"
            raise ValueError(msg)
        self.meta_key = meta_key

    def pre_eval(self, ctx: EvaluateContext) -> None:
        """Start timing an evaluation."""
        ctx.temp.timing_start_ns = time.perf_counter_ns()

    def post_eval(self, ctx: EvaluateContext) -> None:
        """Stop timing and write the elapsed duration to the context."""
        start_ns = ctx.temp.timing_start_ns
        del ctx.temp.timing_start_ns

        elapsed_seconds = (time.perf_counter_ns() - start_ns) / 1_000_000_000
        ctx.meta[self.meta_key] = {"elapsed_seconds": elapsed_seconds}
