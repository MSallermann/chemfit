from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import Generic, TypeVar

from chemfit.abstract_objective_function import EvaluateContext, ExecutorLike
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    evaluate_weighted_term,
)
from chemfit.executor_utils import map_with_context

# CombinedObjectiveFunction is mutable and invariant, so policies preserve its
# exact parameter type.
ParametersT = TypeVar("ParametersT", bound=Mapping[str, object])


class ExecutorPolicy(Generic[ParametersT]):
    def __init__(
        self,
        executor: ExecutorLike | None = None,
    ):
        """
        Initialize an execution policy that uses a concurrent.futures style executor.

        This wrapper evaluates the terms of a ``CombinedObjectiveFunction``
        through an ``ExecutorLike`` instance. Each term is evaluated in its
        own child ``EvaluateContext``, and the resulting term values are
        reduced using the wrapped combined objective's reduction function.

        If no executor is provided here, the wrapper falls back to
        ``ctx.executor`` at call time. If neither is available, a
        ``ThreadPoolExecutor`` is created lazily.

        Args:
            cob: Combined objective function whose terms will be evaluated
                concurrently.
            executor: Optional default executor used when ``ctx.executor``
                is not set.

        """

        self.executor: ExecutorLike | None = executor

    def evaluate_terms(
        self,
        cob: CombinedObjectiveFunction[ParametersT],
        parameters: ParametersT,
        ctx: EvaluateContext,
    ) -> list[float | None]:
        """
        Evaluate the wrapped combined objective using an executor.

        This method prepares one child context per objective term, evaluates
        the terms through the configured executor, filters out any skipped
        terms, and reduces the remaining weighted term values using the
        wrapped combined objective's reduction function.

        Executor selection follows this order:
            1. ``ctx.executor``, if set
            2. ``self.executor``, if set
            3. a lazily created ``ThreadPoolExecutor``

        Args:
            parameters: Parameter dictionary for the evaluation.
            ctx: Parent evaluation context.

        Returns:
            The reduced scalar loss computed from the evaluated terms.

        Side Effects:
            - Initializes the parent context through ``self.cob.prepare_evaluation(...)``.
            - Spawns one child context per objective term.
            - Evaluates terms through the selected executor.
            - Collects child metadata into ``ctx.meta["children"]``.

        """

        executor = ctx.executor or self.executor

        created_executor: ThreadPoolExecutor | None = None
        if executor is None:
            created_executor = ThreadPoolExecutor()
            executor = created_executor

        ctx.meta.update({"n_terms": cob.n_terms()})

        try:
            with ctx.child_contexts(
                cob.n_terms(), configurator=cob.child_context_configurator
            ) as child_contexts:
                return map_with_context(
                    executor,
                    evaluate_weighted_term,
                    cob.objective_functions,
                    cob.weights,
                    repeat(cob.exception_handler, cob.n_terms()),
                    repeat(parameters, cob.n_terms()),
                    range(cob.n_terms()),
                    ctxs=child_contexts,
                )

        finally:
            if created_executor is not None:
                created_executor.shutdown()
