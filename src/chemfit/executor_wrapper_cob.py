from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ExecutorLike,
    ObjectiveFunctor,
)
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_utils import map_with_context


class ExecutorWrapperCOB(ObjectiveFunctor):
    def __init__(
        self, cob: CombinedObjectiveFunction, executor: ExecutorLike | None = None
    ):
        """
        Initialize a concurrent wrapper for a combined objective.

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

        self.cob = cob
        self.executor: ExecutorLike | None = executor

    def __enter__(self):
        """
        Enter the wrapper context.

        This method allows the wrapper to be used as a context manager for
        interface consistency with other combined-objective wrappers.

        Returns:
            The wrapper instance itself.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ): ...

    def __call__(
        self, parameters: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> float:
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
            ctx: Optional parent evaluation context. If ``None``, a new
                ``EvaluateContext`` is created.

        Returns:
            The reduced scalar loss computed from the evaluated terms.

        Side Effects:
            - Initializes the parent context through
            ``self.cob.prepare_evaluation(...)``.
            - Spawns one child context per objective term.
            - Evaluates terms through the selected executor.
            - Collects child metadata into ``ctx.meta["children"]``.
            - Stores the final reduced loss in ``ctx.loss``.

        """

        if ctx is None:
            ctx = EvaluateContext()

        ctx.parameters = parameters

        executor = ctx.executor or self.executor or ThreadPoolExecutor()

        assert executor is not None

        ctx.parameters = parameters
        ctx.meta.update({"n_terms": self.cob.n_terms()})

        with ctx.child_contexts(
            self.cob.n_terms(), configurator=self.cob.child_context_configurator
        ) as child_contexts:
            terms = map_with_context(
                executor,
                self.cob.evaluate_term,
                [parameters for _ in range(self.cob.n_terms())],
                range(self.cob.n_terms()),
                ctxs=child_contexts,
            )

        filtered_terms = self.cob.filter_terms(terms, ctx)

        ctx.loss = self.cob.apply_reduction(filtered_terms, ctx)

        return ctx.loss
