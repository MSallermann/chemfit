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


class AsyncWrapperCOB(ObjectiveFunctor):
    def __init__(
        self, cob: CombinedObjectiveFunction, executor: ExecutorLike | None = None
    ):
        """
        Wrap a CombinedObjectiveFunction for concurrent async evaluation.

        This wrapper allows the terms of a `CombinedObjectiveFunction`
        to be evaluated concurrently using `asyncio` and a thread pool.
        Each term receives its own `EvaluateContext`, and the results
        are aggregated back into the main context.

        Args:
            cob: The combined objective function to evaluate.
            executor: Optional thread pool for running blocking subprocesses.

        """
        self.cob = cob
        self.executor: ExecutorLike | None = executor

    def __enter__(self):
        """
        Enable use as a context manager.

        We implement the Asyncwrapper as a context manager for consistency with MPIWrapperCOB.

        Returns:
            Self.

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
        Synchronously evaluate the objective via asyncio.

        Args:
            params: Parameter dictionary for evaluation.

        Returns:
            Total objective value.

        """

        if ctx is None:
            ctx = EvaluateContext()

        contexts, idx_list = self.cob.prepare_evaluation(parameters=parameters, ctx=ctx)

        if ctx.executor is None:
            self.executor = ThreadPoolExecutor()

        executor = ctx.executor if ctx.executor is not None else self.executor

        assert executor is not None

        terms = map_with_context(
            executor,
            self.cob.evaluate_term,
            [parameters for _ in range(self.cob.n_terms())],
            idx_list,
            ctxs=contexts,
        )

        terms = [t for t in terms if t is not None]

        ctx.loss = self.cob.reduction(list(terms))

        ctx.collect_child_meta_data()

        return ctx.loss
