from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.combined_objective_function import DEFAULT_SLICE, CombinedObjectiveFunction

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor


class AsyncWrapperCOB(ObjectiveFunctor):
    def __init__(
        self, cob: CombinedObjectiveFunction, executor: ThreadPoolExecutor | None = None
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
        self._executor = executor

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

    async def async_term(self, params: dict[str, Any], ctx: EvaluateContext, idx: int):
        """
        Evaluate a single term of the objective asynchronously.

        Args:
            params: Parameter dictionary passed to the objective.
            idx: Index of the term to compute.

        Returns:
            The term's contribution value.

        """
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            self._executor,
            lambda params, ctx: self.cob.weights[idx]
            * self.cob.objective_functions[idx](params, ctx),
            params,
            ctx,
        )

    async def async_call(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,
        idx_slice: slice = DEFAULT_SLICE,
    ) -> float:
        """
        Evaluate all objective terms concurrently.

        Args:
            params: Parameter dictionary for evaluation.

        Returns:
            Sum of all term contributions.

        """

        contexts, idx_list = self.cob.prepare_evaluation(
            parameters=parameters, ctx=ctx, idx_slice=idx_slice
        )

        assert ctx.loss is not None  # for pyright

        futures = [
            self.async_term(parameters, ctx, idx)
            for ctx, idx in zip(contexts, idx_list, strict=True)
        ]

        results = await asyncio.gather(*futures)

        CombinedObjectiveFunction.collect_per_term_data(ctx, contexts)

        ctx.loss = float(sum(results))
        return ctx.loss

    def __call__(
        self,
        params: dict[str, Any],
        ctx: EvaluateContext | None = None,
        idx_slice: slice = DEFAULT_SLICE,
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

        return float(asyncio.run(self.async_call(params, ctx, idx_slice)))
