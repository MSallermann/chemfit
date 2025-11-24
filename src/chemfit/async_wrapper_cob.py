from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from chemfit.abstract_objective_function import ObjectiveFunctor
from chemfit.combined_objective_function import CombinedObjectiveFunction

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor


class AsyncWrapperCOB(ObjectiveFunctor):
    def __init__(
        self, cob: CombinedObjectiveFunction, executor: ThreadPoolExecutor | None = None
    ):
        """
        Wrap a CombinedObjectiveFunction for concurrent async evaluation.

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

    async def get_contrib(self, params: dict[str, Any], idx: int):
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
            self.cob,
            params,
            slice(idx, idx + 1),
        )

    def get_meta_data(self) -> dict[str, Any]:
        """
        Get metadata from the underlying objective.

        Returns:
            Metadata dictionary.

        """
        return self.cob.get_meta_data()

    def gather_meta_data(self) -> list[dict[str, Any] | None]:
        """
        Get metadata for all sub-objectives.

        Returns:
            List of metadata dictionaries or None.

        """
        return self.cob.gather_meta_data()

    async def async_call(self, params: dict[str, Any]) -> float:
        """
        Evaluate all objective terms concurrently.

        Args:
            params: Parameter dictionary for evaluation.

        Returns:
            Sum of all term contributions.

        """
        futures = [self.get_contrib(params, idx) for idx in range(self.cob.n_terms())]
        results = await asyncio.gather(*futures)
        return float(sum(results))

    def __call__(self, params: dict[str, Any]) -> float:
        """
        Synchronously evaluate the objective via asyncio.

        Args:
            params: Parameter dictionary for evaluation.

        Returns:
            Total objective value.

        """
        return float(asyncio.run(self.async_call(params)))
