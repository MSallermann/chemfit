from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

T = TypeVar("T", covariant=True)  # noqa: PLC0105


class Future(Generic[T], Protocol):
    def result(self, timeout: float | None = None) -> T: ...
    def cancel(self): ...


class Executor(Protocol):
    def submit(self, fn: Callable[..., T], /, *args, **kwargs) -> Future[T]: ...

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Generator[T]: ...


class EvaluateContext:
    def __init__(
        self,
        temp: SimpleNamespace | None = None,
        static: SimpleNamespace | None = None,
        executor: Executor | None = None,
    ):
        """
        Container for per-evaluation state.

        A new instance of `EvaluateContext` should be created for each
        evaluation of an objective function or quantity computation.
        Implementations write all per-call information into the context
        rather than storing it in the objective instance. This makes the
        system safe for concurrent or asynchronous evaluation.

        Attributes:
            quantities (dict[str, Any] | None): Intermediate quantities
                computed during evaluation. Implementations may leave this
                as None if no quantities are produced.
            parameters (dict[str, Any] | None): Parameter dictionary used
                for this evaluation.
            loss (float | None): Final scalar loss value. Set by
                `ObjectiveFunctor` implementations.
            meta (dict[str, Any]): Free-form metadata dictionary.
                Implementations may add diagnostic or structural
                information here as needed.
                Meta data from child contexts may be collected into the parent
            temp (SimpleNamespace): Scratch space for temporary values
                during evaluation. Nothing stored here is part of the
                public API. It is omitted from the `to_meta_data` function.
                The `temp` meta is *shallow* copied to child contexts.

        """

        self._set_defaults(temp, static)
        self.executor: Executor | None = executor

    def _set_defaults(
        self, temp: SimpleNamespace | None, static: SimpleNamespace | None
    ):
        self.quantities: dict[str, Any] | None = None
        self.parameters: dict[str, Any] | None = None
        self.loss: float | None = None
        self.temp = SimpleNamespace() if temp is None else temp
        self.static = SimpleNamespace() if static is None else static
        self.meta: dict[str, Any] = {}
        self.executor = None
        self._children: list[EvaluateContext] = []

    def __getstate__(self) -> dict[str, Any]:
        return {"parameters": self.parameters, "temp": self.temp, "static": self.static}

    def __setstate__(self, state: dict[str, Any]):
        self._set_defaults(temp=state["temp"], static=state["static"])
        self.parameters = state["parameters"]

    def spawn_children(self, n_children: int) -> list[EvaluateContext]:
        """Spawns dependent child contexts, with a deepcopy of the `temp` data and access to the same static data."""
        self._children = [
            EvaluateContext(
                temp=copy.deepcopy(self.temp),
                static=self.static,
                executor=self.executor,
            )
            for _ in range(n_children)
        ]
        return self._children

    def collect_child_meta_data(self, recursive: bool = True):
        """Collect the meta data from child contexts."""
        if len(self._children) > 0:
            if recursive:
                [c.collect_child_meta_data(recursive) for c in self._children]
            self.meta["children"] = [c.to_meta_data() for c in self._children]

    def to_meta_data(self) -> dict[str, Any]:
        """
        Return a dictionary summarizing the evaluation state.

        Returns:
            dict[str, Any]: A dictionary containing the fields
            `quantities`, `parameters`, `loss`, and `meta`.

        """
        return {
            "quantities": self.quantities,
            "parameters": self.parameters,
            "loss": self.loss,
            "meta": self.meta,
        }


class ObjectiveFunctor:
    def __call__(
        self, parameters: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> float:
        """
        Evaluate the objective function.

        Implementations should compute a scalar loss from the given
        parameter dictionary. All per-evaluation state must be written
        into the provided `EvaluateContext`. If no context is supplied,
        a fresh one should be created internally.

        Args:
            parameters (dict[str, Any]): Mapping of parameter names to
                float values.
            ctx (EvaluateContext | None): Optional evaluation context. If
                None, a new `EvaluateContext` should be created.

        Returns:
            float: The computed scalar loss value.

        Notes:
            - Implementations should avoid mutating `self` during the
              call. All per-evaluation information should be placed in
              `ctx` instead.
            - This method is synchronous. For concurrent or asynchronous
              evaluation, use one `EvaluateContext` per call and invoke
              this method in multiple threads/tasks.

        """
        raise NotImplementedError


class QuantityComputer:
    def __init__(self):
        """
        Base class for computing intermediate quantities.

        A `QuantityComputer` maps a parameter dictionary to a dictionary
        of intermediate quantities, typically used by an objective
        function. It should not store per-evaluation state internally.

        Attributes:
            static_meta_data (dict[str, Any]): Static metadata associated
                with this quantity computer. This is merged into
                `ctx.meta` on each call.

        """
        self.static_meta_data: dict[str, Any] = {}  # For static meta data

    def __call__(
        self, parameters: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> dict[str, Any]:
        """
        Compute quantities for the given parameters.

        Args:
            parameters (dict[str, Any]): Parameter dictionary.
            ctx (EvaluateContext | None): Optional context. If None, a
                new one is created.

        Returns:
            dict[str, Any]: The computed quantity dictionary.

        Notes:
            Implementations of `_compute` must not mutate `self`. All
            per-evaluation information should be written into `ctx`.

        """

        if ctx is None:
            ctx = EvaluateContext()

        ctx.parameters = parameters
        ctx.quantities = self._compute(parameters, ctx)

        ctx.meta.update(self.static_meta_data)

        return ctx.quantities

    def _compute(
        self, parameters: dict[str, Any], ctx: EvaluateContext
    ) -> dict[str, Any]:
        """Compute dictionary of quantities for a given set of new parameters."""
        raise NotImplementedError


LossFunction = (
    Callable[[dict[str, Any]], float]
    | Callable[[dict[str, Any], dict[str, Any]], float]
)


class QuantityComputerObjectiveFunction(ObjectiveFunctor):
    def __init__(
        self,
        loss_function: LossFunction,
        quantity_computer: QuantityComputer,
    ) -> None:
        """
        Objective function composed of a `QuantityComputer` and a loss.

        This class first evaluates the `quantity_computer` to produce
        intermediate quantities and then applies the `loss_function` to
        compute a scalar loss.

        Args:
            loss_function (Callable): A function with signature:

                `loss_function(quantities) -> float`
                or
                `loss_function(quantities, parameters) -> float`

            quantity_computer (QuantityComputer): Object responsible for
                computing intermediate quantities.

        Attributes:
            static_meta_data (dict[str, Any]): Static metadata associated
                with this objective. Merged into `ctx.meta` on each call.

        """

        super().__init__()
        self.quantity_computer = quantity_computer
        self.static_meta_data: dict[str, Any] = {}
        self.loss_function = loss_function

    def __call__(
        self, parameters: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> float:
        """
        Compute the objective loss.

        This method:
        1. Computes intermediate quantities using the quantity computer.
        2. Applies the loss function.
        3. Stores results in the evaluation context.

        Args:
            parameters (dict[str, Any]): Parameter dictionary.
            ctx (EvaluateContext | None): Optional context. If None, a
                new one is created.

        Returns:
            float: The computed scalar loss.

        """

        if ctx is None:
            ctx = EvaluateContext()

        quantities = self.quantity_computer(parameters, ctx)

        # Update or set static meta data if needed
        ctx.meta.update(self.static_meta_data)

        try:
            ctx.loss = self.loss_function(quantities)  # pyright: ignore[reportCallIssue] # we actually handle this with the signature checking
        except TypeError:
            ctx.loss = self.loss_function(quantities, parameters)  # pyright: ignore[reportCallIssue] # we actually handle this with the signature checking

        return ctx.loss
