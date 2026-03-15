from __future__ import annotations

import contextlib
import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

T = TypeVar("T", covariant=True)  # noqa: PLC0105


class FutureLike(Protocol, Generic[T]):
    """
    Minimal protocol for future-like objects used by the evaluation framework.

    This protocol intentionally mirrors the subset of the interface provided by
    :class:`concurrent.futures.Future`
    """

    def result(self, timeout: float | None = None) -> T: ...
    def cancel(self) -> bool: ...


class ExecutorLike(Protocol):
    """
    Minimal executor protocol used for parallel evaluation.

    This interface is modeled after :class:`concurrent.futures.Executor`.
    """

    def submit(self, fn: Callable[..., T], /, *args, **kwargs) -> FutureLike[T]: ...

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterable[T]: ...


class ChildContextConfigurator(Protocol):
    """
    Protocol for configuring child evaluation contexts.

    The configurator is called once for each child context immediately
    after the parent context has spawned them. It may mutate the child
    context or the parent context in place to configure child-specific
    evaluation behavior, metadata, or resource usage.

    The ``idx_child_ctx`` argument is the absolute index of the current
    child within the spawned batch. The ``child_indices`` argument
    contains the full ordered list of child indices in that batch.

    """

    def __call__(
        self,
        idx_child_ctx: int,
        child_ctx: EvaluateContext,
        child_indices: Sequence[int],
        parent_ctx: EvaluateContext,
    ): ...


class EvaluateContext:
    def __init__(
        self,
        config: SimpleNamespace | None = None,
        shared: SimpleNamespace | None = None,
        executor: ExecutorLike | None = None,
    ):
        """
        Container for per-evaluation state.

        A new instance of `EvaluateContext` should generally be created for each
        evaluation of an objective function or quantity computation.
        Implementations write all per-call information into the context
        rather than storing it in the objective instance.
        This makes evaluation easier to reason about and
        compatible with concurrent execution.

        The context may also own a single batch of child contexts representing
        nested sub-evaluations. Such child contexts can be created explicitly
        with ``spawn_children()`` or managed with the ``child_contexts()``
        context manager.

        Args:
            config:
                Optional child-local evaluation configuration. This
                namespace is copied to spawned child contexts so that
                children inherit parent defaults but can be configured
                independently
            shared:
                Optional namespace for shared read-only state that may be
                reused across related contexts, such as parent/child
                evaluations.
            executor:
                Optional executor-like object that can be used by evaluation
                code to schedule parallel work.

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
            config:
                Child-local evaluation configuration for this context.
            shared:
                Shared state or resources reused across related contexts.

        """

        self._set_defaults(config, shared)
        self.executor: ExecutorLike | None = executor

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

    def _set_defaults(
        self, config: SimpleNamespace | None, shared: SimpleNamespace | None
    ):
        self.quantities: dict[str, Any] | None = None
        self.parameters: dict[str, Any] | None = None
        self.loss: float | None = None
        self.temp = SimpleNamespace()
        self.config = SimpleNamespace() if config is None else config
        self.shared = SimpleNamespace() if shared is None else shared
        self.meta: dict[str, Any] = {}
        self.executor = None
        self._children: list[EvaluateContext] = []

    def spawn_children(
        self, n_children: int, configurator: ChildContextConfigurator | None = None
    ) -> list[EvaluateContext]:
        """
        Create child contexts linked to this context.

        Each child receives a deep copy of ``config``, while sharing the
        same ``shared`` namespace and executor reference as the parent.

        An ``EvaluateContext`` is intended to manage at most one batch of
        child contexts per evaluation. Calling ``spawn_children()`` again on
        the same context replaces the previous child batch.

        In many cases, ``child_contexts()`` is the preferred interface, since
        it automatically collects child metadata when the nested evaluation
        scope exits.

        Args:
            n_children: Number of child contexts to create.
            configurator:
                Optional configurator applied once to each spawned child
                context immediately after creation.

        Returns:
            The newly created child contexts.

        """

        self._children = [
            EvaluateContext(
                config=copy.deepcopy(self.config),
                shared=self.shared,
                executor=self.executor,
            )
            for _ in range(n_children)
        ]

        if configurator is not None:
            for idx_child, child_ctx in enumerate(self._children):
                configurator(
                    idx_child_ctx=idx_child,
                    child_ctx=child_ctx,
                    child_indices=list(range(n_children)),
                    parent_ctx=self,
                )

        return self._children

    def collect_child_meta_data(self, recursive: bool = True):
        """
        Collect metadata from child contexts.

        The collected child metadata is stored in ``self.meta["children"]``.

        Components that spawn child contexts are generally expected to collect
        their child metadata before returning to their caller. The
        ``child_contexts()`` context manager provides a convenient scoped way
        to do this automatically.

        Args:
            If ``True``, metadata from all descendants is collected before
            serializing the immediate children. This produces a fully
            materialized metadata tree.

            If ``False``, only the immediate children are serialized. This
            can be useful when nested components manage their own metadata
            collection and have already populated their ``meta`` fields.

        Notes:
            In most cases ``recursive=True`` is the safest choice, since it
            ensures that nested child contexts are fully represented in the
            resulting metadata structure.

        """

        if len(self._children) > 0:
            if recursive:
                [c.collect_child_meta_data(recursive) for c in self._children]
            self.meta["children"] = [c.to_meta_data() for c in self._children]

    @contextlib.contextmanager
    def child_contexts(
        self,
        n_children: int,
        configurator: ChildContextConfigurator | None = None,
        recursive: bool = True,
    ):
        """
        Create a scoped child-context batch and collect its metadata on exit.

        This context manager is a convenience wrapper around
        ``spawn_children()`` and ``collect_child_meta_data()``. It is intended
        for nested evaluations where the component spawning child contexts is
        also responsible for collecting their metadata before returning.

        Args:
            n_children: Number of child contexts to create.
            configurator:
                Optional configurator applied to each spawned child context.
            recursive:
                Passed to ``collect_child_meta_data()`` when the scope exits.

        Yields:
            The list of spawned child contexts.

        Notes:
            Child metadata is collected automatically when the context manager
            exits, even if an exception is raised inside the managed block.

        """

        children = self.spawn_children(n_children, configurator=configurator)
        try:
            yield children
        finally:
            self.collect_child_meta_data(recursive)

    def __getstate__(self) -> dict[str, Any]:
        """
        Return the worker-input state for this context.

        Returns:
            Dictionary containing the context state needed to initialize the
            context in a worker or other execution environment.

        """
        return {
            "parameters": self.parameters,
            "config": self.config,
            "shared": self.shared,
        }

    def __setstate__(self, state: dict[str, Any]):
        """
        Restore worker-input state into this context.

        Args:
            state:
                State previously produced by ``__getstate__()``.

        """
        self._set_defaults(config=state["config"], shared=state["shared"])
        self.parameters = state["parameters"]

    def to_result_state(self) -> dict[str, Any]:
        """
        Return the result-bearing state of this context.

        Returns:
            Dictionary containing the evaluation results recorded in this
            context. This state is intended for child/worker-to-parent
            synchronization and does not include shared resources or child
            context objects.

        """

        return {
            "parameters": self.parameters,
            "loss": self.loss,
            "quantities": self.quantities,
            "meta": self.meta,
        }

    def apply_result_state(self, state: dict[str, Any]):
        """
        Apply result-bearing state from another context.

        Args:
            state:
                State previously produced by ``to_result_state()``.

        Side Effects:
            Updates ``parameters``, ``loss``, ``quantities``, and ``meta``
            on this context.

        """

        self.parameters = state["parameters"]
        self.loss = state["loss"]
        self.quantities = state["quantities"]
        self.meta = state["meta"]


class ObjectiveFunctor:
    def __call__(
        self, parameters: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> float:
        """
        Evaluate the objective function.

        Implementations should compute a scalar loss from the given
        parameter dictionary. All per-evaluation state must be written
        into the provided `ctx`. If no context is supplied,
        a new one should be created internally.

        Args:
            parameters (dict[str, Any]): Mapping of parameter names to
                float values.
            ctx (EvaluateContext | None): Optional evaluation context. If
                None, a new `EvaluateContext` should be created.

        Returns:
            float: The computed scalar loss.

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
        Initialize a quantity computer.

        A `QuantityComputer` maps a parameter dictionary to a dictionary
        of intermediate quantities, typically used by an objective
        function. Instances may hold static configuration, but
        should not store per-evaluation state internally.

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

        Side Effects:
            Stores ``parameters`` in ``ctx.parameters``.
            Merges ``self.static_meta_data`` into ``ctx.meta``.
            Stores the computed quantities in ``ctx.quantities``.

        """

        if ctx is None:
            ctx = EvaluateContext()

        ctx.parameters = parameters

        ctx.meta.update(self.static_meta_data)
        ctx.quantities = self._compute(parameters, ctx)

        return ctx.quantities

    def _compute(
        self, parameters: dict[str, Any], ctx: EvaluateContext
    ) -> dict[str, Any]:
        """Compute dictionary of quantities for a given set of parameters."""
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
        Objective function composed of a `QuantityComputer` and a loss function.

        This objective first computes intermediate quantities using
        ``quantity_computer`` and then applies ``loss_function`` to
        obtain a scalar loss.

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

        Side Effects:
            Stores the computed loss in ``ctx.loss``.
            Updates ``ctx.meta`` with ``self.static_meta_data`` after the
            wrapped ``QuantityComputer`` may have already added metadata.
            Populates ``ctx.quantities`` and ``ctx.parameters`` via the
            wrapped ``QuantityComputer``.

        Notes:
            ``loss_function`` may accept either ``(quantities)`` or
            ``(quantities, parameters) as positional args``.

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
