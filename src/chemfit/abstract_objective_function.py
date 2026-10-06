from __future__ import annotations

import contextlib
import copy
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from functools import partial
from inspect import signature
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Generic, Protocol, cast

# Python 3.10's typing.Concatenate rejects the ellipsis used in our aliases.
from typing_extensions import Concatenate, Self, TypeVar  # noqa: UP035

if TYPE_CHECKING:
    from collections.abc import Iterable

T = TypeVar("T", covariant=True)  # noqa: PLC0105

FutureLike = Future


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
    evaluation behavior or metadata.

    The ``idx_child_ctx`` argument is the absolute index of the current
    child within the spawned batch.

    """

    def __call__(
        self,
        idx_child_ctx: int,
        child_ctx: EvaluateContext,
        num_children: int,
        parent_ctx: EvaluateContext,
    ): ...


class EvaluateContext:
    def __init__(
        self,
        config: SimpleNamespace | None = None,
        shared: SimpleNamespace | None = None,
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
                Optional namespace for state shared by related contexts, such
                as parent/child evaluations. Mutable contents require
                synchronization when contexts can execute concurrently. A
                process backend serializes this namespace as worker input;
                worker mutations are not returned to the caller.

        Attributes:
            quantities (dict[str, Any] | None): Intermediate quantities
                computed during evaluation. Implementations may leave this
                as None if no quantities are produced.
            parameters (Mapping[str, object] | None): Parameter mapping used
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
        self.parameters: Mapping[str, object] | None = None
        self.loss: float | None = None
        self.temp = SimpleNamespace()
        self.config = SimpleNamespace() if config is None else config
        self.shared = SimpleNamespace() if shared is None else shared
        self.meta: dict[str, Any] = {}
        self._children: list[EvaluateContext] = []

    def spawn_children(
        self, n_children: int, configurator: ChildContextConfigurator | None = None
    ) -> list[EvaluateContext]:
        """
        Create child contexts linked to this context.

        Each child receives a deep copy of ``config``, while sharing the
        same ``shared`` namespace as the parent.

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
            EvaluateContext(config=copy.deepcopy(self.config), shared=self.shared)
            for _ in range(n_children)
        ]

        if configurator is not None:
            for idx_child, child_ctx in enumerate(self._children):
                configurator(
                    idx_child_ctx=idx_child,
                    child_ctx=child_ctx,
                    num_children=n_children,
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
            recursive: If ``True``, metadata from all descendants is collected before
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
        for nested evaluations where the component, that is spawning child contexts,
        is also responsible for collecting their metadata before returning.

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

        Example:
            .. code-block:: python

                with ctx.child_contexts(
                    n_children=self.n_terms(),
                    configurator=self.child_context_configurator,
                ) as child_ctxs:
                    terms = []
                    for idx, ctx_term in enumerate(child_ctxs):
                        terms.append(
                            self.evaluate_term(parameters, idx, ctx_term)
                        )

        """

        try:
            children = self.spawn_children(n_children, configurator=configurator)
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
            "meta": self.meta,
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
        self.meta = state["meta"]

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


# Variance here follows the direction in which values cross the public API:
#
# * Parameters are only consumed by ObjectiveFunctor and QuantityComputer, so
#   they are contravariant. For example, an objective that accepts a broad
#   Mapping[str, object] is safe wherever one accepting dict[str, float] is
#   required. The Mapping bound is only an upper bound; it does not force user
#   callbacks to spell their parameter annotation as Mapping.
# * Quantities are produced by QuantityComputer, so they are covariant.  A
#   computer returning a more specific quantity dictionary can therefore be
#   used where a less specific result is expected.
# * A loss function must consume the same quantity type produced by its paired
#   computer. LossQuantitiesT is consequently kept invariant to tie those two
#   sides together during inference.

ParametersT_contra = TypeVar(
    "ParametersT_contra", bound=Mapping[str, object], contravariant=True
)

QuantitiesT_co = TypeVar(
    "QuantitiesT_co",
    bound=Mapping[str, object],
    covariant=True,
    default=dict[str, Any],
)

LossQuantitiesT = TypeVar(
    "LossQuantitiesT",
    bound=Mapping[str, object],
)


class PreEvaluationHook(Protocol):
    """Callback that runs before an objective evaluation."""

    def pre_eval(self, ctx: EvaluateContext) -> None:
        """Run before the objective evaluation."""
        ...


class PostEvaluationHook(Protocol):
    """Callback that runs after an objective evaluation."""

    def post_eval(self, ctx: EvaluateContext) -> None:
        """Run after the objective evaluation."""
        ...


EvaluationHook = PreEvaluationHook | PostEvaluationHook


class ObjectiveFunctor(Generic[ParametersT_contra]):
    class PostEvalHookError(RuntimeError):
        """
        Report failures raised by one or more post-evaluation hooks.

        Attributes:
            exceptions: Exceptions raised by the post-evaluation hooks, in
                hook registration order.

        """

        def __init__(self, exceptions: list[Exception]) -> None:
            """Initialize."""
            self.exceptions = tuple(exceptions)
            msg = f"{len(exceptions)} post-evaluation hook(s) failed:\n"
            for e in self.exceptions:
                msg += f"{e}\n"

            super().__init__(msg)

        def __reduce__(self):
            return type(self), (list(self.exceptions),)

    def __init__(self) -> None:
        """
        Initialize objective function.

        Attributes:
            static_meta_data: Reusable metadata merged into ``ctx.meta`` after
                objective evaluation and before post-evaluation hooks.

        """

        self.pre_eval_hooks: list[Callable[[EvaluateContext], None]] = []
        self.post_eval_hooks: list[Callable[[EvaluateContext], None]] = []
        self.static_meta_data: dict[str, Any] = {}

    def with_meta(self, /, **meta: Any) -> Self:
        """
        Return a copy with additional static evaluation metadata.

        Existing metadata is preserved unless a key is supplied again. The
        source objective is unchanged. Existing hook objects are retained, but
        the returned objective has independent registration lists, so later
        hook registration does not affect the source.
        """

        new = copy.copy(self)
        new.pre_eval_hooks = self.pre_eval_hooks.copy()
        new.post_eval_hooks = self.post_eval_hooks.copy()
        new.static_meta_data = {**self.static_meta_data, **meta}
        return new

    def _create_context(self) -> EvaluateContext:
        """Create the default context used when none is supplied."""
        return EvaluateContext()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Guard implementations against overriding __call__."""

        allow_custom_call = kwargs.pop("allow_custom_call", False)
        super().__init_subclass__(**kwargs)

        if not allow_custom_call and cls.__call__ is not ObjectiveFunctor.__call__:
            msg = (
                f"{cls.__qualname__} must implement `_evaluate()` "
                "instead of overriding `__call__()`. "
                "Very advanced users may override this behavior by setting "
                "`allow_custom_call=True` in their class definition. "
            )
            raise TypeError(msg)

    def _child_objectives(self) -> tuple[ObjectiveFunctor[Any], ...]:
        """Return objectives evaluated in distinct child contexts."""
        return ()

    def register_eval_hook(
        self,
        *,
        hook: EvaluationHook | None = None,
        pre: Callable[[EvaluateContext], None] | None = None,
        post: Callable[[EvaluateContext], None] | None = None,
        recursive: bool = False,
    ) -> Self:
        """
        Register evaluation callbacks on this objective.

        Pass either a hook object implementing ``pre_eval`` and/or
        ``post_eval``, or pass the callbacks directly through ``pre`` and
        ``post``. At least one callback is required. Registered callbacks are
        invoked in registration order for their respective evaluation phase.

        Args:
            hook: Optional object implementing at least one of
                :class:`PreEvaluationHook` or :class:`PostEvaluationHook`.
                Cannot be combined with ``pre`` or ``post``.
            pre: Optional callback invoked before objective evaluation.
            post: Optional callback invoked after objective evaluation.
            recursive: Also register on descendant objectives in the call tree.
                Defaults to registering only on this objective.

        Raises:
            TypeError: If a hook object is combined with direct callbacks, no
                callback is supplied, or a supplied callback is not callable.

        Note:
            If contexts are reused, pre-evaluation hooks may observe state from
            the previous evaluation. Before the hooks are invoked, ``ctx.parameters``
            is updated and ``ctx.loss`` is set to None.

            Recursive registration uses the same hook instance on every
            objective, so hooks must keep per-evaluation state in ``ctx``.
            Register recursive hooks before submitting work to process
            executors, and on every MPI rank before entering worker loops.

            Every post-evaluation hook is attempted, even if an earlier hook
            raises an exception. If evaluation succeeds, hook exceptions are
            collected and raised together as ``PostEvalHookError``. If
            ``_evaluate`` raises, its exception remains primary and is
            available to post-evaluation hooks as ``ctx.temp.exception``.

            Hooks may run concurrently and, with process-based execution, may
            run in another process. They should not mutate captured mutable
            objects, global variables, or other shared state: such mutations
            can race between threads and are not propagated back from worker
            processes. Store per-evaluation output on ``ctx`` instead, usually
            in ``ctx.meta``. Hooks used by a process-based executor must also
            be serializable.

        """
        if hook is not None and (pre is not None or post is not None):
            msg = "pass either hook or pre/post callbacks, not both"
            raise TypeError(msg)

        if hook is not None:
            pre_eval = getattr(hook, "pre_eval", None)
            post_eval = getattr(hook, "post_eval", None)
        else:
            pre_eval = pre
            post_eval = post

        if pre_eval is None and post_eval is None:
            msg = "evaluation hook must implement pre_eval() or post_eval()"
            raise TypeError(msg)

        if pre_eval is not None and not callable(pre_eval):
            msg = "evaluation hook pre_eval attribute must be callable"
            raise TypeError(msg)

        if post_eval is not None and not callable(post_eval):
            msg = "evaluation hook post_eval attribute must be callable"
            raise TypeError(msg)

        # Traverse child evaluation scopes without recursive Python calls.
        pending: list[ObjectiveFunctor[Any]] = [self]
        while pending:
            objective = pending.pop()
            if pre_eval is not None:
                objective.pre_eval_hooks.append(
                    cast("Callable[[EvaluateContext], None]", pre_eval)
                )
            if post_eval is not None:
                objective.post_eval_hooks.append(
                    cast("Callable[[EvaluateContext], None]", post_eval)
                )
            if recursive:
                pending.extend(reversed(objective._child_objectives()))  # noqa: SLF001

        return self

    def _evaluate(self, parameters: ParametersT_contra, ctx: EvaluateContext) -> float:
        """
        Evaluate the objective function.

        Implementations should compute a scalar loss from the given
        parameter dictionary. All per-evaluation state must be written
        into the provided `ctx`.

        Args:
            parameters: Parameter mapping for the current evaluation.
            ctx (EvaluateContext): Evaluation context.

        Returns:
            float: The computed scalar loss.

        Notes:
            - Implementations should avoid mutating `self` during the
              call. All per-evaluation information should be placed in
              `ctx` instead.

        """
        raise NotImplementedError

    def _invoke_post_eval_hooks(self, ctx: EvaluateContext, evaluation_failed: bool):
        """Invoke the post evaluation hooks."""
        post_hook_exceptions = []
        for cb in self.post_eval_hooks:
            try:
                cb(ctx)
            except Exception as e:  # noqa: PERF203
                post_hook_exceptions.append(e)

        if not evaluation_failed and post_hook_exceptions:
            raise self.PostEvalHookError(post_hook_exceptions)

    def _apply_static_meta(self, ctx: EvaluateContext) -> None:
        """Merge this objective's static metadata into an evaluation context."""

        ctx.meta.update(self.static_meta_data)

    def _begin_evaluation(self, parameters: ParametersT_contra, ctx: EvaluateContext):
        """
        Initialize an objective evaluation and invoke pre-evaluation hooks.

        The context is updated with the current parameters and its previous loss
        is cleared before hooks are invoked.

        This is the first half of the objective evaluation lifecycle. Every
        successful call should eventually be paired with ``_end_evaluation()``.
        """

        ctx.parameters = parameters
        ctx.loss = None
        for cb in self.pre_eval_hooks:
            cb(ctx)

    def _end_evaluation(
        self,
        ctx: EvaluateContext,
        exception: BaseException | None,
    ):
        """
        Finalize an objective evaluation and invoke post-evaluation hooks.

        Args:
            ctx:
                Evaluation context associated with the evaluation.
            exception:
                Exception raised during evaluation, or ``None`` if evaluation
                completed successfully.

        Notes:
            Static objective metadata is merged into ``ctx.meta`` before
            post-evaluation hooks run. This occurs after ``_evaluate()``, so
            objective metadata takes precedence over metadata contributed by
            a wrapped quantity computer.

            When ``exception`` is not ``None``, the context loss is cleared and
            the exception is stored in ``ctx.temp.exception`` before post-hooks
            are invoked.

            Post-hook failures are raised as ``PostEvalHookError`` only when the
            evaluation itself succeeded. If evaluation already failed, post-hook
            failures do not replace the original exception.

        """

        self._apply_static_meta(ctx)

        if exception is None:
            ctx.temp.exception = None
            self._invoke_post_eval_hooks(
                ctx,
                evaluation_failed=False,
            )
        else:
            ctx.loss = None
            ctx.temp.exception = exception
            self._invoke_post_eval_hooks(
                ctx,
                evaluation_failed=True,
            )

    def __call__(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext | None = None,
    ) -> float:
        """
        Evaluate the objective function.

        Args:
            parameters: Parameter mapping for the current evaluation.
            ctx (EvaluateContext | None): Optional evaluation context. If
                None, a new `EvaluateContext` is created.

        Notes:
            - Derived classes must implement ``_evaluate`` rather than
              overriding ``__call__``.
            - This method is synchronous. For batch or concurrent execution,
              prepare the objective with a scheduler. Distinct overlapping
              evaluations must use distinct contexts.

        Raises:
            PostEvalHookError: If evaluation succeeds and one or more
                post-evaluation hooks raise an exception.

        """

        if ctx is None:
            ctx = self._create_context()

        try:
            self._begin_evaluation(parameters=parameters, ctx=ctx)
            value = self._evaluate(parameters, ctx)
            ctx.loss = value
        except BaseException as e:
            self._end_evaluation(ctx, e)
            raise
        else:
            self._end_evaluation(ctx, None)

        return value


LossFunction = Callable[Concatenate[LossQuantitiesT, ...], float]


def _loss_function_takes_parameters(
    loss_function: LossFunction[Any],
) -> bool:
    """Determine which supported positional signature a loss function uses."""

    try:
        loss_signature = signature(loss_function)
    except (TypeError, ValueError) as exception:
        msg = (
            "loss_function must have an inspectable signature accepting "
            "(quantities) or (quantities, parameters)"
        )
        raise TypeError(msg) from exception

    quantities = object()
    parameters = object()

    try:
        loss_signature.bind(quantities)
    except TypeError:
        pass
    else:
        # Preserve the historical preference for the one-argument form when
        # an optional second positional argument or *args allows both forms.
        return False

    try:
        loss_signature.bind(quantities, parameters)
    except TypeError as exception:
        msg = (
            "loss_function must accept (quantities) or "
            f"(quantities, parameters); got signature {loss_signature}"
        )
        raise TypeError(msg) from exception

    return True


class QuantityComputer(Generic[ParametersT_contra, QuantitiesT_co]):
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

    def with_meta(self, /, **meta: Any) -> Self:
        """
        Return a copy with additional static evaluation metadata.

        Existing metadata is preserved unless a key is supplied again. The
        source quantity computer is unchanged.
        """

        new = copy.copy(self)
        new.static_meta_data = {**self.static_meta_data, **meta}
        return new

    def __call__(
        self, parameters: ParametersT_contra, ctx: EvaluateContext | None = None
    ) -> QuantitiesT_co:
        """
        Compute quantities for the given parameters.

        Args:
            parameters: Parameter mapping for the current computation.
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
        self, parameters: ParametersT_contra, ctx: EvaluateContext
    ) -> QuantitiesT_co:
        """Compute dictionary of quantities for a given set of parameters."""
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Guard implementations against overriding __call__."""

        super().__init_subclass__(**kwargs)

        if cls.__call__ is not QuantityComputer.__call__:
            msg = (
                f"{cls.__qualname__} must implement _compute() "
                "instead of overriding __call__()"
            )
            raise TypeError(msg)

    def with_loss(
        self, loss_function: LossFunction[QuantitiesT_co], /, **kwargs: Any
    ) -> QuantityComputerObjectiveFunction[ParametersT_contra, QuantitiesT_co]:
        """
        Create a new QuantityComputerObjectiveFunction from this QuantityComputer.

        Args:
            loss_function (LossFunction): The loss function to use.

        Returns:
            QuantityComputerObjectiveFunction: A new QuantityComputerObjectiveFunction

        """
        return QuantityComputerObjectiveFunction(
            loss_function=partial(loss_function, **kwargs),
            quantity_computer=self,
        )


class QuantityComputerObjectiveFunction(
    ObjectiveFunctor[ParametersT_contra],
    Generic[ParametersT_contra, QuantitiesT_co],
):
    def __init__(
        self,
        loss_function: LossFunction[QuantitiesT_co],
        quantity_computer: QuantityComputer[ParametersT_contra, QuantitiesT_co],
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

        """

        super().__init__()
        self.quantity_computer = quantity_computer
        self.loss_function = loss_function
        self._loss_takes_parameters = _loss_function_takes_parameters(loss_function)

    def _evaluate(self, parameters: ParametersT_contra, ctx: EvaluateContext) -> float:
        """
        Compute the objective loss.

        This method computes intermediate quantities using the quantity
        computer and applies the loss function. The inherited ``__call__``
        method stores the returned loss in ``ctx.loss``.

        Args:
            parameters: Parameter mapping for the current evaluation.
            ctx (EvaluateContext): Evaluation context.

        Returns:
            float: The computed scalar loss.

        Side Effects:
            Updates ``ctx.meta`` with ``self.static_meta_data`` after the
            wrapped ``QuantityComputer`` may have already added metadata.
            Populates ``ctx.quantities`` and ``ctx.parameters`` via the
            wrapped ``QuantityComputer``.

        Notes:
            ``loss_function`` may accept either ``(quantities)`` or
            ``(quantities, parameters)`` as positional arguments. Its
            signature is inspected when this objective is constructed, so a
            ``TypeError`` raised inside the loss function is propagated
            without retrying the call with different arguments.

        """

        quantities = self.quantity_computer(parameters, ctx)

        if self._loss_takes_parameters:
            loss = self.loss_function(quantities, parameters)
        else:
            loss = self.loss_function(quantities)

        return loss
