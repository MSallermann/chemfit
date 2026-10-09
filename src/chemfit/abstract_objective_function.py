from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from functools import partial
from inspect import signature
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Generic, Protocol, cast

# Python 3.10's typing.Concatenate rejects the ellipsis used in our aliases.
from typing_extensions import Concatenate, Self, TypeVar  # noqa: UP035

if TYPE_CHECKING:
    from collections.abc import Sequence


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
            loss (float | None): Scalar loss for this evaluation. On success,
                the raw objective result is stored here before post-evaluation
                hooks run. A post-hook may replace it; the final value is the
                authoritative objective result returned to the caller.
            meta (dict[str, Any]): Free-form metadata dictionary.
                Implementations may add diagnostic information here as needed.
            temp (SimpleNamespace): Scratch space for temporary values
                during evaluation. Nothing stored here is part of the
                public API. It is omitted from the ``to_summary()`` method.
            config:
                Child-local evaluation configuration for this context.
            shared:
                Shared state or resources reused across related contexts.
            children (tuple[EvaluateContext, ...]): Child contexts in creation
                order. The tuple cannot be used to modify the topology.
            parents (tuple[EvaluateContext, ...]): Parent contexts. Current
                evaluation structures are trees, but this tuple permits future
                contexts to have multiple parents.

        """

        self._set_defaults(config, shared)

    def to_summary(self, recursive: bool = False) -> dict[str, Any]:
        """
        Return a dictionary summarizing the evaluation state.

        Args:
            recursive: If ``True``, include recursive summaries of child
                contexts in creation order under ``"children"``. If ``False``,
                summarize only this context.

        Returns:
            dict[str, Any]: A dictionary containing the fields
            ``quantities``, ``parameters``, ``loss``, and ``meta``. Recursive
            summaries additionally contain ``children``.

        """
        summary = {
            "quantities": self.quantities,
            "parameters": self.parameters,
            "loss": self.loss,
            "meta": self.meta,
        }
        if recursive:
            summary["children"] = [
                child.to_summary(recursive=True) for child in self._children
            ]
        return summary

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
        self._parents: list[EvaluateContext] = []
        self._root = self

    @property
    def children(self) -> tuple[EvaluateContext, ...]:
        """Return child contexts in creation order."""

        return tuple(self._children)

    @property
    def parents(self) -> tuple[EvaluateContext, ...]:
        """Return the contexts that are parents of this context."""

        return tuple(self._parents)

    def _clear_dependencies(self) -> None:
        # first we remove `self` from the parent list of the
        # current children
        # strictly speaking this may be unnecessary, but we do it
        # for consistency
        for child in self._children:
            child._parents = [  # noqa: SLF001
                parent
                for parent in child._parents  # noqa: SLF001
                if parent is not self
            ]

        # then clear the list
        self._children.clear()

    def _set_root_recursive(self, root: EvaluateContext) -> None:
        pending: list[EvaluateContext] = [self]
        visited: set[int] = set()

        while pending:
            ctx = pending.pop()

            if id(ctx) in visited:
                continue

            visited.add(id(ctx))
            ctx._root = root  # noqa: SLF001
            pending.extend(ctx._children)  # noqa: SLF001

    def _add_dependency(self, child: EvaluateContext) -> None:
        if any(existing is child for existing in self._children):
            return

        self._children.append(child)
        child._parents.append(self)
        child._set_root_recursive(self._root)

    def spawn_children(self, n_children: int) -> list[EvaluateContext]:
        """
        Create child contexts linked to this context.

        Each child receives a deep copy of ``config``, while sharing the
        same ``shared`` namespace as the parent.

        An ``EvaluateContext`` is intended to manage at most one batch of
        child contexts per evaluation. Calling ``spawn_children()`` again on
        the same context replaces the previous child batch.

        In many cases, ``child_contexts()`` is the preferred interface for a
        scoped nested evaluation.

        Args:
            n_children: Number of child contexts to create.
            configurator:
                Optional configurator applied once to each spawned child
                context immediately after creation.

        Returns:
            The newly created child contexts.

        """

        self._clear_dependencies()

        children = [
            EvaluateContext(
                config=copy.deepcopy(self._root.config), shared=self._root.shared
            )
            for _ in range(n_children)
        ]
        for child in children:
            self._add_dependency(child)

        return self._children

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
            synchronization. Descendant result states are represented without
            serializing context objects or reverse parent references. Shared
            resources, configuration, and temporary state are omitted.

        """

        return {
            "parameters": self.parameters,
            "loss": self.loss,
            "quantities": self.quantities,
            "meta": self.meta,
            "children": [child.to_result_state() for child in self._children],
        }

    def apply_result_state(self, state: dict[str, Any]):
        """
        Apply result-bearing state from another context.

        Args:
            state:
                State previously produced by ``to_result_state()``.

        Side Effects:
            Updates ``parameters``, ``loss``, ``quantities``, and ``meta``
            on this context, and restores result state on its descendants.

        """

        self.parameters = state["parameters"]
        self.loss = state["loss"]
        self.quantities = state["quantities"]
        self.meta = state["meta"]

        child_states = state.get("children", [])
        if len(self._children) != len(child_states):
            self.spawn_children(len(child_states))
        for child, child_state in zip(self._children, child_states, strict=True):
            child.apply_result_state(child_state)


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
    bound=dict[str, Any],
    covariant=True,
    default=dict[str, Any],
)

LossQuantitiesT = TypeVar(
    "LossQuantitiesT", bound=dict[str, Any], default=dict[str, Any]
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


EvaluationHook = PreEvaluationHook | PostEvaluationHook


InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)


class NoOutput: ...


NO_OUTPUT = NoOutput()


class ObjectiveComponent(Generic[InputT_contra, OutputT_co]):
    def __init__(self) -> None:
        """Initialize a generic objective component."""

        self.pre_eval_hooks: list[Callable[[EvaluateContext], None]] = []
        self.post_eval_hooks: list[Callable[[EvaluateContext], None]] = []
        self._ctx_configurator: (
            Callable[Concatenate[EvaluateContext, ...], None] | None
        ) = None
        self._ctx_configurator_args: tuple[Any, ...] = ()
        self._ctx_configurator_kwargs: dict[str, Any] = {}
        self.static_meta_data: dict[str, Any] = {}

    def copy(self) -> Self:
        new = copy.copy(self)
        new.pre_eval_hooks = self.pre_eval_hooks.copy()
        new.post_eval_hooks = self.post_eval_hooks.copy()
        new._ctx_configurator_kwargs = (  # noqa: SLF001
            self._ctx_configurator_kwargs.copy()
        )
        new.static_meta_data = self.static_meta_data.copy()
        return new

    def _create_context(self) -> EvaluateContext:
        """Create the default context used when none is supplied."""
        return EvaluateContext()

    def with_meta(self, /, **meta: Any) -> Self:
        """
        Return a copy with additional static evaluation metadata.

        Existing metadata is preserved unless a key is supplied again. The
        source objective is unchanged. Existing hook objects are retained, but
        the returned objective has independent registration lists, so later
        hook registration does not affect the source.
        """

        new = self.copy()
        new.static_meta_data = {**self.static_meta_data, **meta}

        return new

    def with_ctx_configurator(
        self,
        configurator: Callable[Concatenate[EvaluateContext, ...], None],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        new = self.copy()
        new._ctx_configurator = configurator  # noqa: SLF001
        new._ctx_configurator_args = args  # noqa: SLF001
        new._ctx_configurator_kwargs = kwargs.copy()  # noqa: SLF001
        return new

    def _apply_ctx_configurator(self, ctx: EvaluateContext) -> None:
        if self._ctx_configurator is None:
            return

        self._ctx_configurator(
            ctx, *self._ctx_configurator_args, **self._ctx_configurator_kwargs
        )

    def child_components(self) -> Sequence[ObjectiveComponent[Any, Any]]:
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
        Register evaluation callbacks on this component.

        Pass either a hook object implementing ``pre_eval`` and/or
        ``post_eval``, or pass the callbacks directly through ``pre`` and
        ``post``. At least one callback is required. Registered callbacks are
        invoked in registration order for their respective evaluation phase.

        Args:
            hook: Optional object implementing at least one of
                :class:`PreEvaluationHook` or :class:`PostEvaluationHook`.
                Cannot be combined with ``pre`` or ``post``.
            pre: Optional callback invoked before component evaluation.
            post: Optional callback invoked after component evaluation.
            recursive: Also register on descendant component in the call tree.
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
        pending: list[ObjectiveComponent[Any, Any]] = [self]
        while pending:
            component = pending.pop()

            if pre_eval is not None and pre_eval not in component.pre_eval_hooks:
                component.pre_eval_hooks.append(
                    cast("Callable[[EvaluateContext], None]", pre_eval)
                )
            if post_eval is not None and post_eval not in component.post_eval_hooks:
                component.post_eval_hooks.append(
                    cast("Callable[[EvaluateContext], None]", post_eval)
                )
            if recursive:
                pending.extend(reversed(component.child_components()))

        return self

    def _invoke_post_eval_hooks(
        self, ctx: EvaluateContext, evaluation_failed: bool
    ) -> None:
        """Invoke the post evaluation hooks."""
        post_hook_exceptions = []
        for cb in self.post_eval_hooks:
            try:
                cb(ctx)
            except Exception as e:  # noqa: PERF203
                post_hook_exceptions.append(e)

        if not evaluation_failed and post_hook_exceptions:
            raise PostEvalHookError(post_hook_exceptions)

    def _begin_evaluation(
        self,
        input_: InputT_contra,
        /,
        ctx: EvaluateContext,
    ) -> None:
        self._apply_ctx_configurator(ctx)
        for hook in self.pre_eval_hooks:
            hook(ctx)

    def _run(
        self,
        input_: InputT_contra,
        /,
        ctx: EvaluateContext,
    ) -> OutputT_co:
        raise NotImplementedError

    def _end_evaluation(
        self,
        output: OutputT_co | NoOutput,
        /,
        ctx: EvaluateContext,
        exception: BaseException | None,
    ) -> None:
        ctx.meta.update(self.static_meta_data)
        ctx.temp.exception = exception
        self._invoke_post_eval_hooks(
            ctx,
            evaluation_failed=exception is not None,
        )

    def __call__(
        self, input_: InputT_contra, /, ctx: EvaluateContext | None = None
    ) -> OutputT_co:
        if ctx is None:
            ctx = self._create_context()

        try:
            self._begin_evaluation(input_, ctx=ctx)
            result = self._run(input_, ctx)
        except BaseException as e:
            self._end_evaluation(NO_OUTPUT, ctx, e)
            raise
        else:
            self._end_evaluation(result, ctx, None)

        return result


class ObjectiveFunctor(ObjectiveComponent[ParametersT_contra, float]):
    def _begin_evaluation(
        self, parameters: ParametersT_contra, /, ctx: EvaluateContext
    ) -> None:
        ctx.parameters = parameters
        ctx.loss = None
        super()._begin_evaluation(parameters, ctx)

    def _end_evaluation(
        self,
        output: float | NoOutput,
        /,
        ctx: EvaluateContext,
        exception: BaseException | None,
    ) -> None:
        if isinstance(output, NoOutput):
            ctx.loss = None
        else:
            ctx.loss = output
        super()._end_evaluation(output, ctx, exception)


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


class QuantityComputer(ObjectiveComponent[ParametersT_contra, QuantitiesT_co]):
    def _begin_evaluation(
        self, parameters: ParametersT_contra, /, ctx: EvaluateContext
    ) -> None:
        ctx.parameters = parameters
        ctx.quantities = None
        super()._begin_evaluation(parameters, ctx)

    def _end_evaluation(
        self,
        output: QuantitiesT_co | NoOutput,
        /,
        ctx: EvaluateContext,
        exception: BaseException | None,
    ) -> None:
        if isinstance(output, NoOutput):
            ctx.quantities = None
        else:
            ctx.quantities = output
        super()._end_evaluation(output, ctx, exception)

    def _compute(
        self, parameters: ParametersT_contra, /, ctx: EvaluateContext
    ) -> QuantitiesT_co:
        raise NotImplementedError

    def _run(
        self, parameters: ParametersT_contra, /, ctx: EvaluateContext
    ) -> QuantitiesT_co:
        return self._compute(parameters, ctx)

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

    def _run(self, parameters: ParametersT_contra, ctx: EvaluateContext) -> float:
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
