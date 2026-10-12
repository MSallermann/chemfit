from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeAlias, cast

# Python 3.10's typing.Concatenate rejects the ellipsis used in our aliases.
from typing_extensions import Concatenate, Self, TypeVar  # noqa: UP035

from chemfit.evaluate_context import NO_OUTPUT, EvaluateContext, NoOutput


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

    def with_loss(
        self,
        loss_function: Callable[Concatenate[OutputT_co, ...], float],
        /,
        **kwargs: Any,
    ) -> OpaquePipeline[InputT_contra, float]:
        return OpaquePipeline(
            first=self,
            second=WrappedComponent(loss_function).bind(**kwargs),
        )

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

    def register_eval_hook(
        self,
        *,
        hook: EvaluationHook | None = None,
        pre: Callable[[EvaluateContext], None] | None = None,
        post: Callable[[EvaluateContext], None] | None = None,
        recursive: bool = False,
    ) -> Self:
        """TODO."""
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
            if recursive and isinstance(component, StructuralComponent):
                pending.extend(reversed(component.components()))

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
        ctx._reset_evaluation_state()  # noqa: SLF001
        self._apply_ctx_configurator(ctx)
        ctx.input = input_
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
        outcome: OutputT_co | NoOutput | BaseException,
        /,
        ctx: EvaluateContext,
    ) -> None:
        ctx.meta.update(self.static_meta_data)

        if isinstance(outcome, BaseException):
            ctx.output = NO_OUTPUT
            ctx.exception = outcome
            evaluation_failed = True
        else:
            ctx.output = outcome
            ctx.exception = None
            evaluation_failed = False

        self._invoke_post_eval_hooks(
            ctx,
            evaluation_failed=evaluation_failed,
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
            self._end_evaluation(e, ctx)
            raise
        else:
            self._end_evaluation(result, ctx)

        return result


MiddleT = TypeVar("MiddleT")
NextT = TypeVar("NextT")


class OpaquePipeline(ObjectiveComponent[InputT_contra, OutputT_co]):
    def __init__(
        self,
        first: ObjectiveComponent[InputT_contra, MiddleT],
        second: ObjectiveComponent[MiddleT, OutputT_co],
    ) -> None:
        """Execute a sequence of components as one schedulable unit."""
        super().__init__()
        self._components: tuple[ObjectiveComponent[Any, Any], ...] = (
            first,
            second,
        )

    def then(
        self,
        next_component: ObjectiveComponent[OutputT_co, NextT],
    ) -> OpaquePipeline[InputT_contra, NextT]:
        new = self.copy()
        new._components = (*self._components, next_component)  # noqa: SLF001
        return cast("OpaquePipeline[InputT_contra, NextT]", new)

    def _run(self, input_: InputT_contra, /, ctx: EvaluateContext) -> OutputT_co:
        ctx.pipeline = []
        result: Any = input_

        for component in self._components:
            stage_ctx = ctx.spawn_ctx()
            ctx.pipeline.append(stage_ctx)
            result = component(result, stage_ctx)

        return cast("OutputT_co", result)


class WrappedComponent(
    ObjectiveComponent[InputT_contra, OutputT_co], Generic[InputT_contra, OutputT_co]
):
    def __init__(
        self,
        func: Callable[Concatenate[InputT_contra, ...], OutputT_co],
        pass_ctx: bool = False,
        func_args: tuple[Any, ...] | None = None,
        func_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize a wrapped component.

        Args:
            func: Callable to wrap as an ``ObjectiveComponent``. The callable may
                either accept only ``input_`` or accept both
                ``input_`` and ``ctx``.
            pass_ctx: If ``True``, call ``func(parameters, ctx)``. If
                ``False``, call ``func(parameters)``.

        """
        super().__init__()
        self.func = func
        self.pass_ctx = pass_ctx

        if func_args is None:
            self.func_args = ()
        else:
            self.func_args = func_args

        if func_kwargs is None:
            self.func_kwargs = {}
        else:
            self.func_kwargs = func_kwargs

    def bind(self, /, *args: Any, **kwargs: Any) -> Self:
        """
        Return a new component with extra arguments bound.

        The bound arguments are passed to the wrapped function in addition
        to the usual ChemFit arguments. Static metadata and registered hook
        objects are retained, while the returned functor has independent hook
        registration lists. Repeated calls append positional arguments and
        merge keyword arguments, with later keyword values taking precedence.

        Args:
            *args: Positional arguments to bind after ``parameters`` (and
                after ``ctx`` as well if ``pass_ctx=True``).
            **kwargs: Keyword arguments to bind.

        Returns:
            A new wrapped objective functor with the requested arguments
            pre-applied.

        """
        new = super().copy()
        new.func_args = (*self.func_args, *args)
        new.func_kwargs = {**self.func_kwargs, **kwargs}

        return new

    def _run(self, input_: InputT_contra, /, ctx: EvaluateContext) -> OutputT_co:
        """
        Evaluate the wrapped callable as an objective component.

        If ``pass_ctx`` is ``True``, the wrapped callable receives both the
        parameter dictionary and the evaluation context. Otherwise, it
        receives only the parameter dictionary.

        Args:
            parameters: Parameter dictionary for the current evaluation.
            ctx: Evaluation context for the current call.

        Returns:
            Scalar loss value returned by the wrapped callable.

        """

        if self.pass_ctx:
            loss = self.func(input_, *self.func_args, **self.func_kwargs, ctx=ctx)
        else:
            loss = self.func(input_, *self.func_args, **self.func_kwargs)

        return loss


def component(
    *,
    pass_ctx: bool = False,
) -> Callable[
    [Callable[Concatenate[InputT_contra, ...], OutputT_co]],
    WrappedComponent[InputT_contra, OutputT_co],
]:
    """Wrap a callable as an objective component."""

    def wrap(
        func: Callable[Concatenate[InputT_contra, ...], OutputT_co],
    ) -> WrappedComponent[InputT_contra, OutputT_co]:
        return WrappedComponent(
            func,
            pass_ctx=pass_ctx,
        )

    return wrap


ComponentExpr = ObjectiveComponent[InputT_contra, OutputT_co]


class ParentInput_: ...  # noqa: N801


PARENT_INPUT = ParentInput_()

InputSource: TypeAlias = int | ParentInput_


class StructuralComponent(ObjectiveComponent[InputT_contra, OutputT_co]):
    def components(
        self,
    ) -> Sequence[ComponentExpr[Any, Any]]:
        """
        Return the immediate child components in topological order.

        Dependencies must always precede their consumers. The returned order
        also determines the indices used by the other structural methods.
        """
        raise NotImplementedError

    def component_dependencies(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        """
        Return the dependencies of each child component.

        Each entry contains the indices of the children that must complete
        before the corresponding child can execute. An empty tuple indicates
        that the child has no dependencies and may execute immediately.

        The scheduler uses this information to determine execution order
        and which components can run concurrently.
        """
        raise NotImplementedError

    def component_inputs(self) -> tuple[tuple[InputSource, ...], ...]:
        """
        Declare the input sources for each component.

        Each outer tuple entry corresponds to a component in components().

        PARENT_INPUT means the input received by this structural component.
        An integer refers to the
        """
        raise NotImplementedError

    def combine_inputs(self) -> tuple[InputSource, ...]:
        """
        Input sources passed to combine_outcomes().

        Default simply passes all of them.
        """
        return tuple(range(len(self.components())))

    def combine_outcomes(
        self,
        outcomes: tuple[Any, ...],
        ctx: EvaluateContext,
    ) -> OutputT_co:
        """
        Combine child outcomes into the structural component's result.

        Called after the required child evaluations have completed.
        Outcomes may contain exceptions, allowing implementations to
        handle individual child failures.

        Args:
            outcomes: Child outcomes in the order of child_components().
            ctx: Evaluation context of this structural component.

        Returns:
            The combined result.

        Raises:
            Exception: If the child outcomes cannot produce a valid result.

        """
        raise NotImplementedError

    def _run(self, input_: InputT_contra, /, ctx: EvaluateContext) -> OutputT_co:
        components = self.components()
        input_specs = self.component_inputs()

        child_ctxs = [ctx.spawn_ctx() for _ in components]
        for child_ctx in child_ctxs:
            ctx.add_incoming(child_ctx)

        # Children are in topological order, so all dependencies
        # of a child have completed before it is evaluated.
        outcomes: list[Any] = [NO_OUTPUT] * len(components)

        for idx, (comp, child_ctx, sources) in enumerate(
            zip(components, child_ctxs, input_specs, strict=True)
        ):
            try:
                # Construct the input using completed dependency outcomes.
                inputs = [
                    input_ if source is PARENT_INPUT else outcomes[cast("int", source)]
                    for source in sources
                ]
                for value in inputs:
                    if isinstance(value, Exception):
                        raise value
                child_input = inputs[0] if len(inputs) == 1 else tuple(inputs)
                outcomes[idx] = comp(child_input, child_ctx)

            except Exception as e:  # noqa: PERF203
                outcomes[idx] = e

        combine_sources = self.combine_inputs()
        combine_inputs = tuple(
            input_ if source is PARENT_INPUT else outcomes[cast("int", source)]
            for source in combine_sources
        )
        return self.combine_outcomes(combine_inputs, ctx)


class JoinedComponent(StructuralComponent[InputT_contra, OutputT_co]):
    def __init__(self, components: tuple[ObjectiveComponent[InputT_contra, Any], ...]):
        """Initialize a nonempty collection of independent components."""
        super().__init__()

        self._components = components

        if not self._components:
            msg = "Need at least one component to construct a joined Component"
            raise ValueError(msg)

    def n_components(self):
        return len(self._components)

    def components(self) -> Sequence[ObjectiveComponent[Any, Any]]:
        """Return the independently evaluated components."""
        return self._components

    def component_dependencies(self) -> tuple[tuple[int, ...], ...]:
        """All components are independent."""
        return tuple(() for _ in self._components)

    def component_inputs(
        self,
    ) -> Any:
        """All components receive the parent input."""
        return tuple((PARENT_INPUT,) for _ in self.components())

    def combine_outcomes(
        self,
        outcomes: Sequence[Any | Exception | NoOutput],
        ctx: EvaluateContext,
    ) -> OutputT_co:
        """Simply return the tuple of outcomes."""
        raise NotImplementedError


class TupleJoinedComponent(JoinedComponent[InputT_contra, tuple[Any, ...]]):
    def combine_outcomes(
        self,
        outcomes: Sequence[Any],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> tuple[Any, ...]:
        return tuple(outcomes)


class PipelineComponent(StructuralComponent[InputT_contra, OutputT_co]):
    def __init__(
        self,
        first: ObjectiveComponent[InputT_contra, MiddleT],
        second: ObjectiveComponent[MiddleT, OutputT_co],
    ) -> None:
        """Create a structural sequence of two components."""
        super().__init__()
        self._components: list[ObjectiveComponent[Any, Any]] = [first, second]

    def then(
        self,
        next_component: ObjectiveComponent[OutputT_co, NextT],
    ) -> PipelineComponent[InputT_contra, NextT]:
        new = self.copy()
        new._components = [*self._components, next_component]  # noqa: SLF001
        return cast("PipelineComponent[InputT_contra,NextT]", new)

    def components(self):
        return self._components

    def component_dependencies(self):
        return tuple((i - 1,) if i else () for i in range(len(self._components)))

    def component_inputs(self):
        return (
            (PARENT_INPUT,),
            *((i - 1,) for i in range(1, len(self._components))),
        )

    def combine_inputs(self) -> tuple[InputSource, ...]:
        return (len(self._components) - 1,)

    def combine_outcomes(
        self,
        outcomes: tuple[Any | Exception | NoOutput, ...],
        ctx: EvaluateContext,  # noqa: ARG002
    ):
        outcome = outcomes[-1]
        if isinstance(outcome, Exception):
            raise outcome
        return cast("OutputT_co", outcome)


# Variance here follows the direction in which values cross the public API:
#
# * Parameters are only consumed by objective components, so
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

QuantityComputer = ObjectiveComponent[ParametersT_contra, QuantitiesT_co]


def quantity(
    *,
    pass_ctx: bool = False,
) -> Callable[
    [Callable[Concatenate[ParametersT_contra, ...], QuantitiesT_co]],
    WrappedComponent[ParametersT_contra, QuantitiesT_co],
]:
    """Wrap a callable that returns a quantity mapping."""

    def wrap(
        func: Callable[Concatenate[ParametersT_contra, ...], QuantitiesT_co],
    ) -> WrappedComponent[ParametersT_contra, QuantitiesT_co]:
        return WrappedComponent(
            func,
            pass_ctx=pass_ctx,
        )

    return wrap


def objective(
    *,
    pass_ctx: bool = False,
) -> Callable[
    [Callable[Concatenate[ParametersT_contra, ...], float]],
    WrappedComponent[ParametersT_contra, float],
]:
    """Wrap a callable that returns a scalar objective value."""

    def wrap(
        func: Callable[Concatenate[ParametersT_contra, ...], float],
    ) -> WrappedComponent[ParametersT_contra, float]:
        return WrappedComponent(
            func,
            pass_ctx=pass_ctx,
        )

    return wrap
