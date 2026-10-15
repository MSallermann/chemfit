from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeAlias, cast, overload

# Python 3.10's typing.Concatenate rejects the ellipsis used in our aliases.
from typing_extensions import Concatenate, Self, TypeVar  # noqa: UP035

from chemfit.evaluate_context import NO_OUTPUT, EvaluateContext, NoOutput


class PreEvaluationHook(Protocol):
    """Callback that runs before a component evaluation."""

    def pre_eval(self, ctx: EvaluateContext) -> None | Any:
        """Run before the component evaluation."""
        ...


class PostEvaluationHook(Protocol):
    """Callback that runs after a component evaluation."""

    def post_eval(self, ctx: EvaluateContext) -> None:
        """Run after the component evaluation."""
        ...


EvaluationHook = PreEvaluationHook | PostEvaluationHook


InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)


class Component(Generic[InputT_contra, OutputT_co]):
    """
    Base class for a context-aware computation from one input to one output.

    ``Component`` is ChemFit's general runtime abstraction. Despite
    the class name, an instance need not be a complete scalar objective: its
    role is determined by its input and output types. For example, a quantity
    computer maps fitting parameters to a quantity dictionary, while an
    optimizer-facing objective maps fitting parameters to a scalar loss.
    Pipelines, joins, and intermediate transforms are components as well.
    """

    def __init__(self) -> None:
        """Initialize a generic component."""

        self.pre_eval_hooks: list[Callable[[EvaluateContext], None | OutputT_co]] = []
        self.post_eval_hooks: list[Callable[[EvaluateContext], None]] = []
        self._context_configurator: (
            Callable[Concatenate[EvaluateContext, ...], None] | None
        ) = None
        self._context_configurator_args: tuple[Any, ...] = ()
        self._context_configurator_kwargs: dict[str, Any] = {}
        self.static_meta_data: dict[str, Any] = {}

    def copy(self) -> Self:
        new = copy.copy(self)
        new.pre_eval_hooks = self.pre_eval_hooks.copy()
        new.post_eval_hooks = self.post_eval_hooks.copy()
        new._context_configurator_kwargs = (  # noqa: SLF001
            self._context_configurator_kwargs.copy()
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

    @overload
    def with_loss(
        self: StructuralComponent[InputT_contra, OutputT_co],
        loss_function: Callable[Concatenate[OutputT_co, ...], float],
        /,
        **kwargs: Any,
    ) -> PipelineComponent[InputT_contra, float]: ...

    @overload
    def with_loss(
        self,
        loss_function: Callable[Concatenate[OutputT_co, ...], float],
        /,
        **kwargs: Any,
    ) -> OpaquePipeline[InputT_contra, float]: ...

    def with_loss(
        self,
        loss_function: Callable[Concatenate[OutputT_co, ...], float],
        /,
        **kwargs: Any,
    ) -> OpaquePipeline[InputT_contra, float] | PipelineComponent[InputT_contra, float]:
        """
        Return a new pipeline that applies a loss to this component's output.

        The source component is retained as the first stage and is not
        modified. Structural components produce a structural
        :class:`PipelineComponent`, preserving their graph-visible descendants.
        Other components produce an :class:`OpaquePipeline` and remain one
        schedulable unit with their loss. When the source is a quantity
        computer, the returned pipeline has the ``parameters -> float``
        contract of an objective term.

        Args:
            loss_function: Callable receiving this component's output followed
                by any arguments supplied through ``kwargs``.
            **kwargs: Keyword arguments bound to ``loss_function``.

        Returns:
            A new structural or opaque pipeline containing this component and
            the wrapped loss function.

        """
        loss_component = WrappedComponent(loss_function).bind(**kwargs)
        if isinstance(self, StructuralComponent):
            return PipelineComponent(self, loss_component)

        return OpaquePipeline(first=self, second=loss_component)

    def with_context_configurator(
        self,
        configurator: Callable[Concatenate[EvaluateContext, ...], None],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        new = self.copy()
        new._context_configurator = configurator  # noqa: SLF001
        new._context_configurator_args = args  # noqa: SLF001
        new._context_configurator_kwargs = kwargs.copy()  # noqa: SLF001
        return new

    def _apply_context_configurator(self, ctx: EvaluateContext) -> None:
        if self._context_configurator is None:
            return

        self._context_configurator(
            ctx, *self._context_configurator_args, **self._context_configurator_kwargs
        )

    def register_eval_hook(
        self,
        *,
        hook: EvaluationHook | None = None,
        pre: Callable[[EvaluateContext], None | OutputT_co] | None = None,
        post: Callable[[EvaluateContext], None] | None = None,
        recursive: bool = False,
    ) -> Self:
        """
        Register callbacks around this component's evaluation lifecycle.

        Pass either a hook object implementing ``pre_eval`` and/or
        ``post_eval``, or direct ``pre`` and ``post`` callbacks. With
        ``recursive=True``, registration traverses current structural
        descendants and modifies those component instances directly. A
        descendant shared with another composition therefore has the hook in
        that composition as well.

        Args:
            hook: Object providing ``pre_eval`` and/or ``post_eval`` methods.
            pre: Callback invoked before component execution. A non-``None``
                return value short-circuits the component body.
            post: Callback invoked after the outcome is stored in the context.
            recursive: Whether to register on graph-visible structural
                descendants as well as this component.

        Returns:
            This component, allowing chained registration.

        Raises:
            TypeError: If the supplied forms conflict or provide no callable
                hook methods.

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
        pending: list[Component[Any, Any]] = [self]
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

    def _invoke_post_eval_hooks(self, ctx: EvaluateContext) -> None:
        """Invoke the post evaluation hooks."""

        for cb in self.post_eval_hooks:
            try:
                cb(ctx)
            except Exception as e:  # noqa: PERF203
                ctx.output = e
                raise

    def _attach_context_metadata(self, ctx: EvaluateContext) -> None:
        """Attach component-specific metadata to an evaluation context."""

    def _begin_evaluation(
        self,
        input_: InputT_contra,
        /,
        ctx: EvaluateContext,
    ) -> None | OutputT_co:
        ctx._reset_evaluation_state()  # noqa: SLF001
        self._apply_context_configurator(ctx)
        ctx.input = input_

        self._attach_context_metadata(ctx)

        result: None | OutputT_co = None
        for hook in self.pre_eval_hooks:
            r = hook(ctx)
            if r is not None:
                if result is not None:
                    msg = "More than one pre-eval hooks returned a value."
                    raise ValueError(msg)
                result = r

        return result

    def _run(
        self,
        input_: InputT_contra,
        /,
        ctx: EvaluateContext,
    ) -> OutputT_co:
        raise NotImplementedError

    def _end_evaluation(
        self,
        outcome: OutputT_co | NoOutput | Exception,
        /,
        ctx: EvaluateContext,
    ) -> OutputT_co:
        ctx.meta.update(self.static_meta_data)
        self._attach_context_metadata(ctx)

        ctx.output = outcome
        self._invoke_post_eval_hooks(ctx)

        if isinstance(ctx.output, Exception):
            raise ctx.output

        return cast("OutputT_co", ctx.output)

    def __call__(
        self, input_: InputT_contra, /, ctx: EvaluateContext | None = None
    ) -> OutputT_co:
        if ctx is None:
            ctx = self._create_context()

        try:
            pre_eval_result = self._begin_evaluation(input_, ctx=ctx)
            if pre_eval_result is None:
                result = self._run(input_, ctx)
            else:
                result = pre_eval_result
        except Exception as e:
            result = e

        return self._end_evaluation(result, ctx)


MiddleT = TypeVar("MiddleT")
NextT = TypeVar("NextT")


class OpaquePipeline(Component[InputT_contra, OutputT_co]):
    def __init__(
        self,
        first: Component[InputT_contra, MiddleT],
        second: Component[MiddleT, OutputT_co],
    ) -> None:
        """Execute a sequence of components as one schedulable unit."""
        super().__init__()
        self._components: tuple[Component[Any, Any], ...] = (
            first,
            second,
        )

    def then(
        self,
        next_component: Component[OutputT_co, NextT],
    ) -> OpaquePipeline[InputT_contra, NextT]:
        new = self.copy()
        new._components = (*self._components, next_component)  # noqa: SLF001
        return cast("OpaquePipeline[InputT_contra, NextT]", new)

    def _run(self, input_: InputT_contra, /, ctx: EvaluateContext) -> OutputT_co:
        result: Any = input_

        for component in self._components:
            stage_ctx = ctx.spawn_ctx()
            ctx.children.append(stage_ctx)
            result = component(result, stage_ctx)

        return cast("OutputT_co", result)


class WrappedComponent(
    Component[InputT_contra, OutputT_co], Generic[InputT_contra, OutputT_co]
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
            func: Callable to wrap as a ``Component``. The callable may
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
            A new wrapped component with the requested arguments pre-applied.

        """
        new = super().copy()
        new.func_args = (*self.func_args, *args)
        new.func_kwargs = {**self.func_kwargs, **kwargs}

        return new

    def _run(self, input_: InputT_contra, /, ctx: EvaluateContext) -> OutputT_co:
        """
        Evaluate the wrapped callable as a component.

        If ``pass_ctx`` is ``True``, the wrapped callable receives both the
        parameter dictionary and the evaluation context. Otherwise, it
        receives only the parameter dictionary.

        Args:
            input_: Input value for the current evaluation.
            ctx: Evaluation context for the current call.

        Returns:
            Value returned by the wrapped callable.

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
    """Wrap a generic input-to-output callable as a component."""

    def wrap(
        func: Callable[Concatenate[InputT_contra, ...], OutputT_co],
    ) -> WrappedComponent[InputT_contra, OutputT_co]:
        return WrappedComponent(
            func,
            pass_ctx=pass_ctx,
        )

    return wrap


ComponentExpr = Component[InputT_contra, OutputT_co]


class ParentInput_:  # noqa: N801
    """Sentinel referring to the input of the enclosing structural component."""


PARENT_INPUT = ParentInput_()


@dataclass(frozen=True)
class Skipped:
    """
    Represent an evaluation that was not executed.

    Args:
        reason: Optional exception responsible for skipping the evaluation.
            None indicates that the evaluation was skipped without an error.

    """

    reason: Exception | None = None


SKIPPED = Skipped()

InputSource: TypeAlias = int | ParentInput_ | Skipped


class StructuralComponent(Component[InputT_contra, OutputT_co]):
    """
    Base class for components composed of other components.

    Structural components declare their children, their dependencies, and
    the sources of their inputs. They also control whether individual
    children execute and how their outcomes are combined.

    Child outcomes may be successful values, exceptions, or Skipped
    instances. A skipped child does not execute its evaluation lifecycle.

    The reference implementation in _run() evaluates children sequentially
    in topological order. Graph schedulers may evaluate independent children
    concurrently, but must preserve the same evaluation semantics.
    """

    @staticmethod
    def _describe_source(source: InputSource) -> dict[str, Any]:
        """Return a description of one declared input source."""
        if source is PARENT_INPUT:
            return {"kind": "parent_input"}
        if isinstance(source, Skipped):
            return {"kind": "skipped"}
        return {"kind": "child", "index": source}

    def _structure_metadata(self) -> dict[str, Any]:
        """Return a fresh description of this component's immediate structure."""
        return {
            "children": [
                {
                    "index": index,
                    "name": getattr(
                        getattr(component, "func", None),
                        "__name__",
                        type(component).__name__,
                    ),
                }
                for index, component in enumerate(self.components())
            ],
            "inputs": [
                [self._describe_source(source) for source in sources]
                for sources in self.component_inputs()
            ],
            "dependencies": [
                list(dependencies) for dependencies in self.component_dependencies()
            ],
            "outputs": [
                self._describe_source(source) for source in self.combine_inputs()
            ],
        }

    def _attach_context_metadata(self, ctx: EvaluateContext) -> None:
        """Attach this component's immediate structure to an evaluation context."""
        ctx.meta["chemfit.structure"] = self._structure_metadata()

    def components(
        self,
    ) -> Sequence[ComponentExpr[Any, Any]]:
        """
        Return the immediate child components in topological order.

        Dependencies must precede their consumers. The returned order
        determines the indices used by the other structural methods.
        """
        raise NotImplementedError

    def component_dependencies(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        """
        Return the dependencies of each child component.

        Each entry contains the indices of the children that must complete
        before the corresponding child can be considered for execution.

        An empty tuple indicates that the child has no dependencies and
        may execute immediately.

        Dependencies determine execution order and concurrency. They do
        not determine whether a child should be skipped.
        """
        raise NotImplementedError

    def component_inputs(self) -> tuple[tuple[InputSource, ...], ...]:
        """
        Declare the input sources for each child component.

        Each outer tuple corresponds to a child in components().

        Input sources may be:

        - PARENT_INPUT: The input of this structural component.
        - int: The outcome of the child at that index.
        - Skipped: An explicitly skipped input.

        Referenced child outcomes must be available before the consuming
        child is evaluated.
        """
        raise NotImplementedError

    def resolve_child_input(
        self,
        index: int,  # noqa: ARG002
        inputs: tuple[Any, ...],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> Any | Skipped:
        """
        Resolve the input of a child or decide to skip its evaluation.

        Called after the child's declared dependencies have completed
        and its input sources have been assembled.

        Args:
            index: Index of the child in components().
            inputs: Values obtained from component_inputs(). These may
                include successful results, exceptions, or Skipped instances.
            ctx: Evaluation context of the enclosing structural component.

        Returns:
            The input to pass to the child, or a Skipped instance to
            prevent its evaluation.

        Notes:
            By default, a child is skipped if any input is an exception or
            Skipped instance. Existing skip reasons are propagated, and
            exceptions are preserved as skip reasons.

            Subclasses may override this method to implement different
            execution policies, including conditional execution or recovery
            from failed dependencies.

        """
        for value in inputs:
            if isinstance(value, Skipped):
                return value

            if isinstance(value, Exception):
                return Skipped(reason=value)

        # Preserve the usual convention that a single input is unpacked,
        # while multiple inputs are passed as a tuple.
        return inputs[0] if len(inputs) == 1 else inputs

    def combine_inputs(self) -> tuple[InputSource, ...]:
        """
        Declare the input sources passed to combine_outcomes().

        By default, all immediate child outcomes are passed in component
        order.

        Subclasses may select a subset of child outcomes, the parent
        input, or explicitly skipped values.
        """
        return tuple(range(len(self.components())))

    def combine_outcomes(
        self,
        outcomes: tuple[Any, ...],
        ctx: EvaluateContext,
    ) -> OutputT_co:
        """
        Combine the selected outcomes into the structural component's result.

        Called after the required child evaluations have completed or
        been skipped.

        Args:
            outcomes: Values selected by combine_inputs(). These may
                contain successful results, exceptions, or Skipped instances.
            ctx: Evaluation context of this structural component.

        Returns:
            The combined result.

        Raises:
            Exception: If the available outcomes cannot produce a valid
                result.

        Notes:
            The structural component is responsible for interpreting skipped
            outcomes. Skipping does not necessarily indicate a failure.

        """
        raise NotImplementedError

    def _spawn_child_contexts(self, ctx: EvaluateContext) -> list[EvaluateContext]:
        """Create and initialize one context per immediate child component."""
        child_ctxs = []
        for component in self.components():
            child_ctx = ctx.spawn_ctx()
            component._attach_context_metadata(child_ctx)  # noqa: SLF001
            child_ctxs.append(child_ctx)

        ctx.children.extend(child_ctxs)
        return child_ctxs

    def _run(
        self,
        input_: InputT_contra,
        /,
        ctx: EvaluateContext,
    ) -> OutputT_co:
        """
        Evaluate the structural component using reference call semantics.

        Children are considered sequentially in topological order.
        Independent children may be evaluated concurrently by graph
        schedulers implementing the same structural interface.

        Each child either produces a value, raises an exception, or is
        skipped. Exceptions from child evaluation or input resolution
        are recorded as child outcomes.

        Skipped children do not invoke their evaluation lifecycle and
        retain their initial context state.
        """
        components = self.components()
        input_specs = self.component_inputs()

        # Create child contexts before evaluating any children.
        # This preserves the context structure even for skipped children.
        child_ctxs = self._spawn_child_contexts(ctx)

        # Store one outcome per child, including skipped evaluations.
        outcomes: list[Any] = [NO_OUTPUT] * len(components)

        def resolve_source(source: InputSource) -> Any:
            """Resolve an input source against completed child outcomes."""
            if source is PARENT_INPUT:
                return input_

            if isinstance(source, Skipped):
                return source

            return outcomes[cast("int", source)]

        # Children are in topological order, so all declared input
        # sources must already be available when a child is considered.
        for idx, (comp, child_ctx, sources) in enumerate(
            zip(components, child_ctxs, input_specs, strict=True)
        ):
            inputs = tuple(resolve_source(source) for source in sources)

            try:
                child_input = self.resolve_child_input(idx, inputs, ctx)
            except Exception as e:
                child_input = Skipped(reason=e)

            if isinstance(child_input, Skipped):
                outcomes[idx] = child_input
                continue

            try:
                outcomes[idx] = comp(child_input, child_ctx)
            except Exception as e:
                outcomes[idx] = e

        # Only pass the sources explicitly selected by combine_inputs().
        # Skipped outcomes retain their optional reasons.
        combine_inputs = tuple(
            resolve_source(source) for source in self.combine_inputs()
        )

        return self.combine_outcomes(combine_inputs, ctx)


class JoinedComponent(StructuralComponent[InputT_contra, OutputT_co]):
    """Base class for independently evaluated child components."""

    def __init__(
        self,
        components: tuple[Component[InputT_contra, Any], ...],
    ) -> None:
        """Initialize a nonempty collection of independent components."""
        super().__init__()

        if not components:
            msg = "Need at least one component to construct a joined Component"
            raise ValueError(msg)

        self._components = components

    def n_components(self) -> int:
        """Return the number of child components."""
        return len(self._components)

    def components(self) -> Sequence[Component[Any, Any]]:
        """Return the independently evaluated components."""
        return self._components

    def component_dependencies(self) -> tuple[tuple[int, ...], ...]:
        """All children are independent and may execute concurrently."""
        return tuple(() for _ in self._components)

    def component_inputs(self) -> tuple[tuple[InputSource, ...], ...]:
        """Every child receives the parent component's input."""
        return tuple((PARENT_INPUT,) for _ in self._components)

    def combine_outcomes(
        self,
        outcomes: tuple[Any, ...],
        ctx: EvaluateContext,
    ) -> OutputT_co:
        """
        Combine the independently evaluated child outcomes.

        Outcomes may contain successful values, exceptions, or Skipped
        instances. Subclasses define how these are combined.
        """
        raise NotImplementedError


class TupleJoinedComponent(JoinedComponent[InputT_contra, tuple[Any, ...]]):
    """Evaluate independent components and collect their outcomes in a tuple."""

    def combine_outcomes(
        self,
        outcomes: tuple[Any, ...],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> tuple[Any, ...]:
        """
        Return the child outcomes in their declared order.

        Exceptions and skipped evaluations are preserved as entries
        rather than raised or discarded.
        """
        return outcomes


class PipelineComponent(StructuralComponent[InputT_contra, OutputT_co]):
    """Evaluate components sequentially, passing each result to the next."""

    def __init__(
        self,
        first: Component[InputT_contra, MiddleT],
        second: Component[MiddleT, OutputT_co],
    ) -> None:
        """Initialize a pipeline containing two components."""
        super().__init__()
        self._components: list[Component[Any, Any]] = [first, second]

    def then(
        self,
        next_component: Component[OutputT_co, NextT],
    ) -> PipelineComponent[InputT_contra, NextT]:
        """Return a copy of the pipeline with an additional final component."""
        new = self.copy()
        new._components = [*self._components, next_component]  # noqa: SLF001
        return cast("PipelineComponent[InputT_contra, NextT]", new)

    def components(self) -> Sequence[Component[Any, Any]]:
        """Return the components in execution order."""
        return self._components

    def component_dependencies(self) -> tuple[tuple[int, ...], ...]:
        """Each component depends on its immediate predecessor."""
        return tuple((i - 1,) if i else () for i in range(len(self._components)))

    def component_inputs(self) -> tuple[tuple[InputSource, ...], ...]:
        """
        Pass the parent input to the first component.

        Subsequent components receive their predecessor's outcome.
        """
        return (
            (PARENT_INPUT,),
            *((i - 1,) for i in range(1, len(self._components))),
        )

    def combine_inputs(self) -> tuple[InputSource, ...]:
        """Only the final component's outcome determines the pipeline result."""
        return (len(self._components) - 1,)

    def combine_outcomes(
        self,
        outcomes: tuple[Any, ...],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> OutputT_co:
        """
        Return the final component's outcome.

        If the final component was skipped because of an upstream
        exception, re-raise that original exception.

        An intentional skip without an exception cannot produce
        a pipeline result and therefore raises RuntimeError.
        """
        outcome = outcomes[0]

        if isinstance(outcome, Skipped):
            if outcome.reason is not None:
                raise outcome.reason

            msg = "Pipeline did not produce a result: final component was skipped."
            raise RuntimeError(msg)

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

QuantityComputer = Component[ParametersT_contra, QuantitiesT_co]


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
