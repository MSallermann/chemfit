from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any


class NoOutput: ...


NO_OUTPUT = NoOutput()


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
            "input": self.input,
            "output": self.output,
            "meta": self.meta,
        }
        if recursive and self._incoming:
            summary["incoming"] = [
                incoming.to_summary(recursive=True) for incoming in self._incoming
            ]

        if recursive and self.pipeline:
            summary["pipeline"] = [
                stage.to_summary(recursive=True) for stage in self.pipeline
            ]
        return summary

    def _set_defaults(
        self, config: SimpleNamespace | None, shared: SimpleNamespace | None
    ):
        self.input: Any = None
        self.output: Any | NoOutput = NO_OUTPUT
        self.temp = SimpleNamespace()
        self.config = SimpleNamespace() if config is None else config
        self.shared = SimpleNamespace() if shared is None else shared
        self.meta: dict[str, Any] = {}
        self.pipeline: list[EvaluateContext] = []
        self._incoming: list[EvaluateContext] = []
        self._outgoing: list[EvaluateContext] = []
        self.exception: BaseException | None = None

    def _reset_evaluation_state(self) -> None:
        """Clear per-call results while preserving configuration and shared state."""
        self._clear_incoming()
        self.input = None
        self.output = NO_OUTPUT
        self.temp = SimpleNamespace()
        self.meta = {}
        self.pipeline = []
        self.exception = None

    def spawn_ctx(self) -> EvaluateContext:
        return EvaluateContext(config=copy.deepcopy(self.config), shared=self.shared)

    @property
    def incoming(self) -> tuple[EvaluateContext, ...]:
        """Return incoming contexts."""

        return tuple(self._incoming)

    @property
    def outgoing(self) -> tuple[EvaluateContext, ...]:
        """Return the outgoing contexts contexts."""

        return tuple(self._outgoing)

    def _clear_incoming(self) -> None:
        for ctx in self._incoming:
            ctx._outgoing.remove(self)  # noqa: SLF001
        self._incoming.clear()

    def add_incoming(self, ctx_incoming: EvaluateContext) -> None:
        if ctx_incoming is self:
            msg = "A context cannot depend on itself."
            raise ValueError(msg)
        if any(existing is ctx_incoming for existing in self._incoming):
            return
        self._incoming.append(ctx_incoming)
        ctx_incoming._outgoing.append(self)

    def __getstate__(self) -> dict[str, Any]:
        """
        Return the worker-input state for this context.

        Returns:
            Dictionary containing the context state needed to initialize the
            context in a worker or other execution environment.

        """
        return {
            "input": self.input,
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
        self.input = state["input"]
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
            "input": self.input,
            "output": self.output,
            "meta": self.meta,
            "exception": self.exception,
            "pipeline": [p.to_result_state() for p in self.pipeline],
        }

    def apply_result_state(self, state: dict[str, Any]):
        """
        Apply result-bearing state from another context.

        Args:
            state:
                State previously produced by ``to_result_state()``.

        Side Effects:
            Updates input, output, exception, and metadata on this context and
            restores result state on opaque-pipeline descendants.

        """

        self.input = state["input"]
        self.output = state["output"]
        self.meta = state["meta"]
        self.exception = state["exception"]
        pipeline_states = state["pipeline"]

        if len(self.pipeline) != len(pipeline_states):
            self.pipeline = [self.spawn_ctx() for _ in pipeline_states]

        for self_p, res_p in zip(self.pipeline, pipeline_states, strict=True):
            self_p.apply_result_state(res_p)
