from __future__ import annotations

import copy

# Keep Mapping available for runtime annotation introspection of the public API.
from collections.abc import Mapping  # noqa: TC003
from types import SimpleNamespace
from typing import Any


class NoOutput: ...


NO_OUTPUT = NoOutput()

# subclasses may not use any of these keys for their custom `record_sections``
RESERVED_RECORD_SECTION_KEYS = ("input", "output", "meta", "status", "children", "path")


class EvaluateContext:
    def __init__(
        self,
        config: SimpleNamespace | None = None,
        shared: SimpleNamespace | None = None,
    ):
        """
        Container for per-evaluation state.

        A new instance of `EvaluateContext` should generally be created for each
        component evaluation.
        Implementations write all per-call information into the context
        rather than storing it in the component instance.
        This makes evaluation easier to reason about and
        compatible with concurrent execution.

        Nested component invocations are recorded in ``children`` in
        declaration order. Contexts describe the evaluation hierarchy only;
        execution dependencies belong to the compiled call graph.

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
                Structural components reserve ``"chemfit.structure"`` for a
                description of their immediate structure.
            temp (SimpleNamespace): Scratch space for temporary values
                during evaluation. Nothing stored here is part of the
                public API. It is omitted from the ``to_summary()`` method.
            config:
                Child-local evaluation configuration for this context.
            shared:
                Shared state or resources reused across related contexts.
            children (list[EvaluateContext]): Ordered immediate child
                evaluation contexts.

        """

        self._set_defaults(config, shared)

    def to_summary(self, recursive: bool = False) -> dict[str, Any]:
        """
        Return a dictionary summarizing the evaluation state.

        Args:
            recursive: If ``True``, recursively summarize child contexts. If
                ``False``, summarize only this context.

        Returns:
            Dictionary containing ``input``, ``output``, and ``meta``.
            Recursive summaries additionally contain ``children``.

        """
        summary = {
            "input": self.input,
            "output": self.output,
            "meta": self.meta,
        }
        if recursive:
            summary["children"] = [
                child.to_summary(recursive=True) for child in self.children
            ]
        return summary

    def to_record(self, *, strict: bool = False) -> dict[str, Any]:
        """
        Return a versioned, JSON-compatible record of this evaluation.

        The record contains the complete child hierarchy and can be passed
        directly to ``json.dumps(..., allow_nan=False)``. Configuration,
        shared resources, and temporary state are intentionally omitted.

        Args:
            strict: Raise a descriptive serialization error for unsupported
                values. If ``False``, replace them with explicit tagged
                representations.

        Returns:
            Self-contained record intended for persistent inspection and
            analysis. Records do not reconstruct executable contexts.

        """
        # The serializer imports EvaluateContext for traversal, so defer this
        # reverse dependency until the method is called.
        from chemfit.context_serialization import context_to_record  # noqa: PLC0415

        return context_to_record(self, strict=strict)

    def record_sections(self) -> Mapping[str, Any]:
        """
        Return additional named sections for persistent serialization.

        Context subclasses should override this method when they own state
        that belongs in ``to_record()``. Each mapping key becomes a top-level
        section on that context node. Section values pass through the same
        JSON-compatible conversion and cycle detection as the core context
        fields, without requiring the serializer to know the subclass.

        Overrides should preserve sections returned by ``super()`` so multiple
        levels of context specialization compose cleanly.

        NOTE: None of the keys in RESERVED_RECORD_SECTIONS may be used.
        """
        return {}

    def _set_defaults(
        self, config: SimpleNamespace | None, shared: SimpleNamespace | None
    ):
        self.input: Any = None
        self.output: Any | NoOutput = NO_OUTPUT
        self.temp = SimpleNamespace()
        self.config = SimpleNamespace() if config is None else config
        self.shared = SimpleNamespace() if shared is None else shared
        self.meta: dict[str, Any] = {}
        self.children: list[EvaluateContext] = []

    def _reset_evaluation_state(self) -> None:
        """Clear per-call results while preserving configuration and shared state."""
        self.input = None
        self.output = NO_OUTPUT
        self.temp = SimpleNamespace()
        self.meta = {}
        self.children = []

    def spawn_ctx(self) -> EvaluateContext:
        """Create a child-compatible context with copied configuration."""
        return EvaluateContext(config=copy.deepcopy(self.config), shared=self.shared)

    def __deepcopy__(self, memo: dict[int, Any]) -> EvaluateContext:
        """
        Copy the complete context state, including its evaluation tree.

        Worker transport intentionally uses :meth:`__getstate__`, which omits
        outputs and children. A normal deep copy has different semantics: it
        is an in-process snapshot, so it must retain every context attribute
        and recursively copy the child hierarchy.
        """

        duplicate = type(self).__new__(type(self))
        memo[id(self)] = duplicate
        for name, value in vars(self).items():
            setattr(duplicate, name, copy.deepcopy(value, memo))
        return duplicate

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
            "children": [child.to_result_state() for child in self.children],
        }

    def apply_result_state(self, state: dict[str, Any]):
        """
        Apply result-bearing state from another context.

        Args:
            state:
                State previously produced by ``to_result_state()``.

        Side Effects:
            Updates input, output, and metadata on this context and
            recursively restores result state on child contexts.

        """

        self.input = state["input"]
        self.output = state["output"]
        self.meta = state["meta"]
        child_states = state["children"]

        if len(self.children) != len(child_states):
            self.children = [self.spawn_ctx() for _ in child_states]

        for child, child_state in zip(self.children, child_states, strict=True):
            child.apply_result_state(child_state)
