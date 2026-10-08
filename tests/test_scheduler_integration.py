from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_scheduler import ExecutorTreeScheduler
from chemfit.scheduling import EvaluationRequest, Scheduler, SerialScheduler
from chemfit.tree_schedule import SerialTreeScheduler
from chemfit.wrap_funcs import quantity

Parameters = dict[str, float]
ContextState = dict[str, Any]
SchedulerFactory = Callable[[], Scheduler]
Outcome = float | tuple[str, str]


def context_state(ctx: EvaluateContext) -> ContextState:
    """Snapshot one context and its descendants for backend comparisons."""

    return ctx.to_summary(recursive=True)


class RecoverableLeafError(RuntimeError):
    """Failure converted to a replacement term by a parent objective."""


class OmittedLeafError(RuntimeError):
    """Failure whose term is omitted by a parent objective."""


class FatalLeafError(RuntimeError):
    """Failure propagated through one combined objective to its parent."""


class SetupPhaseError(RuntimeError):
    """Failure raised by a pre-evaluation hook."""


class ConfigurationSetupError(RuntimeError):
    """Failure raised while child contexts are configured."""


class ReductionPhaseError(RuntimeError):
    """Failure raised by a custom aggregator."""


class PostEvaluationError(RuntimeError):
    """Failure raised by a post-evaluation hook."""


class LifecycleHook:
    """Record deterministic pre/post hook observations in each context."""

    def pre_eval(self, ctx: EvaluateContext) -> None:
        assert ctx.parameters is not None
        ctx.meta["lifecycle"] = {
            "pre_parameters": dict(ctx.parameters),
            "pre_loss": ctx.loss,
            "config_depth": ctx.config.depth,
        }

    def post_eval(self, ctx: EvaluateContext) -> None:
        exception = ctx.temp.exception
        ctx.meta["lifecycle"].update(
            post_loss=ctx.loss,
            post_exception=None if exception is None else type(exception).__name__,
        )


class ConditionalSetupFailureHook:
    """Fail a selected setup phase before an objective evaluates."""

    def __init__(self, trigger: float, name: str) -> None:
        """Initialize the hook with its parameter trigger and objective name."""

        self.trigger = trigger
        self.name = name

    def pre_eval(self, ctx: EvaluateContext) -> None:
        assert ctx.parameters is not None
        if ctx.parameters.get("setup_failure") == self.trigger:
            msg = f"setup failed in {self.name}"
            raise SetupPhaseError(msg)


class ConditionalPostFailureHook:
    """Fail a selected post-evaluation phase after reduction succeeds."""

    def post_eval(self, ctx: EvaluateContext) -> None:
        assert ctx.parameters is not None
        if ctx.parameters.get("evaluation_failure") == 2.0:
            msg = "post-evaluation hook failed in level_three"
            raise PostEvaluationError(msg)


class RecordingExceptionHandler:
    """Record and then replace, omit, or propagate a child failure."""

    def __init__(self, name: str, action: str, replacement: float = 0.0) -> None:
        """Initialize the handler action and optional replacement value."""

        self.name = name
        self.action = action
        self.replacement = replacement

    def __call__(
        self,
        exception: Exception,
        ctx: EvaluateContext,
        idx: int,
    ) -> float | None:
        ctx.meta["handled_exception"] = {
            "handler": self.name,
            "action": self.action,
            "type": type(exception).__name__,
            "index": idx,
        }

        if self.action == "replace":
            return self.replacement + idx
        if self.action == "omit":
            return None
        if self.action == "raise":
            raise exception

        msg = f"Unknown exception-handler action: {self.action}"
        raise ValueError(msg)


class RecordingAggregator:
    """Reduce weighted terms while recording all aggregator inputs."""

    def __init__(self, name: str, offset: float) -> None:
        """Initialize the aggregator name and additive offset."""

        self.name = name
        self.offset = offset

    def __call__(
        self,
        terms: list[float],
        quantities: list[dict[str, Any] | None],
        ctx: EvaluateContext,
    ) -> float:
        quantity_labels = [
            None if quantities_for_term is None else quantities_for_term["label"]
            for quantities_for_term in quantities
        ]
        ctx.meta["reduction"] = {
            "name": self.name,
            "terms": list(terms),
            "quantity_labels": quantity_labels,
            "config_depth": ctx.config.depth,
        }

        assert ctx.parameters is not None
        failure_mode = ctx.parameters.get("evaluation_failure")
        if (self.name == "level_two" and failure_mode == 1.0) or (
            self.name == "root" and failure_mode == 3.0
        ):
            msg = f"reduction failed in {self.name}"
            raise ReductionPhaseError(msg)

        quantity_adjustment = sum(
            0.01 * float(quantities_for_term["raw"])
            for quantities_for_term in quantities
            if quantities_for_term is not None
        )
        return sum((idx + 1) * term for idx, term in enumerate(terms)) + (
            self.offset + quantity_adjustment
        )


def configure_child_context(
    idx_child_ctx: int,
    child_ctx: EvaluateContext,
    num_children: int,
    parent_ctx: EvaluateContext,
) -> None:
    """Give every child deterministic inherited configuration and metadata."""

    child_ctx.config.depth = parent_ctx.config.depth + 1
    child_ctx.meta["configured_child"] = {
        "depth": child_ctx.config.depth,
        "index": idx_child_ctx,
        "siblings": num_children,
    }
    assert parent_ctx.parameters is not None
    if (
        parent_ctx.parameters.get("setup_failure") == 2.0
        and parent_ctx.config.depth == 2
        and idx_child_ctx == 1
    ):
        msg = "child configuration failed in level_two"
        raise ConfigurationSetupError(msg)


@quantity(pass_ctx=True)
def compute_leaf_quantities(
    parameters: Parameters,
    *,
    scale: float,
    offset: float,
    label: str,
    ctx: EvaluateContext,
) -> dict[str, Any]:
    """Compute deterministic quantities and leave worker-visible metadata."""

    raw = scale * parameters["x"] + offset + parameters["bias"]
    ctx.meta["leaf_label"] = label
    return {"label": label, "raw": raw, "scale": scale}


def leaf_loss(quantities: dict[str, Any], *, failure: str | None) -> float:
    """Compute a leaf loss or raise the requested test failure."""

    label = str(quantities["label"])
    if failure == "recoverable":
        raise RecoverableLeafError(label)
    if failure == "omitted":
        raise OmittedLeafError(label)
    if failure == "fatal":
        raise FatalLeafError(label)

    raw = float(quantities["raw"])
    return raw**2 + float(quantities["scale"])


def make_leaf(
    label: str,
    scale: float,
    offset: float,
    *,
    failure: str | None = None,
) -> ObjectiveFunctor[Parameters]:
    """Build one serializable quantity-computer objective leaf."""

    return compute_leaf_quantities.bind(
        scale=scale,
        offset=offset,
        label=label,
    ).with_loss(leaf_loss, failure=failure)


def make_combined(
    name: str,
    terms: list[ObjectiveFunctor[Parameters]],
    weights: list[float],
    *,
    exception_handler: RecordingExceptionHandler,
    offset: float,
) -> CombinedObjectiveFunction[Parameters]:
    """Build one configured level of the nested objective tree."""

    return CombinedObjectiveFunction(
        terms,
        weights=weights,
        child_context_configurator=configure_child_context,
        aggregator=RecordingAggregator(name, offset),
        exception_handler=exception_handler,
    )


def make_objective() -> CombinedObjectiveFunction[Parameters]:
    """Build the four-level objective shared by every scheduler test."""

    level_three = make_combined(
        "level_three",
        [
            make_leaf("deep_left", 0.75, 1.0),
            make_leaf("recoverable", 1.25, -2.0, failure="recoverable"),
            make_leaf("deep_right", -1.5, 0.5),
        ],
        [0.5, 1.5, 2.0],
        exception_handler=RecordingExceptionHandler(
            "level_three", "replace", replacement=-7.5
        ),
        offset=0.3,
    )

    level_two = make_combined(
        "level_two",
        [
            make_leaf("gamma", 2.0, -1.0),
            level_three,
            make_leaf("omitted", 0.2, 3.0, failure="omitted"),
        ],
        [1.25, 0.75, 2.5],
        exception_handler=RecordingExceptionHandler("level_two", "omit"),
        offset=-0.4,
    )

    level_one = make_combined(
        "level_one",
        [
            make_leaf("beta", -0.5, 2.0),
            level_two,
            make_leaf("delta", 1.1, -0.5),
        ],
        [2.0, 0.4, 1.1],
        exception_handler=RecordingExceptionHandler(
            "level_one", "replace", replacement=-3.0
        ),
        offset=1.2,
    )

    failing_branch = make_combined(
        "failing_branch",
        [
            make_leaf("branch_ok", -0.8, 4.0),
            make_leaf("fatal", 0.6, -1.0, failure="fatal"),
        ],
        [1.0, 3.0],
        exception_handler=RecordingExceptionHandler("failing_branch", "raise"),
        offset=2.0,
    )

    root = make_combined(
        "root",
        [
            make_leaf("alpha", 1.0, 0.0),
            level_one,
            failing_branch,
            make_leaf("omega", 1.4, 0.25),
        ],
        [0.25, 1.5, 0.8, 2.0],
        exception_handler=RecordingExceptionHandler(
            "root", "replace", replacement=11.0
        ),
        offset=-1.0,
    )
    root.register_eval_hook(hook=LifecycleHook(), recursive=True)
    level_three.register_eval_hook(hook=ConditionalSetupFailureHook(1.0, "level_three"))
    level_three.register_eval_hook(hook=ConditionalPostFailureHook())
    root.register_eval_hook(hook=ConditionalSetupFailureHook(3.0, "root"))
    return root


PARAMETER_BATCH = [
    {"x": 2.0, "bias": 0.5},
    {"x": -1.25, "bias": 1.0},
    {"x": 0.0, "bias": -0.75},
]

PHASE_FAILURE_BATCH = [
    {"x": 0.5, "bias": 0.25, "setup_failure": 1.0},
    {"x": 0.75, "bias": -0.5, "setup_failure": 2.0},
    {"x": 1.0, "bias": 0.0, "setup_failure": 3.0},
    {"x": 1.25, "bias": 0.5, "evaluation_failure": 1.0},
    {"x": 1.5, "bias": -0.25, "evaluation_failure": 2.0},
    {"x": 1.75, "bias": 0.75, "evaluation_failure": 3.0},
]

ALL_PARAMETERS = PARAMETER_BATCH + PHASE_FAILURE_BATCH


def evaluate_one(
    schedule: Any, parameters: Parameters, ctx: EvaluateContext
) -> Outcome:
    """Evaluate one request and normalize a raised exception as an outcome."""

    try:
        return schedule.evaluate(parameters, ctx)
    except Exception as exception:
        return type(exception).__name__, str(exception)


def evaluate_serial_reference() -> tuple[list[Outcome], list[ContextState]]:
    """Evaluate each request through the authoritative serial schedule."""

    objective = make_objective()
    contexts = [
        EvaluateContext(config=SimpleNamespace(depth=0)) for _ in ALL_PARAMETERS
    ]
    outcomes: list[Outcome] = []

    with SerialScheduler().prepare(objective) as schedule:
        for parameters, ctx in zip(ALL_PARAMETERS, contexts, strict=True):
            outcomes.append(evaluate_one(schedule, parameters, ctx))

    return outcomes, [context_state(ctx) for ctx in contexts]


def evaluate_batch(scheduler: Scheduler) -> tuple[list[Outcome], list[ContextState]]:
    """Evaluate one common batch and return outcomes plus context states."""

    objective = make_objective()
    contexts = [
        EvaluateContext(config=SimpleNamespace(depth=0)) for _ in ALL_PARAMETERS
    ]
    requests = [
        EvaluationRequest(parameters=parameters, ctx=ctx)
        for parameters, ctx in zip(ALL_PARAMETERS, contexts, strict=True)
    ]

    with scheduler.prepare(objective) as schedule:
        completed = sorted(
            schedule.evaluate_many(requests),
            key=lambda result: result.index,
        )

    outcomes: list[Outcome] = []
    for result in completed:
        if isinstance(result.value, Exception):
            outcomes.append((type(result.value).__name__, str(result.value)))
        else:
            outcomes.append(result.value)

    return outcomes, [context_state(ctx) for ctx in contexts]


def child_at(state: ContextState, *path: int) -> ContextState:
    """Return a descendant's serialized context state."""

    for child_idx in path:
        state = state["children"][child_idx]
    return state


def assert_lifecycle_tree(
    state: ContextState,
    parameters: Parameters,
    *,
    depth: int = 0,
) -> int:
    """Assert hook/configurator semantics recursively and count tree nodes."""

    if "lifecycle" not in state["meta"]:
        # A configurator can fail after spawn_children() has allocated the
        # complete child batch but before any child objective starts. The
        # serial reference retains those inactive contexts in its metadata.
        assert state["parameters"] is None
        assert state["loss"] is None
        assert state["quantities"] is None
        assert state["children"] == []
        return 1

    assert state["parameters"] == parameters
    lifecycle = state["meta"]["lifecycle"]
    assert lifecycle["pre_parameters"] == parameters
    assert lifecycle["pre_loss"] is None
    assert lifecycle["post_loss"] == state["loss"]
    assert lifecycle["config_depth"] == depth

    if depth > 0:
        assert state["meta"]["configured_child"]["depth"] == depth

    children = state["children"]
    return 1 + sum(
        assert_lifecycle_tree(child, parameters, depth=depth + 1) for child in children
    )


def assert_composed_semantics(
    outcomes: list[Outcome], states: list[ContextState]
) -> None:
    """Assert the nontrivial semantics represented by the serial oracle."""

    values = outcomes[: len(PARAMETER_BATCH)]
    normal_states = states[: len(PARAMETER_BATCH)]
    assert len(values) == len(PARAMETER_BATCH)
    assert all(isinstance(value, float) for value in values)
    assert len(set(values)) == len(values)

    for value, state, parameters in zip(
        values, normal_states, PARAMETER_BATCH, strict=True
    ):
        assert state["loss"] == value
        assert assert_lifecycle_tree(state, parameters) == 16

        root = state
        level_one = child_at(root, 1)
        level_two = child_at(root, 1, 1)
        level_three = child_at(root, 1, 1, 1)
        recoverable = child_at(root, 1, 1, 1, 1)
        omitted = child_at(root, 1, 1, 2)
        failing_branch = child_at(root, 2)
        fatal = child_at(root, 2, 1)

        assert level_three["meta"]["skipped_indices"] == []
        assert level_two["meta"]["skipped_indices"] == [2]
        assert level_one["meta"]["skipped_indices"] == []
        assert root["meta"]["skipped_indices"] == []

        assert level_three["meta"]["reduction"]["quantity_labels"] == [
            "deep_left",
            "recoverable",
            "deep_right",
        ]
        assert level_two["meta"]["reduction"]["quantity_labels"] == [
            "gamma",
            None,
        ]
        assert level_one["meta"]["reduction"]["quantity_labels"] == [
            "beta",
            None,
            "delta",
        ]
        assert root["meta"]["reduction"]["quantity_labels"] == [
            "alpha",
            None,
            None,
            "omega",
        ]

        assert recoverable["meta"]["handled_exception"] == {
            "handler": "level_three",
            "action": "replace",
            "type": "RecoverableLeafError",
            "index": 1,
        }
        assert omitted["meta"]["handled_exception"] == {
            "handler": "level_two",
            "action": "omit",
            "type": "OmittedLeafError",
            "index": 2,
        }
        assert fatal["meta"]["handled_exception"] == {
            "handler": "failing_branch",
            "action": "raise",
            "type": "FatalLeafError",
            "index": 1,
        }
        assert failing_branch["meta"]["handled_exception"] == {
            "handler": "root",
            "action": "replace",
            "type": "FatalLeafError",
            "index": 2,
        }

        assert recoverable["meta"]["lifecycle"]["post_exception"] == (
            "RecoverableLeafError"
        )
        assert omitted["meta"]["lifecycle"]["post_exception"] == ("OmittedLeafError")
        assert fatal["meta"]["lifecycle"]["post_exception"] == "FatalLeafError"
        assert failing_branch["meta"]["lifecycle"]["post_exception"] == (
            "FatalLeafError"
        )
        assert "reduction" not in failing_branch["meta"]


def assert_setup_failure_semantics(
    phase_outcomes: list[Outcome], phase_states: list[ContextState]
) -> None:
    """Assert setup failures follow serial lifecycle semantics."""

    pre_setup, configuration, root_setup = phase_states[:3]

    assert isinstance(phase_outcomes[0], float)
    assert assert_lifecycle_tree(pre_setup, PHASE_FAILURE_BATCH[0]) == 13
    failed_level_three = child_at(pre_setup, 1, 1, 1)
    assert failed_level_three["loss"] is None
    assert failed_level_three["meta"]["lifecycle"]["post_exception"] == (
        "SetupPhaseError"
    )
    assert failed_level_three["children"] == []
    assert failed_level_three["meta"]["handled_exception"] == {
        "handler": "level_two",
        "action": "omit",
        "type": "SetupPhaseError",
        "index": 1,
    }
    assert child_at(pre_setup, 1, 1)["meta"]["skipped_indices"] == [1, 2]

    assert isinstance(phase_outcomes[1], float)
    assert assert_lifecycle_tree(configuration, PHASE_FAILURE_BATCH[1]) == 13
    failed_level_two = child_at(configuration, 1, 1)
    failed_level_one = child_at(configuration, 1)
    assert failed_level_two["meta"]["lifecycle"]["post_exception"] == (
        "ConfigurationSetupError"
    )
    inactive_children = failed_level_two["children"]
    assert len(inactive_children) == 3
    assert all("lifecycle" not in child["meta"] for child in inactive_children)
    assert failed_level_two["meta"]["handled_exception"]["handler"] == "level_one"
    assert failed_level_two["meta"]["handled_exception"]["action"] == "replace"
    assert failed_level_one["meta"]["lifecycle"]["post_exception"] is None
    assert failed_level_one["meta"]["reduction"]["name"] == "level_one"

    assert phase_outcomes[2] == (
        "SetupPhaseError",
        "setup failed in root",
    )
    assert assert_lifecycle_tree(root_setup, PHASE_FAILURE_BATCH[2]) == 1
    assert root_setup["loss"] is None
    assert root_setup["meta"]["lifecycle"]["post_exception"] == "SetupPhaseError"
    assert root_setup["children"] == []


def assert_evaluation_failure_semantics(
    phase_outcomes: list[Outcome], phase_states: list[ContextState]
) -> None:
    """Assert evaluation failures follow serial lifecycle semantics."""

    reduction, post_hook, root_reduction = phase_states[3:]

    assert isinstance(phase_outcomes[3], float)
    assert assert_lifecycle_tree(reduction, PHASE_FAILURE_BATCH[3]) == 16
    failed_reduction = child_at(reduction, 1, 1)
    failed_reduction_parent = child_at(reduction, 1)
    assert failed_reduction["meta"]["lifecycle"]["post_exception"] == (
        "ReductionPhaseError"
    )
    assert failed_reduction["meta"]["reduction"]["name"] == "level_two"
    assert failed_reduction["meta"]["handled_exception"]["handler"] == ("level_one")
    assert failed_reduction["meta"]["handled_exception"]["action"] == "replace"
    assert failed_reduction_parent["meta"]["lifecycle"]["post_exception"] is None
    assert failed_reduction_parent["meta"]["reduction"]["name"] == "level_one"

    assert isinstance(phase_outcomes[4], float)
    assert assert_lifecycle_tree(post_hook, PHASE_FAILURE_BATCH[4]) == 16
    failed_post_hook = child_at(post_hook, 1, 1, 1)
    assert failed_post_hook["loss"] is not None
    assert failed_post_hook["meta"]["lifecycle"]["post_exception"] is None
    assert failed_post_hook["meta"]["handled_exception"] == {
        "handler": "level_two",
        "action": "omit",
        "type": "PostEvalHookError",
        "index": 1,
    }
    assert child_at(post_hook, 1, 1)["meta"]["skipped_indices"] == [1, 2]

    assert phase_outcomes[5] == (
        "ReductionPhaseError",
        "reduction failed in root",
    )
    assert assert_lifecycle_tree(root_reduction, PHASE_FAILURE_BATCH[5]) == 16
    assert root_reduction["loss"] is None
    assert root_reduction["meta"]["reduction"]["name"] == "root"
    assert root_reduction["meta"]["lifecycle"]["post_exception"] == (
        "ReductionPhaseError"
    )


def assert_phase_failure_semantics(
    outcomes: list[Outcome], states: list[ContextState]
) -> None:
    """Assert setup and evaluation failures follow serial lifecycle semantics."""

    phase_outcomes = outcomes[len(PARAMETER_BATCH) :]
    phase_states = states[len(PARAMETER_BATCH) :]
    assert len(phase_outcomes) == len(PHASE_FAILURE_BATCH)

    assert_setup_failure_semantics(phase_outcomes, phase_states)
    assert_evaluation_failure_semantics(phase_outcomes, phase_states)


def make_serial_tree_scheduler() -> Scheduler:
    return SerialTreeScheduler()


def make_thread_scheduler() -> Scheduler:
    return ExecutorTreeScheduler(
        executor_factory=partial(ThreadPoolExecutor, max_workers=4)
    )


def make_loky_scheduler() -> Scheduler:
    loky = pytest.importorskip("loky", reason="Missing loky")
    return ExecutorTreeScheduler(
        executor_factory=partial(loky.ProcessPoolExecutor, max_workers=3)
    )


@pytest.mark.parametrize(
    "scheduler_factory",
    [
        pytest.param(make_serial_tree_scheduler, id="serial-tree"),
        pytest.param(make_thread_scheduler, id="thread-pool"),
        pytest.param(make_loky_scheduler, id="loky-process-pool"),
    ],
)
def test_composed_scheduler_semantics_match_serial_reference(
    scheduler_factory: SchedulerFactory,
) -> None:
    expected_values, expected_states = evaluate_serial_reference()
    assert_composed_semantics(expected_values, expected_states)
    assert_phase_failure_semantics(expected_values, expected_states)

    values, states = evaluate_batch(scheduler_factory())

    assert values == expected_values
    assert states == expected_states


def test_mpi_composed_semantics_match_serial_reference() -> None:
    mpi_scheduler = pytest.importorskip(
        "chemfit.mpi_scheduler", reason="Missing mpi4py"
    )
    expected_values, expected_states = evaluate_serial_reference()

    objective = make_objective()
    contexts = [
        EvaluateContext(config=SimpleNamespace(depth=0)) for _ in ALL_PARAMETERS
    ]
    requests = [
        EvaluationRequest(parameters=parameters, ctx=ctx)
        for parameters, ctx in zip(ALL_PARAMETERS, contexts, strict=True)
    ]

    scheduler = mpi_scheduler.MPITreeScheduler(mpi_debug_log=False)
    with scheduler.prepare(objective) as schedule:
        if schedule.rank == 0:
            completed = sorted(
                schedule.evaluate_many(requests),
                key=lambda result: result.index,
            )
            values: list[Outcome] = []
            for result in completed:
                if isinstance(result.value, Exception):
                    values.append((type(result.value).__name__, str(result.value)))
                else:
                    values.append(result.value)

            states = [context_state(ctx) for ctx in contexts]
            assert_composed_semantics(values, states)
            assert_phase_failure_semantics(values, states)
            assert values == expected_values
            assert states == expected_states
        else:
            schedule.worker_loop()
