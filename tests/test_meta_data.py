from __future__ import annotations

import pytest

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveFunctor,
    QuantityComputerObjectiveFunction,
)
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.wrap_funcs import objective, quantity


class MyFunctor(ObjectiveFunctor):
    def __init__(self, f: float) -> None:
        """Initialize My Functor."""
        super().__init__()
        self.f = f
        self.meta_data = {}

    def _evaluate(self, parameters: dict, ctx: EvaluateContext) -> float:
        val = self.f * parameters["x"] ** 2
        ctx.meta["last_value"] = val
        return val


@objective()
def a(p: dict):
    return p["y"] ** 2


@quantity()
def quants(p: dict):
    return {"x_plus_y": p["x"] + p["y"]}


def loss(q: dict, p: dict):
    return q["x_plus_y"] + p["y"]


INITIAL_PARAMS = {"x": 1.0, "y": 2.0}

COB = CombinedObjectiveFunction(
    [
        a,
        MyFunctor(1),
        QuantityComputerObjectiveFunction(loss_function=loss, quantity_computer=quants),
    ]
)

EXPECTED = [
    {
        "quantities": None,
        "parameters": {"x": 1.0, "y": 2.0},
        "loss": 4.0,
        "meta": {},
    },
    {
        "quantities": None,
        "parameters": {"x": 1.0, "y": 2.0},
        "loss": 1.0,
        "meta": {"last_value": 1.0},
    },
    {
        "quantities": {"x_plus_y": 3.0},
        "parameters": {"x": 1.0, "y": 2.0},
        "loss": 5.0,
        "meta": {},
    },
]


def test_gather_meta_data():
    # Evaluate the objective function
    COB(INITIAL_PARAMS, ctx := EvaluateContext())
    meta_data = ctx.to_meta_data()["meta"]["children"]

    print(f"{meta_data = }")
    print(f"{EXPECTED = }")

    assert meta_data == EXPECTED


def test_with_meta_is_copy_on_write_and_merges() -> None:
    plain_objective = MyFunctor(1).with_meta(kind="custom")
    plain_ctx = EvaluateContext()
    plain_objective(INITIAL_PARAMS, plain_ctx)
    assert plain_ctx.meta == {"kind": "custom", "last_value": 1.0}

    objective_base = a
    objective_tagged = objective_base.with_meta(dataset="training")
    objective_retagged = objective_tagged.with_meta(dataset="validation", split=2)

    assert objective_base.static_meta_data == {}
    assert objective_tagged.static_meta_data == {"dataset": "training"}

    objective_ctx = EvaluateContext()
    objective_retagged(INITIAL_PARAMS, objective_ctx)
    assert objective_ctx.meta == {"dataset": "validation", "split": 2}

    computer_base = quants
    computer_tagged = computer_base.with_meta(source="simulation")
    computer_retagged = computer_tagged.with_meta(source="cache", replica=3)

    assert computer_base.static_meta_data == {}
    assert computer_tagged.static_meta_data == {"source": "simulation"}

    computer_ctx = EvaluateContext()
    computer_retagged(INITIAL_PARAMS, computer_ctx)
    assert computer_ctx.meta == {"source": "cache", "replica": 3}


def test_objective_fluent_variants_have_independent_hook_lists() -> None:
    @objective()
    def base(parameters: dict, *, extra: str = "") -> float:
        del extra
        return parameters["x"]

    def hook_a(ctx: EvaluateContext) -> None:
        del ctx

    def hook_b(ctx: EvaluateContext) -> None:
        del ctx

    base = base.with_meta(dataset="source")
    base.register_eval_hook(pre=hook_a, post=hook_a)

    tagged = base.with_meta(dataset="training")
    tagged.register_eval_hook(pre=hook_b, post=hook_b)

    assert base.pre_eval_hooks == [hook_a]
    assert tagged.pre_eval_hooks == [hook_a, hook_b]
    assert base.post_eval_hooks == [hook_a]
    assert tagged.post_eval_hooks == [hook_a, hook_b]

    bound = base.bind(extra="value")
    bound.register_eval_hook(pre=hook_b, post=hook_b)

    assert base.pre_eval_hooks == [hook_a]
    assert bound.pre_eval_hooks == [hook_a, hook_b]
    assert base.post_eval_hooks == [hook_a]
    assert bound.post_eval_hooks == [hook_a, hook_b]
    assert bound.static_meta_data == base.static_meta_data


def test_composed_static_metadata_precedence_in_child_context() -> None:
    computer = quants.with_meta(dataset="liquid", owner="quantity")
    term = computer.with_loss(loss).with_meta(
        observable="density",
        owner="objective",
    )

    ctx = EvaluateContext()
    CombinedObjectiveFunction([term])(INITIAL_PARAMS, ctx)

    child_meta = ctx.meta["children"][0]["meta"]
    assert child_meta == {
        "dataset": "liquid",
        "observable": "density",
        "owner": "objective",
    }


def test_gather_meta_data_mpi():
    mpi_scheduler = pytest.importorskip(
        "chemfit.mpi_scheduler", reason="Missing mpi4py"
    )

    scheduler = mpi_scheduler.MPITreeScheduler(mpi_debug_log=False)
    with scheduler.prepare(COB) as mpi:
        if mpi.rank == 0:
            mpi.evaluate(INITIAL_PARAMS, ctx := EvaluateContext())
            meta_data = ctx.to_meta_data()["meta"]["children"]

            print(f"{meta_data = }")
            print(f"{EXPECTED = }")

            assert meta_data == EXPECTED

        else:
            mpi.worker_loop()
