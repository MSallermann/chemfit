import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ase import Atoms
from ase.calculators.lj import LennardJones

import chemfit


def test_fit_nevergrad_is_the_only_high_level_fit_export():
    assert callable(chemfit.fit_nevergrad)
    assert not hasattr(chemfit, "fit")


def test_fit_nevergrad_accepts_initial_observations():
    evaluated: list[dict[str, float]] = []

    def objective(parameters: dict[str, float]) -> float:
        evaluated.append(parameters)
        return parameters["x"] ** 2

    observations = (({"x": 0.0}, 0.0) for _ in range(1))
    result = chemfit.fit_nevergrad(
        objective,
        initial={"x": 1.0},
        budget=1,
        optimizer="OnePlusOne",
        initial_observations=observations,
    )

    assert len(evaluated) == 1
    assert result.best_parameters == {"x": 0.0}
    assert result.best_loss == 0.0


def test_public_api_workflow(tmp_path: Path):
    @chemfit.quantity()
    def model(parameters: dict[str, float], scale: float):
        return {"value": scale * parameters["x"]}

    assert not hasattr(model, "resources")

    def squared_error(quantities: dict[str, float], target: float):
        return (quantities["value"] - target) ** 2

    def external_command(
        parameters: dict[str, float],
        _workdir: Path,
        _ctx: chemfit.EvaluateContext,
    ):
        return [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "Path('result.txt').write_text(sys.argv[1])",
            str(parameters["x"]),
        ]

    def parse_external(output: Path):
        return {"value": float(output.read_text(encoding="utf-8"))}

    external_term = (
        chemfit.external_quantity(tmp_path)
        .with_cmd(external_command)
        .with_parser(parse_external, "result.txt")
        .with_loss(squared_error, target=2.0)
    )

    def calculator(
        parameters: dict[str, float],
        _atoms: Atoms,
        _ctx: chemfit.EvaluateContext,
    ):
        return LennardJones(epsilon=parameters["x"], sigma=1.0, rc=100.0)

    ase_term = (
        chemfit.ase_quantity(Atoms("Ar2", positions=[(0, 0, 0), (2 ** (1 / 6), 0, 0)]))
        .with_calculator(calculator)
        .with_loss(lambda quantities: (quantities["n_atoms"] - 2) ** 2)
    )

    def aggregate(
        terms: list[float],
        quantities: list[dict[str, object] | None],
        ctx: chemfit.EvaluateContext,
    ) -> float:
        ctx.meta["quantity_terms"] = sum(value is not None for value in quantities)
        return sum(terms)

    objective = chemfit.combine(
        model.bind(scale=2.0).with_loss(squared_error, target=4.0),
        external_term,
        ase_term,
        lambda parameters: (parameters["x"] - 2.0) ** 2,
        aggregator=aggregate,
    )
    assert not hasattr(objective, "resources")

    with ThreadPoolExecutor(max_workers=2) as executor:
        contexts = chemfit.evaluate_many(
            objective,
            [{"x": 1.0}, {"x": 2.0}],
            executor=executor,
        )

    assert [ctx.loss for ctx in contexts] == [6.0, 0.0]
    assert contexts[0].meta["children"][1]["quantities"] == {"value": 1.0}
    assert "energy" in contexts[0].meta["children"][2]["quantities"]
    assert contexts[0].meta["quantity_terms"] == 3

    result = chemfit.fit_nevergrad(
        objective,
        initial={"x": 2.0},
        budget=2,
        optimizer="OnePlusOne",
    )

    assert result.best_loss == result.best_context.opt_loss
    assert result.best_parameters == result.best_context.opt_params
