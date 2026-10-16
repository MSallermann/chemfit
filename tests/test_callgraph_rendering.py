from pathlib import Path
from typing import Any

import pytest

from chemfit.abstract_objective_function import component
from chemfit.callgraph import call_graph_to_dot, objective_to_call_graph


def test_svg_rendering_does_not_require_a_dot_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> None:
        calls.append((command, kwargs))

    monkeypatch.setattr("chemfit.callgraph.subprocess.run", run)
    svg_path = tmp_path / "nested" / "graph.svg"
    graph = objective_to_call_graph(component()(lambda value: value))

    dot_source = call_graph_to_dot(
        graph,
        svg_output_path=svg_path,
        path_to_dot_exec="custom-dot",
    )

    assert svg_path.parent.is_dir()
    assert calls == [
        (
            ["custom-dot", "-Tsvg", "-o", str(svg_path)],
            {
                "check": True,
                "input": dot_source,
                "text": True,
                "encoding": "utf-8",
            },
        )
    ]
