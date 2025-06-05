from typing import Any, List, Dict, Callable, Optional


class CombinedObjectiveFunction:
    def __init__(
        self,
        objective_functions: List[Callable[[Dict[str, float]], float]],
        weights: Optional[List[float]] = None,
    ):
        self.objective_functions = objective_functions

        if weights is None:
            self.weights = [1.0 for ob in self.objective_functions]
        else:
            self.weights = weights

        assert len(self.weights) == len(self.objective_functions)

    def n_terms(self):
        return len(self.weights)

    def add(self, obj_func: float, weight: float = 1.0):
        self.objective_functions.append(obj_func)
        self.weights.append(weight)
        return self

    def __call__(self, params: Dict[str, float]) -> Any:
        result = 0
        for i, w in enumerate(self.weights):
            # We make a copy of params here, just in case the objective function modifies it
            p = params.copy()
            result += self.objective_functions[i](p) * w

        return result
