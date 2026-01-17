from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from loky import ProcessPoolExecutor
from scipy.optimize import curve_fit

from chemfit.abstract_objective_function import (
    EvaluateContext,
    QuantityComputerObjectiveFunction,
)
from chemfit.async_wrapper_cob import AsyncWrapperCOB
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.wrap_funcs import to_quantity_computer


def gil_sleep_busy(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


@to_quantity_computer(pass_ctx=True)
def do_stuff(parameters: dict[str, Any], ctx: EvaluateContext):
    if ctx.static.release_gil:
        time.sleep(ctx.static.wait_time)
    else:
        gil_sleep_busy(ctx.static.wait_time)
    return {f"{k}_2": v**2 for k, v in parameters.items()}


def rmsd(quantities: dict[str, Any]) -> float:
    return sum(quantities.values())


class Method(str, Enum):
    threadpool = "threadpool"
    loky_processpool = "loky_processpool"
    synchronous = "synchronous"


@dataclass
class BenchmarkParams:
    label: str = ""

    n_params: int = 10
    n_terms: int = 10
    release_gil: bool = True
    n_evals: int = 10

    n_workers: int | None = None

    n_warmup: int = 10

    method: Method = Method.synchronous

    wait_times: list[float] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    params: BenchmarkParams
    time_taken_list: list[float]
    time_slope: float
    time_offset: float


def run_benchmark(bm_params: BenchmarkParams) -> BenchmarkResult:
    terms = [
        QuantityComputerObjectiveFunction(
            loss_function=rmsd, quantity_computer=do_stuff
        )
        for _ in range(bm_params.n_terms)
    ]
    cob = CombinedObjectiveFunction(terms)

    params = {chr(i): float(i) for i in range(bm_params.n_params)}

    ctx = EvaluateContext()
    ctx.static.release_gil = bm_params.release_gil

    if bm_params.method == Method.threadpool:
        ob = AsyncWrapperCOB(cob)
        ctx.executor = ThreadPoolExecutor(max_workers=bm_params.n_workers)
    elif bm_params.method == Method.loky_processpool:
        ob = AsyncWrapperCOB(cob)
        ctx.executor = ProcessPoolExecutor(max_workers=bm_params.n_workers)
    elif bm_params.method == Method.synchronous:
        ob = cob

    time_taken_list = []
    for wait_time in bm_params.wait_times:
        ctx.static.wait_time = wait_time

        # some warmup iterations
        for _ in range(bm_params.n_warmup):
            ob(params, ctx)

        _time_total = 0.0
        for _ in range(bm_params.n_evals):
            params = {chr(i): random.random() for i in range(bm_params.n_params)}  # noqa: S311
            time_start = time.perf_counter()
            _ = ob(params, ctx)
            _time_total += time.perf_counter() - time_start

        time_taken_list.append(_time_total / bm_params.n_evals)

    def cost_func(x: float, a: float, b: float):
        return a * x + b

    popt, pcov = curve_fit(
        cost_func,
        bm_params.wait_times,
        time_taken_list,
        sigma=1.0 / np.log(time_taken_list),
    )

    return BenchmarkResult(
        params=bm_params,
        time_taken_list=time_taken_list,
        time_slope=popt[0],
        time_offset=popt[1],
    )
