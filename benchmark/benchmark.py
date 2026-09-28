from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import tomllib
from dask.distributed import Client
from loky import ProcessPoolExecutor
from mpi4py import MPI

from chemfit.abstract_objective_function import (
    EvaluateContext,
    QuantityComputerObjectiveFunction,
)
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_policy import ExecutorPolicy
from chemfit.mpi_policy import MPIPolicy
from chemfit.wrap_funcs import to_quantity_computer


def gil_sleep_busy(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


@to_quantity_computer(pass_ctx=True)
def do_stuff(parameters: dict[str, Any], ctx: EvaluateContext):
    if ctx.config.release_gil:
        time.sleep(ctx.config.wait_time)
    else:
        gil_sleep_busy(ctx.config.wait_time)
    return {f"{k}_2": v**2 for k, v in parameters.items()}


def rmsd(quantities: dict[str, Any]) -> float:
    return sum(quantities.values())


class Method(str, Enum):
    threadpool = "threadpool"
    loky_processpool = "loky_processpool"
    synchronous = "synchronous"
    mpi = "mpi"
    dask = "dask"


@dataclass
class BenchmarkParams:
    label: str = ""

    release_gil: bool = True
    n_evals: int = 10
    n_workers: int | None = None
    n_warmup: int = 10
    method: Method = Method.synchronous

    n_params_list: list[int] = field(default_factory=list)
    n_terms_list: list[int] = field(default_factory=list)
    wait_times: list[float] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    params: BenchmarkParams
    time_taken_list: list[dict]


def run_benchmark(bm_params: BenchmarkParams) -> BenchmarkResult:
    time_taken_list = []
    client = None
    executor = None
    executor_policy = None

    if bm_params.method == Method.threadpool:
        executor = ThreadPoolExecutor(max_workers=bm_params.n_workers)
        executor_policy = ExecutorPolicy(executor)
    elif bm_params.method == Method.loky_processpool:
        executor = ProcessPoolExecutor(max_workers=bm_params.n_workers)
        executor_policy = ExecutorPolicy(executor)
    elif bm_params.method == Method.dask:
        client = Client(scheduler_file="./scheduler.json")
        executor_policy = ExecutorPolicy(client.get_executor())

    for n_terms in bm_params.n_terms_list:
        terms = [
            QuantityComputerObjectiveFunction(
                loss_function=rmsd, quantity_computer=do_stuff
            )
            for _ in range(n_terms)
        ]
        mpi_policy = (
            MPIPolicy(mpi_debug_log=False) if bm_params.method == Method.mpi else None
        )
        execution_policy = mpi_policy if mpi_policy is not None else executor_policy
        cob = CombinedObjectiveFunction(terms, execution_policy=execution_policy)
        if mpi_policy is not None and mpi_policy.rank != 0:
            mpi_policy.worker_loop(cob)
            continue

        for n_params in bm_params.n_params_list:
            params = {chr(i): float(i) for i in range(n_params)}

            ctx = EvaluateContext()
            ctx.config.release_gil = bm_params.release_gil

            for wait_time in bm_params.wait_times:
                ctx.config.wait_time = wait_time

                # some warmup iterations
                for _ in range(bm_params.n_warmup):
                    cob(params, ctx)

                _time_total = 0.0

                for i_eval in range(bm_params.n_evals):
                    print(
                        f"{n_terms = } {n_params = } {wait_time = }, eval {i_eval + 1} / {bm_params.n_evals}"
                    )

                    # different parameters each time, so nothing get get cached or anything funny like that
                    params = {chr(i): random.random() for i in range(n_params)}  # noqa: S311

                    # We only time the eval part
                    time_start = time.perf_counter()
                    res = cob(params, ctx)
                    _time_total += time.perf_counter() - time_start

                    # Ensure the results are correct
                    expected = cob.n_terms() * sum(v**2 for v in params.values())

                    print(f"   {res = }")
                    print(f"   {expected = }")

                    assert math.isclose(res, expected)

                avg_time = _time_total / bm_params.n_evals
                time_taken_list.append(
                    {
                        "n_params": n_params,
                        "n_terms": n_terms,
                        "wait_time": wait_time,
                        "time_taken": avg_time,
                    }
                )

        if mpi_policy is not None:
            mpi_policy.release_workers()

    if executor is not None:
        executor.shutdown()

    if client is not None:
        client.close()

    return BenchmarkResult(
        params=bm_params,
        time_taken_list=time_taken_list,
    )


def main(input_file: Path, output_folder: Path):
    with input_file.open("rb") as f:
        input_data = tomllib.load(f)

    default_values = input_data.get("DEFAULT", {})

    for k, v in input_data.items():
        if k == "DEFAULT":
            continue

        print(f"Running benchmark: {k}")

        bm_param_dict = default_values.copy()
        bm_param_dict.update(v)
        params = BenchmarkParams(**bm_param_dict)
        result = run_benchmark(params)
        output = output_folder / f"{k}.json"

        if MPI.COMM_WORLD.Get_rank() == 0:
            with output.open("w") as f:
                json.dump(asdict(result), f, indent=4)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("-i", type=Path, required=True)
    cli.add_argument("-o", type=Path, required=True)

    args = cli.parse_args()

    input_file = Path(args.i)
    output_folder = Path(args.o)
    output_folder.mkdir(exist_ok=True)

    main(input_file=input_file, output_folder=output_folder)
