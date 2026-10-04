from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
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
from chemfit.executor_scheduler import ExecutorTreeScheduler
from chemfit.mpi_scheduler import MPITreeSchedule, MPITreeScheduler
from chemfit.scheduling import SerialScheduler
from chemfit.wrap_funcs import quantity


def gil_sleep_busy(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


@quantity(pass_ctx=True)
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


def make_scheduler(bm_params: BenchmarkParams, resources: ExitStack):
    method = Method(bm_params.method)

    if method == Method.threadpool:
        executor = resources.enter_context(
            ThreadPoolExecutor(max_workers=bm_params.n_workers)
        )
        return ExecutorTreeScheduler(executor=executor)

    if method == Method.loky_processpool:
        executor = resources.enter_context(
            ProcessPoolExecutor(max_workers=bm_params.n_workers)
        )
        return ExecutorTreeScheduler(executor=executor)

    if method == Method.dask:
        scheduler_file = Path(__file__).with_name("scheduler.json")
        client = resources.enter_context(
            Client(scheduler_file=str(scheduler_file), timeout="30s")
        )
        if bm_params.n_workers is not None:
            client.wait_for_workers(bm_params.n_workers, timeout=30.0)
        return ExecutorTreeScheduler(executor=client.get_executor())

    if method == Method.mpi:
        return MPITreeScheduler(mpi_debug_log=False)

    return SerialScheduler()


def make_objective(n_terms: int) -> CombinedObjectiveFunction:
    terms = [
        QuantityComputerObjectiveFunction(
            loss_function=rmsd,
            quantity_computer=do_stuff,
        )
        for _ in range(n_terms)
    ]
    return CombinedObjectiveFunction(terms)


def benchmark_objective(
    cob: CombinedObjectiveFunction,
    bm_params: BenchmarkParams,
    n_terms: int,
    time_taken_list: list[dict],
) -> None:
    for n_params in bm_params.n_params_list:
        params = {chr(i): float(i) for i in range(n_params)}

        ctx = EvaluateContext()
        ctx.config.release_gil = bm_params.release_gil

        for wait_time in bm_params.wait_times:
            ctx.config.wait_time = wait_time

            for _ in range(bm_params.n_warmup):
                cob(params, ctx)

            time_total = 0.0

            for i_eval in range(bm_params.n_evals):
                print(
                    f"{n_terms = } {n_params = } {wait_time = }, "
                    f"eval {i_eval + 1} / {bm_params.n_evals}"
                )

                # Use different parameters so neither backend nor objective caches
                # can turn repeated benchmark iterations into no-op evaluations.
                params = {chr(i): random.random() for i in range(n_params)}  # noqa: S311

                # Scheduler construction, worker startup, and warmup are excluded.
                time_start = time.perf_counter()
                res = cob(params, ctx)
                time_total += time.perf_counter() - time_start

                expected = cob.n_terms() * sum(v**2 for v in params.values())

                print(f"   {res = }")
                print(f"   {expected = }")

                assert math.isclose(res, expected)

            time_taken_list.append(
                {
                    "n_params": n_params,
                    "n_terms": n_terms,
                    "wait_time": wait_time,
                    "time_taken": time_total / bm_params.n_evals,
                }
            )


def run_benchmark(bm_params: BenchmarkParams) -> BenchmarkResult:
    time_taken_list = []

    with ExitStack() as resources:
        scheduler = make_scheduler(bm_params, resources)

        for n_terms in bm_params.n_terms_list:
            cob = make_objective(n_terms)
            with scheduler.prepare(cob) as schedule:
                if isinstance(schedule, MPITreeSchedule) and schedule.rank != 0:
                    schedule.worker_loop()
                    continue

                benchmark_objective(cob, bm_params, n_terms, time_taken_list)

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
