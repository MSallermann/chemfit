import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import benchmark

# Synchronous
params = benchmark.BenchmarkParams(
    n_params=10,
    n_terms=10,
    release_gil=True,
    n_evals=100,
    wait_times=np.logspace(-7, -2, 10).tolist(),
)
output = Path("bm_result_sync.json")
res = benchmark.run_benchmark(params)
with output.open("w") as f:
    json.dump(asdict(res), f, indent=4)


# Threadpool
params = benchmark.BenchmarkParams(
    n_params=10,
    n_terms=10,
    release_gil=True,
    n_evals=100,
    wait_times=np.logspace(-7, -2, 10).tolist(),
    use_threads=True,
    n_workers=4,
)
output = Path("bm_result_threads.json")
res = benchmark.run_benchmark(params)
with output.open("w") as f:
    json.dump(asdict(res), f, indent=4)
