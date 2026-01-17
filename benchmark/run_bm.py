import json
from dataclasses import asdict
from pathlib import Path

import tomllib

import benchmark

INPUT = Path("./benchmark_settings.toml")
OUTPUT_FOLDER = Path("./results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

with INPUT.open("rb") as f:
    input_data = tomllib.load(f)

default_values = input_data.get("DEFAULT", {})

for k, v in input_data.items():
    if k == "DEFAULT":
        continue

    print(f"Running benchmark: {k}")

    bm_param_dict = default_values.copy()
    bm_param_dict.update(v)
    params = benchmark.BenchmarkParams(**bm_param_dict)
    result = benchmark.run_benchmark(params)
    output = OUTPUT_FOLDER / f"bm_result_{k}.json"
    with output.open("w") as f:
        json.dump(asdict(result), f, indent=4)
