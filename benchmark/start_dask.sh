#!/usr/bin/env bash

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

input_file=${1:-benchmark_settings_dask.toml}
output_dir=${2:-testerino}
workers=${CHEMFIT_DASK_WORKERS:-4}
scheduler_pid=
worker_pid=

cleanup() {
    [[ -z "${worker_pid}" ]] || kill "${worker_pid}" 2>/dev/null || true
    [[ -z "${scheduler_pid}" ]] || kill "${scheduler_pid}" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f scheduler.json
}
trap cleanup EXIT

rm -f scheduler.json
dask scheduler --host 127.0.0.1 --port 8786 --no-dashboard \
    --scheduler-file scheduler.json &
scheduler_pid=$!

dask worker --scheduler-file scheduler.json --nworkers "${workers}" \
    --nthreads 1 --no-dashboard &
worker_pid=$!

python3 benchmark.py -i "${input_file}" -o "${output_dir}"
