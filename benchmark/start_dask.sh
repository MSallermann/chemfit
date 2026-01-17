dask scheduler --host 127.0.0.1 --port 8786 --preload "./benchmark.py" &
dask worker 127.0.0.1:8786 --preload "./benchmark.py" --nworkers 1 --nthreads 1 &
dask worker 127.0.0.1:8786 --preload "./benchmark.py" --nworkers 1 --nthreads 1 &
dask worker 127.0.0.1:8786 --preload "./benchmark.py" --nworkers 1 --nthreads 1 &
dask worker 127.0.0.1:8786 --preload "./benchmark.py" --nworkers 1 --nthreads 1 &
