from concurrent.futures import Executor, Future


class SequentialExecutor(Executor):
    def __init__(*args, **kwargs) -> None: ...

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True):
        pass  # No resources to shut down
