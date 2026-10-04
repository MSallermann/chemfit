Evaluation history and dashboard
================================

ChemFit can record evaluations in a local SQLite database and display them in
an optional Streamlit dashboard. Tabs separate the run overview, an interactive
context inspector, and raw context data. The overview shows recorded loss,
best loss within the selected window, durations, and a history table.
It refreshes every two seconds while open.

Recording every evaluation
--------------------------

Register the logger after the UUID and timing hooks so its record includes
their output. Attach it to the outermost objective::

    from chemfit.wrap_funcs import WrappedObjectiveFunctor
    from chemfit.objective_hooks import UUIDHook, TimingHook
    from chemfit.evaluation_logging import SQLiteEvaluationLogger

    objective = WrappedObjectiveFunctor(lambda p: p["x"] ** 2)
    objective.register_eval_hook(UUIDHook())
    objective.register_eval_hook(TimingHook(), recursive=True)
    recorder = SQLiteEvaluationLogger("chemfit-run.sqlite", run_name="first fit")
    objective.register_eval_hook(recorder)

    objective({"x": 2.0})
    objective({"x": 1.0})

Each new recorder creates a separate run in the same database. The logger
requires the default UUID metadata key (``evaluation_id``); timing is optional.
With ``recursive=True``, timing is registered on all current descendant objectives,
including nested combined terms. Leave recursion disabled for the recorder so it records only
outer evaluations. Terms added later must be registered separately.
The UUID is regenerated for each evaluation, including when a context is reused.
Records with the same UUID in the same run are ignored on subsequent writes.

Evaluation rows contain only ``run_id``, ``evaluation_id``, and ``context_json``.
Run identity is kept for grouping, while all evaluation data remains inside the
JSON snapshot. The logger does not interpret metadata or duplicate loss, timing,
status, exceptions, optimizer steps, or worker indices into separate columns.
``read_evaluations`` returns those identities and the decoded ``context``.

The dashboard derives its display columns from that JSON. It recognizes optional
``meta.status`` values (``completed``, ``failed``, ``invalid_loss``) and
``meta.exception``. Otherwise, it classifies records by whether the loss is finite;
this does not establish that the evaluation succeeded. Error details are shown
only when included in metadata. These display conventions do not change storage.

Databases created with the previous schema remain readable. To record new runs
with this schema, use a new database file; existing files are not migrated.

The post hook records successful and failed evaluations. It records the loss
at the point where it is attached, which can precede the fitter's replacement
of invalid losses with penalties. Register it last to include preceding hooks'
metadata. Pre-hook failures prevent evaluation and are not recorded. The logger
persists only ``ctx.to_meta_data()``; scratch exceptions are not included.
To retain exception details or an explicit evaluation status, put them in
``ctx.meta`` using a preceding hook. The logger preserves those fields as-is.

Nested contexts
---------------

SQLite stores one row for each outer evaluation. The ``context_json`` column
contains ``ctx.to_meta_data()``, including the full ``meta.children`` tree::

    {
        "loss": 12.0,
        "parameters": {"x": 2.0},
        "quantities": null,
        "meta": {
            "evaluation_id": "...",
            "children": [
                {"loss": 8.0, "meta": {"children": [...]}},
                {"loss": 4.0, "meta": {}}
            ]
        }
    }

Children do not need separate database rows or UUIDs. Their ordering and nesting
are preserved. The **Context tree** tab draws the parent-child relationships,
with loss and any recorded timing on each node. Click a node (or use the context
selector) to highlight it and inspect its parameters, quantities, and metadata.
Its history plots and table show raw loss and timing across the selected window.
Missing, skipped, and nonfinite values appear as gaps, not zeros. Paths use
zero-based child positions: ``root/0/1`` means child 1 of child 0. Skipped
children are labelled when the parent records ``skipped_indices``.
History matches nodes by child position, so it assumes that objective ordering
and structure remain consistent. A child can have a recorded result even when
the outer evaluation fails; failure details are available only if recorded in
the context metadata.

Child losses are raw, unweighted values: weights and custom reducers mean they
need not sum to the parent's loss. To see child timings, register ``TimingHook``
on the root with ``recursive=True``. Missing timing is shown as a dash, not zero.
Parent timings include nested work; timings should not be added together,
especially when children execute in parallel.

For contexts with timed children, the dashboard estimates effective parallelism::

    effective_parallelism = sum(direct_child_durations) / parent_duration
    parallel_efficiency = effective_parallelism / worker_count

Enter **Workers available to this context** to display efficiency as a percentage.
For example, 8 seconds of child work in 2.5 seconds of parent wall time gives
3.2x effective parallelism, or 80% efficiency with four workers. The worker count
is not recorded in the database; the entered count applies to that node's history
within the window. Both metrics also appear when timings are available.

Only direct children are summed, to avoid double-counting descendants. Skipped
children are excluded; missing or invalid required timings, zero parent duration,
and leaf contexts leave the metrics blank. This is a wall-time estimate, not CPU
utilization or a measured speedup against a serial baseline. Nested parallelism
or an incorrect worker count can produce values above 100%; these are not capped.

The **Raw data** tab retains the full expandable JSON for the selected evaluation.
SQLite JSON functions can also query the stored tree. Scratch state, live executors, and
``ctx._children`` objects are excluded. The logger uses the already gathered
metadata so it preserves children returned by MPI ranks.

NumPy arrays and scalars are converted to JSON lists and numbers. Nonfinite
floats are stored as the strings ``"nan"``, ``"inf"``, and ``"-inf"`` in JSON.
Other unsupported objects raise a
serialization error. Large quantities are stored in full in this initial version.

Parallel fitting
----------------

For MPI, register the recorder on the combined objective on rank zero, after the
metadata hooks. Only rank zero writes; child metadata has already been gathered.
Recursive timing is also registered on the combined objective. For MPI,
register it on every rank before entering
``MPITreeSchedule.worker_loop()``; registration does not broadcast hooks to
other ranks. For loky, register before submitting tasks.

When entire candidate evaluations run in loky workers, a fitter callback can
write their returned contexts from the driver::

    # Here objective has UUIDHook and TimingHook, but no SQLite post hook.
    fitter.register_callback(recorder, n_steps=1)

This saves the latest context per candidate slot at each callback. Idle
contexts in a partial batch and repeated callbacks do not produce duplicate
records. It is a sampled history: in particular, SciPy can evaluate several
candidates between optimizer callbacks. Callbacks cannot record exceptions
that abort a fit, and exception scratch state is not returned from loky
workers. The logger does not infer success from loss or introduce status
metadata.

The post hook can also write from workers on the same machine: it opens a fresh
SQLite connection per record. Writes are serialized with a 30-second busy
timeout. Use a local disk; WAL mode requires readers and writers on the same
machine, not a shared NFS database across cluster nodes. For cluster runs,
record on the driver and view the dashboard on that host.

Opening the dashboard
---------------------

Install Streamlit, Polars, and Plotly and launch the app::

    pip install streamlit polars plotly
    python -m chemfit.dashboard chemfit-run.sqlite

The server listens on localhost. Open the URL printed by Streamlit. The
dashboard opens the database read-only, can inspect previous runs, and does
not need the optimizer to be running.

Large histories
---------------

The dashboard reads at most 2,000 evaluations into memory, following the latest
evaluations by default. Disable **Follow latest evaluations** and enter
**Start at insertion** to inspect older windows. Insertion IDs are database-wide
SQLite cursors; gaps are expected when runs are interleaved. All summary metrics,
including best loss and failure counts, describe the selected window, not the
entire run. Exact run-wide aggregation is not performed on refresh.

Tables and the evaluation picker show 200 records per page. Plot sampling retains
each interval's minimum, maximum, and a missing value when present. Summary
metrics use every evaluation in the window, regardless of sampling. Unchanged
windows reuse decoded contexts and node histories, and replacing the selected
window releases the previous cache.

The run/insertion index permits bounded queries without reading all earlier JSON
records or using increasingly expensive SQL OFFSET pagination. New loggers
create this administrative index automatically; it contains no context metadata
and changes none of the three evaluation columns. To index an existing database
once before launching::

    python -m chemfit.dashboard --index history.sqlite

Building the index can take time for a large existing database and requires write
access. Ordinary dashboard reads remain read-only. The windowed design avoids
loading tens of millions of contexts into the browser or server, but it does not
provide a global best-loss statistic or overview plot across all those records;
that would require a separate aggregation strategy.

Programmatic access::

    from chemfit.evaluation_logging import read_runs, read_evaluations

    runs = read_runs("chemfit-run.sqlite")
    records = read_evaluations("chemfit-run.sqlite", runs[0]["run_id"])
    children = records[0]["context"]["meta"].get("children", [])

For large runs, use bounded access instead of ``read_evaluations``, which loads
the entire run::

    from chemfit.evaluation_logging import read_evaluation_window

    latest = read_evaluation_window("chemfit-run.sqlite", runs[0]["run_id"], limit=200)
    older = read_evaluation_window(
        "chemfit-run.sqlite", runs[0]["run_id"], start_rowid=1_000_000, limit=200
    )
