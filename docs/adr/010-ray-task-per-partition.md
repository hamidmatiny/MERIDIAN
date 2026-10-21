# ADR-010: Ray task per partition for Sentinel

- Status: Accepted

## Context

Sentinel has to run the same pandera and PSI checks over many partitions. Doing that in one process hides the shape the later Ray replay will use, and a slow partition blocks the rest. The checks themselves are already deterministic: `quality.evaluate` returns `Incident` records whose ids do not depend on which worker ran them.

## Decision

Each `PartitionRef` is one Ray task. The task loads the parquet file and the reference-profile JSON, then calls `quality.evaluate`. The driver deduplicates by `Incident.id`. Ray is started in `ray_runtime`, which calls `ray.shutdown` on exit. `num_cpus` comes from settings (`RAY_NUM_CPUS`, default 2). A serial collector runs the same function in-process so a test can require the two result lists to be equal.

`uv run` makes Ray's default runtime hook set the worker executable to `uv run` and upload the working directory. Each worker then ignores `VIRTUAL_ENV` and builds a new project environment. `prepare_ray_runtime` sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before `ray.init`, including on the constant Ray already imported, and passes an explicit `runtime_env` that only carries that variable. Workers use the driver interpreter. The sandbox cluster uses the same helper. The CLI and the live Fix graph hold that cluster with `shared_ray` for the whole run. Sentinel's shutdown must not force the pre-critic dry-run to call `ray.init` again in the same process: Ray then raises `Perhaps you called ray.init twice by accident?` even when `ignore_reinit_error` is set. `start_ray` serializes the init and treats that assertion as "the cluster is already up" when the driver is connected.

## Consequences

- Sentinel does not call a model. A clean partition list ends the graph before Triage.
- Workers need a filesystem path, so the contract stores paths rather than the frames.
- Ray is initialized per Sentinel invocation and shut down afterwards, so a resumed run that does not re-enter Sentinel does not start Ray again.
- The equivalence test is the check that distribution across workers did not change the incidents.

## Alternatives considered

- A thread pool in-process. Rejected because the rest of the system is specified to use Ray, and a second executor would drift from the Verifier.
- One Ray task for the whole batch. Rejected because a single slow partition would block the batch, and the task boundary would not match a partition.
