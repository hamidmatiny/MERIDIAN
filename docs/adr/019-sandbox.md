# ADR-019: Sandbox for untrusted remediation code

- Status: Accepted

## Context

The fix agent writes the remediation. That source is untrusted. It has to run on the failing partition so the verifier can replay it, and it must not be able to read the repo, the network, or the process environment while it does.

## Decision

Two steps, in order.

1. A static AST gate in the parent process. The repair module may import only `pandas`, `numpy`, `math`, and `datetime`. A regression test may also import `pandera`, because the test is the pandera check and the repair itself must not depend on it. The gate rejects `open`, `exec`, `eval`, `compile`, `__import__`, `getattr` on a dunder or a non-constant name, any attribute whose name starts with `__`, and the modules `os`, `sys`, `subprocess`, `socket`, `pathlib`, and `importlib`. A relative import is rejected. A rejection becomes a `VerificationResult` failure. The source is not executed.

2. Execution is a Ray remote task. The task runs in a worker process, with `MERIDIAN_SANDBOX_TIMEOUT_S` (default 20 seconds). On timeout the parent cancels the task. The worker clears `os.environ` for the duration of the exec, creates a temporary working directory, and runs the source under a restricted `__builtins__` whose `__import__` only allows the same module list. The entry points are `repair`, `quarantine`, and `check`. The cluster is process-wide: a remediation does not pay `ray.init` or `ray.shutdown` on every call, and `ray_runtime` does not shut down a cluster it did not start. The first call also waits for a one-task warmup so Ray's worker environment build is not charged against the remediation timeout. The timeout then bounds the untrusted task.

Memory and network are best-effort, and they are weaker on macOS.

- Ray's `memory=` option is a scheduling hint (`MERIDIAN_SANDBOX_MEMORY_MB`, default 512). It does not kill a task that exceeds it.
- On Linux the worker also calls `setrlimit(RLIMIT_AS)` to that size, and ignores the error when the platform refuses the limit.
- On macOS `RLIMIT_AS` accounts virtual address space, not resident memory. A few hundred megabytes of virtual address space is enough to abort a Python worker that has not used that much RAM. The worker therefore does not set `RLIMIT_AS` on Darwin. There is no hard memory cap on macOS in this sandbox.
- There is no unprivileged network namespace on macOS, and `sandbox-exec` is not used. The control on network access is the import gate: `socket` and the other banned modules cannot be imported. A banned import never reaches the worker. That is not a packet filter. The Ray worker process itself starts with the parent's environment; only the remediation exec sees an empty environment.

## Consequences

- A patch that fails the gate counts as a verifier failure and can be revised, up to the fix-attempt cap.
- Pandera is available to the regression test and not to the repair module.
- A timeout is a failed check named `timeout`. The code is not left running.
- The sandbox calls `prepare_ray_runtime`, so a worker does not rebuild the project with `uv`. The warmup task only touches an already-imported interpreter.
- A demo on a Mac does not have a hard memory cap or a network namespace. The ADR is the record of that limit. A later Linux deployment can turn the `setrlimit` path on without changing the gate.

## Alternatives considered

- `RestrictedPython`. Rejected for this phase because the AST gate plus a subprocess already covers the banned list, and a second policy language would drift from it.
- Running the source with the parent's `__builtins__` after the gate. Rejected because a gate bug would then be able to call `open`.
- `sandbox-exec` profiles on macOS. Rejected because the tool is deprecated and SIP makes the profile unreliable.
- A container per remediation. Rejected for the local demo. The Ray task is the process boundary this phase can start and stop.
