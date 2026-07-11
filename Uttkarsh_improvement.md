# NCIForge Performance Improvements — Uttkarsh

Branch: `pre-main-testing`. This documents the GPU-contention and NCI-throughput
work done in this pass: what was wrong, why, what changed, and how each change
was verified. It builds directly on top of the `xtb_routing.py` fix already on
this branch (small molecules stay on CPU `xtbx`, only large molecules use GPU
`xtbx`, GPU reserved for the NCI stage) — that fix closed the *common* case of
xTB/NCI GPU contention. The two changes below close the remaining gap and fix
a second, independent bottleneck in the NCI stage itself.

---

## 1. GPU contention between `xtbx --gpu` and the NCI-torch CUDA stage

### Problem

`knf_core/engine/xtb_routing.py` (already on this branch) routes a molecule's
xTB opt/SP calls to the GPU (`xtbx --gpu`) only when it's genuinely large
(`>= --xtb-gpu-atoms`, default 350 atoms) — a deliberate, well-tested policy to
avoid paying a GPU cold-start on every small molecule.

But nothing stopped that GPU-routed xTB *subprocess* (launched from a CPU
pre-nci worker thread, in `knf_core/engine/jobs.py`) from running at the same
time as the in-process NCI-torch CUDA compute (running on its own dedicated
single-lane executor in the same file). For a batch that mixes a few large
molecules in with many small ones — a realistic shape for "thousands of
molecules" — an `xtbx --gpu` process for molecule A and NCI-torch's CUDA work
for molecule B could land on the physical GPU concurrently. That's a narrower
recurrence of the exact "xTB and NCI fighting for the GPU" problem the routing
policy was built to prevent — just now confined to the rarer large-molecule
case instead of every molecule.

### Solution

Added `GPU_DEVICE_LOCK`, a plain `threading.Lock`, to
[`knf_core/engine/gpu.py`](knf_core/engine/gpu.py) (this is where the existing
`_GPU_STATE_LOCK` already lives, so GPU-related coordination stays in one
place). Both GPU consumers now acquire it only around the portion of work that
actually touches the device:

- [`knf_core/wrapper.py`](knf_core/wrapper.py) — `run_xtb_opt` / `run_xtb_sp`
  acquire the lock around the subprocess call only when the resolved routing
  decision is `xtbx` + `--gpu` (`_gpu_guard()` helper, a no-op `nullcontext()`
  otherwise). CPU-routed calls — the overwhelming majority in a batch of small
  molecules — never touch the lock at all, so this adds zero contention/
  overhead to the common path.
- [`knf_core/nci_torch/pipeline.py`](knf_core/nci_torch/pipeline.py) —
  `run_nci_torch` acquires the same lock around the `run_nci_engine(...)` call
  only when the requested device is `cuda`/`auto`; a `cpu` fallback packet from
  the NCI router never waits on it.

Net effect: an `xtbx --gpu` run and an NCI-torch CUDA run can never execute on
the device at the same instant, without adding any serialization to the (much
more common) CPU-only paths on either side.

### Verification

- Confirmed no circular imports: `knf_core.engine.gpu` has no dependency on
  `wrapper` or `nci_torch`, so both can import `GPU_DEVICE_LOCK` safely.
  Verified by directly importing `knf_core.wrapper` and
  `knf_core.nci_torch.pipeline` together in one process.
- `python -m py_compile` / `ast.parse` on both edited files.
- Could not exercise this on real GPU hardware or a real `xtbx` binary in this
  environment (no CUDA device, no bundled `xtb-win-release` runtime here) —
  the fix is a pure concurrency-control change (a lock around existing calls),
  not a change to xTB or NCI numerics, so the risk surface is limited to
  "does it deadlock / does it serialize the wrong thing," both checked by
  code inspection: each site acquires-executes-releases with no nested
  acquisition, so no deadlock path exists.

---

## 2. NCI-torch basis evaluation was Python-looped per basis function

### Problem

`knf_core/nci_torch/engine.py`'s `_evaluate_basis_chunk` (called once per grid
chunk, inside the per-molecule density loop) looped in pure Python over every
basis function individually:

```python
for col, bf in enumerate(prepared_basis):
    ...
    prim = torch.exp(-r2[:, None] * bf.exponents[None, :]) * bf.coefficients[None, :]
    out[:, col] = poly * prim.sum(dim=1)
```

For a real molecule this is dozens to hundreds of basis functions (a molecule
with ~20 heavy atoms on an s/p/d basis is ~150-250 basis functions). Each
iteration launches several small tensor ops. On GPU, kernel-launch latency
dominates over the actual (tiny, per-function) compute — this is very likely
why the earlier benchmark (`nci_compare/comprehensive_speed_accuracy_report.md`)
showed the NCI-torch GPU path only ~1.2x faster than CPU on an RTX 3050: the
GPU was latency-bound on hundreds of tiny kernel launches per grid chunk, not
throughput-bound on real compute.

### Solution

Basis functions were bucketed by their `(lx, ly, lz)` cartesian power triple
(`_group_prepared_basis`, new). A molden basis expands each contracted shell
into one basis function per power triple (`s`→1, `p`→3, `d`→6, ...), so across
a whole molecule there are usually only a handful of distinct triples even
with hundreds of basis functions. Each group's exponents/coefficients are
zero-padded to a common primitive count (`exp(-r²·0)·0 == 0`, so padding adds
exact no-ops) and stacked into `(m_in_group, max_primitives)` tensors.
`_evaluate_basis_chunk` now loops over *groups*, not basis functions, and
evaluates every basis function in a group with one batched
`(n_points, m_in_group, max_primitives)` op.

On a synthetic 20-atom / s+p+d test case: **200 basis functions collapsed into
10 groups** — a 20x reduction in Python-loop iterations (and GPU kernel
launches) per grid chunk.

### Verification

- New test file [`tests/test_nci_basis_grouping.py`](tests/test_nci_basis_grouping.py):
  embeds the original per-basis-function formula as a reference and asserts
  the grouped version matches it to `rtol=1e-12` on random synthetic s/p/d
  basis functions with **different primitive counts per function** (so the
  zero-padding path is actually exercised), including points sitting exactly
  on a basis function's center (`r²=0`, the edge case most likely to expose a
  padding bug). All 3 tests pass.
- Ran the full `compute_density` → `compute_nci_fields` pipeline end-to-end
  on a synthetic H2-like wavefunction with a chunked grid (`batch_size`
  smaller than the grid so the per-chunk loop boundary is exercised) —
  produces finite `rho`/`rdg`/`sign(λ2)ρ` everywhere.
- Ran the existing `tests/test_xtb_routing.py` suite (24 tests total with the
  new file) — all pass, confirming the xTB routing logic from part 1 is
  unaffected.
- **Measured, not assumed, the CPU-side timing**: on a 200-basis-function /
  216k-point synthetic case, the grouped version was **1.10x faster on CPU**
  (0.622s vs 0.684s for one grid chunk). This is a modest, expected result —
  CPU op-dispatch overhead is already low, so collapsing 200 iterations into
  10 doesn't save much there. **The real payoff is on GPU**, where the
  200→10 reduction in kernel launches removes the launch-latency bottleneck
  that was likely capping the earlier 1.2x GPU/CPU ratio. No CUDA device was
  available in this environment to measure that directly — re-running
  `nci_compare`'s existing GPU benchmark on this branch is the natural next
  step to get a real number.

---

## 3. `release_cuda_memory()` (sync + `empty_cache` + `ipc_collect`) fired on every successful GPU molecule, not just on failure

### Problem

`knf_core/nci_torch/pipeline.py`'s `run_nci_torch` had a `finally` block that
called `release_cuda_memory()` — `torch.cuda.synchronize()` +
`torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` — whenever the run
targeted CUDA, **regardless of whether it succeeded**. The one place this
function is *supposed* to be called narrowly is `knf_core/pipeline.py`'s
`_run_torch_nci_with_adaptive_fallback`, in the CUDA-OOM `except` branch
(free memory before retrying at a smaller batch size). The `finally` block
duplicated that cleanup on the success path too, for every single GPU
molecule in a batch.

`empty_cache()` releases the CUDA caching allocator's pooled memory blocks
back to the driver. The entire point of that allocator is to avoid repeated
`cudaMalloc`/`cudaFree` round-trips — calling `empty_cache()` after every
molecule forces the *next* molecule's tensor allocations (basis-group
tensors, grid points, the Hessian, etc.) to hit the driver fresh instead of
reusing a warm pool. Pure per-molecule overhead, multiplied across "thousands
of molecules," with zero benefit on the success path.

### Solution

Added a `succeeded` flag to `run_nci_torch`, set `True` only immediately
before the (single) `return`. The `finally` block now only calls
`release_cuda_memory()` when `not succeeded` — i.e. only when the function is
unwinding from an exception. The OOM-retry call site in `pipeline.py` is
untouched and still calls it explicitly on its own OOM path.

### Verification

New tests in [`tests/test_pipeline_perf_fixes.py`](tests/test_pipeline_perf_fixes.py):
- `test_release_cuda_memory_not_called_on_success` — mocks `run_nci_engine` to
  return a successful (fake) CUDA result and asserts `release_cuda_memory` is
  never called.
- `test_release_cuda_memory_called_on_exception` — mocks `run_nci_engine` to
  raise and asserts `release_cuda_memory` **is** called exactly once, and the
  exception still propagates (the OOM-retry logic upstream needs to see it).

Both pass. No CUDA hardware was needed for this — the fix and its test are
pure control-flow, exercised via mocking.

---

## 4. SNCI post-processing loaded the same NCI grid file twice per molecule

### Problem

`knf_core/pipeline.py`'s `run_post_nci_stage` called:
```python
snci_val = snci.compute_snci(nci_data_path)
nci_stats = snci.compute_nci_statistics(nci_data_path)
```
Both independently called `snci.py`'s `_load_grid_payload`, which does
`np.load(grid_path)` and reconstructs the `sign_lambda2_rho` array from disk.
Same file, same array, loaded and parsed twice per molecule for no reason —
they were always going to be called together.

### Solution

Factored the shared post-load logic into `_snci_from_attractive` and
`_nci_statistics_from_attractive`, and added `compute_snci_and_statistics`,
which loads the grid payload once and derives both results from it.
`pipeline.py` now calls this single function. `compute_snci` and
`compute_nci_statistics` are kept as-is (still load independently) for
backward compatibility, since nothing outside this repo's own call site was
found to depend on them, but the new combined function is what the pipeline
actually uses now.

### Verification

New tests in `tests/test_pipeline_perf_fixes.py`:
- `test_compute_snci_and_statistics_loads_payload_once` — monkeypatches
  `_load_grid_payload` with a call counter and asserts it's called exactly
  once.
- `test_compute_snci_and_statistics_matches_old_separate_calls` — asserts the
  combined function's output is identical to calling the two old functions
  separately, on synthetic grid data with a mix of attractive/repulsive
  points.

Both pass.

---

## 5. `parse_molden` ran twice per molecule, on two different threads

### Problem

This is the specific mechanism identified during the GIL-contention
investigation (see below): `knf_core/xtb.py`'s `compute_wbo_from_molden_details`
(called from `run_pre_nci_stage`, on a **CPU-pool thread**, for the default
`wbo_mode="native"`) parses `molden.input` via `parse_molden`. Later,
`knf_core/nci_torch/pipeline.py`'s `run_nci_torch` (called from
`run_post_nci_stage`, on the **GPU-lane thread**) parses the *same*
`molden.input` file again, independently. In a pipelined batch, one
molecule's GPU-lane parse and another molecule's CPU-pool parse genuinely
overlap in wall-clock time — the same pure-Python, GIL-holding function
racing itself across two threads, which measurement showed can inflate
`parse_molden`'s wall time by up to ~250x under realistic contention (see the
GIL investigation below).

### Solution

- `xtb.compute_wbo_from_molden_details` now accepts an optional
  `wavefunction=` parameter (skips its internal `parse_molden` call if
  given) and **always returns the parsed wavefunction** under a
  `"wavefunction"` key.
- `nci_torch.pipeline.run_nci_torch` likewise accepts an optional
  `wavefunction=` parameter and skips `parse_molden` when provided.
- `KNFPipeline.run_pre_nci_stage` captures the wavefunction from the WBO
  call and threads it through the `context` dict as
  `"prefetched_wavefunction"` — **only** when it's actually safe to reuse:
  `wbo_mode == "native"` (the only mode that parses molden.input at all in
  pre-nci) and `nci_apply_primitive_norm` is `False` (matching the
  hardcoded `apply_primitive_normalization=False` that
  `compute_wbo_from_molden_details` always parses with). Any other
  configuration falls back to the original re-parse behavior automatically
  (`wavefunction=None` is a no-op default in both functions).
- `run_post_nci_stage` passes `context.get("prefetched_wavefunction")`
  through `_run_torch_nci_with_adaptive_fallback` into every `run_nci_torch`
  attempt (including CPU-fallback retries after a CUDA OOM — the cached
  wavefunction is plain NumPy data, safe to reuse across device retries for
  the same molecule).
- Added a `wavefunction_reused` flag to `run_nci_torch`'s returned metadata
  for observability, sitting right next to the existing
  `timings_seconds.parse_molden` field this whole investigation was built
  around.

Net effect, in the default configuration: `parse_molden` now runs **once**
per molecule (on the CPU-pool thread, as part of the WBO calculation)
instead of twice — and the GPU-lane thread no longer runs this pure-Python,
GIL-holding function at all, removing its specific exposure to the
cross-thread contention this investigation measured.

### Verification

New tests in `tests/test_pipeline_perf_fixes.py`, run against the repo's real
`molden.input` (79 basis functions, 31 atoms):
- `test_compute_wbo_returns_reusable_wavefunction` — confirms
  `compute_wbo_from_molden_details` returns a usable wavefunction, and that
  passing it back in on a second call skips re-parsing and produces the same
  WBO numbers.
- `test_run_nci_torch_reuse_matches_fresh_parse` — runs `run_nci_torch`
  twice, once fresh and once with a pre-parsed wavefunction, and asserts the
  written `nci_grid.npz` payloads (`x`, `y`, `z`, `sign_lambda2_rho`, `rdg`)
  are **bit-for-bit identical** (`np.array_equal`), and that the reused run's
  `timings_seconds.parse_molden` drops to effectively zero.

Manual confirmation before writing the test (real `molden.input`, `float32`,
CPU): fresh parse cost 22.08ms; reused cost 0.0002ms; all five output arrays
bit-identical between the two runs.

---

## GIL contention investigation (why fix 5 matters)

Before making the above changes, I audited whether the CPU
`ThreadPoolExecutor` pool and the single-lane GPU executor thread in
`knf_core/engine/jobs.py` — which share one process and therefore one GIL —
could explain a real, previously-unexplained anomaly in this repo's own
benchmark data (`nci_compare/postopt_benchmark/`): the same molecule's
`parse_molden` cost went from ~2.8ms (sequential mode) to ~256ms (overlap
mode), and `compute_fields` (GPU compute) went from ~325ms to ~531ms.

Using the actual repository code (`knf_core.nci_torch.molden.parse_molden`,
`geoinit.optimize.relax.relax`) under controlled synthetic and real-code
contention on this machine, with a same-workload thread-vs-process control to
separate GIL contention from plain CPU-core oversubscription:

| Busy workers sharing the GIL | Same workload in separate processes (no GIL sharing) |
|---|---|
| up to **246x** slower (pure-Python busy threads) | ~1-2x (same busy loops, separate processes) |
| up to **23.5x** slower at 4 threads, using the *real* `geoinit.optimize.relax.relax()` scalar path as the "busy worker" | — |

The process-based control ruling out plain core oversubscription as the
dominant cause, combined with `compute_wbo_from_molden_details` and
`run_nci_torch` calling the identical `parse_molden` function from two
different threads for different molecules at overlapping times, is what
motivated fix 5 above — it's the one concrete, low-risk change available
*right now* that removes a proven contention source without needing to
touch the batch scheduler's concurrency model itself.

---

## What was intentionally *not* changed in this pass

Identified as further improvements but deferred — each has a real risk or
verification gap that shouldn't be papered over:

- **CPU pre-nci pool: `ThreadPoolExecutor` → `ProcessPoolExecutor`.** This is
  now the highest-confidence, highest-magnitude remaining item — the GIL
  investigation above measured the *mechanism* directly (thread-based
  contention up to 246x worse than process-based contention at matched
  load), and it would address the CPU-pool-vs-GPU-lane contention broadly,
  not just the one specific `parse_molden` double-call fixed in item 5. Still
  **not done in this pass**, deliberately: it requires redesigning
  `knf_core/engine/jobs.py`'s `progress_callback` (currently a closure
  capturing `lambda: completed`, not picklable across a process boundary —
  likely needs a `multiprocessing.Queue`-based event channel instead),
  auditing that no GPU/CUDA state or thread locks (`GPU_DEVICE_LOCK`, the
  NCI router's internal lock) leak into worker processes, and validating
  against the full pipeline with real xTB/RDKit — none of which is possible
  to do safely and completely in this environment (no RDKit, no xTB binary,
  no GPU). Estimated effort: a bounded, well-understood ~1-2 day refactor,
  not a quick patch. Recommended next step: prototype it on a machine with
  the full dependency stack, validate against `tests/test_engine_regression.py`,
  and re-run the sequential-vs-overlap timing comparison to confirm the
  measured mechanism translates into the predicted real-pipeline win before
  merging.
- **GPU-aware CPU worker sizing.** `autoconfig.resolve_multi_config` sizes
  workers from CPU/RAM only; it has no idea whether the run is GPU-bound.
  Right-sizing this needs live calibration (measure a few molecules' actual
  pre-nci vs. NCI-GPU time on the target machine) rather than a static
  heuristic — deferred until it can be tuned against a real GPU run.
- **Batching multiple molecules through the GPU NCI stage at once**, and
  **replacing the polling two-future-dict loop in `jobs.py` with a blocking
  bounded-queue pipeline.** Both are legitimate further throughput levers
  but are larger structural changes; doing them without a GPU to benchmark
  against risks a regression I can't detect before it ships.
- **Closed-form 3×3 eigenvalue formula** for `_compute_lambda2_batched`
  (replacing the generic batched `torch.linalg.eigvalsh` call, since only
  the middle eigenvalue λ2 is ever used). Plausible win, but touches the
  actual scientific output (λ2's sign feeds directly into the NCI
  descriptor) — needs a dedicated numerical-equivalence test suite
  (including degenerate/near-degenerate eigenvalue cases) before it should
  be considered, which doesn't exist in this repo yet.

If you want to move on any of these next, the `ProcessPoolExecutor` migration
is the highest-leverage item given the evidence gathered this session, but it
should go through the validation steps above — on real hardware, with the
full dependency stack — before merging, not as a blind rewrite.
