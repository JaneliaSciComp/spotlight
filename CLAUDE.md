# Working notes for spotlight

Reasoning, measurements and dead ends. The code keeps decisions and pointers; the
derivations live here so the modules stay readable.

## Working agreements

- **Never `git commit` or `push` unless asked in that turn.** The repo owner commits.
- **The cluster pixi env is on a mounted volume** (`/Volumes/schweinfurthl/spotlight`) —
  never run pixi against it. The local `.pixi` is on the internal disk and is fine.
- `python` is not on the local PATH. Use `pixi run python`, or `/usr/bin/python3` for
  throwaway scripts that must not import the package.
- macOS has no `timeout(1)`.

## The cluster contract: load ≤ 2× reserved slots

Scicomp measures a host's 1-minute load average against the slots LSF *reserved*, not
against the host's physical cores. This is the constraint that shapes every thread pool
here.

The part that isn't obvious: **Linux load counts threads that are runnable *or* in
uninterruptible sleep.** A thread parked in an NFS read is charged exactly like one
burning a core. There is no free I/O concurrency, which is why an idle-looking pool of
1920 file-I/O threads is not free.

`stores.slots()` is the single definition of the denominator. Every pool sizes from it —
the numbers only add up if they all measure against the same one. A test greps the
package to keep anything from reading `LSB_DJOB_NUMPROC` on its own.

### Why `data_copy = slots` but `file_io = slots // 2`

`bench/sweep_threads.py correct 0` at 64 slots. "peak busy" is threads runnable or in
uninterruptible sleep — the job's own contribution to the load average, sampled from
`/proc/<pid>/task/*/stat` rather than `/proc/loadavg` so neighbours don't pollute it.

| file_io | data_copy | peak busy | / slots | wall |
|---|---|---|---|---|
| `slots*64` | `slots` | 1157 | 18.1× | 44.1 s |
| `slots*16` | `slots/2` | 1045 | 16.3× | 53.7 s |
| `slots*4` | `slots/2` | 295 | 4.6× | 50.4 s |
| `slots/2` | `slots/2` | 71 | 1.1× | 49.4 s |
| `slots` | `slots/2` | 102 | 1.6× | 48.6 s |
| **`slots/2`** | **`slots`** | **104** | **1.6×** | **44.6 s** ← shipped |
| `slots` | `slots` | 138 | 2.2× | 44.3 s (over) |

The two pools are not interchangeable, and only measuring separated them:

- **`file_io_concurrency` buys no throughput at all.** 32 → 64 → 256 → 1024 went 49.4 →
  48.6 → 50.4 → 53.7 s — flat, then worse — while costing load the whole way. The
  `slots*64` default this replaced was not paying for itself even before anyone mentioned
  load: at 30 slots it was 1920 threads, and one correction element peaked at **2165**,
  72× its allocation. Half the slots keeps an unsharded n5 source's opens overlapping.
- **`data_copy_concurrency` is where the time was.** zstd decode and encode — real CPU
  work on cores the job reserved. Halving it cost 11% wall clock outright.

Worst-case sum is 2.5× slots; measured peak is 1.6×, because a read, its decode and the
kernel's pass over the result don't peak together. **The bound is not the metric — measure.**

Consequence worth stating to anyone tuning this: **concurrency is bought with slots.** A
stage that wants more reads in flight raises its `bsub -n` (`n_cores_*`), not a multiplier.

### Still unmeasured

The **stats** stage. It drops from 192 in-flight opens to 2 at 3 slots, and it's the stage
the old `slots*64` was tuned for, holding a twentieth of the slots — so it will cost more
than the table above. `bench/sweep_threads.py stats <camera> <start> <stop>`.

The **basic** stage, whose BaSiC fit is an SVD — the one place BLAS threads would be
runnable rather than sleeping. See below.

## Dead ends — measured and rejected, don't redo these

### BLAS/OpenMP thread caps in `scripts.runner()`

`OMP_NUM_THREADS` and friends default to the **host's** logical core count, not the LSF
allocation, so a 30-slot element on a 128-core host opens 128 BLAS threads before doing any
work. Verified locally: one `np.linalg.svd` on a 16-core box takes the process from 1
thread to 16, and the caps take it to exactly the cap. That's also what put 187 threads in
the resource summary of array elements that died inside `load_config()` having read
*nothing*.

It looks like an obvious fix and it isn't. Those threads sit in **interruptible** sleep,
which the load average ignores. Measured on `correct` at 64 slots: peak busy 1157 uncapped
against **1245 capped** — worse, inside the noise — while the tensorstore pool split alone
took the same stage to 70. The caps buy ~64 fewer idle threads in `ps` and nothing else.

`bench/sweep_threads.py` no longer carries the arm. Re-add it before re-adding the vars,
and measure `basic`, not `correct` — an env var in that string is charged to every stage.

### `SHARDS_IN_FLIGHT` cap on the correction loop

Added and removed in the same session. `_concurrency` bounds in-flight shards by memory
only, so a 450 GB allocation permits all 27 shards of a tile at once. That produces no
partial progress — every shard starts together and none finishes until they all do, so the
loop reports `0/27` for its entire duration whether it's healthy or wedged. **`0/27` after
21 minutes is not evidence of a hang**; I claimed it was and was wrong. Left as-is by
request.

### A slab loop for the pyramid write

Also added and reverted. `await lvl.with_transaction(ltxn).write(ds)` buffers a whole level
(17.5 GB at level 1 of a 3344×2560×4096 tile) before commit, and I additionally asserted a
`data_copy_concurrency` deadlock I never demonstrated. The stall was in the level-0 shard
loop, not here. `tests/test_local.py::test_the_output_pyramid_matches_a_numpy_downsample`
survives the revert and is the only coverage this path has ever had — the default test
store has no level 1, so the loop never ran in tests.

### `MALLOC_ARENA_MAX=4` — this one stayed

The exception that proves the rule about measuring. glibc gives each thread its own malloc
arena and freed memory stays resident. Stats stage, both arms on one host: peak RSS
**49.1 GB uncapped against 5.8 GB at 4**, and the capped arm was slightly *faster* (210 s
vs 227 s). Must be in the environment ahead of the process — glibc reads it when it creates
the first arena, long before `__main__`.

## One script for a cluster run: `run <pipeline> --cluster`

`python -m spotlight run both --cluster` writes `bsub_pipeline_both.sh`, which submits every
stage and chains each on the previous stage's **job ids**. One `bsub` per stage, run once,
walk away. The three-scripts-and-wait workflow is still there for rerunning a single stage.

`local.PIPELINES` is the shared definition — `write_pipeline_script` imports `_plan`,
`_CORRECT_MODE` and `apply_basic_for` from `local` rather than restating them, so `run both`
and `run both --cluster` cannot drift on stage order or on which correction `correct`
applies. `--start-at` / `--stop-after` work on both, which is also the resume mechanism:
regenerate from the stage that failed.

### IDs, not names — that is why this works now

`create_intensity_correction_script`'s docstring has always said name-based `-w` proved
unreliable here, and that is why the intensity stages were three files with "wait for each
one" in prose. ID-based dependencies are a different mechanism: `bsub` prints `Job <12345>
is submitted to queue <normal>.`, the script captures the number, and the next stage gets
`-w "done(12345)"`.

Two things this makes load-bearing:

- **The `-w` expression must be DOUBLE-quoted, not `shlex.quote`d.** It holds `&&` and
  parentheses (needs quoting) *and* `$J_STATS_0` (needs expanding). Single quotes submit the
  literal `done($J_STATS_0)`, which LSF accepts and never satisfies. Caught while writing it,
  not by a test — the test exists now.
- **An unparseable job id has to kill the script.** An empty variable makes the next
  `-w "done()"` another expression LSF accepts and never satisfies, so one failed `bsub`
  would silently detach every stage after it. `jsub` exits 1 instead; verified by running the
  generated shell against a stub `bsub` that fails mid-chain (2 of 3 submitted, exit 1).

Requeue interacts *correctly*: `-Q 140` returns an element to PEND under the same job id, so
`done(<id>)` is not satisfied until it finally finishes.

### `done()` and not `ended()`

`done()` requires success; `ended()` fires either way. Chosen `done()` because a stage that
starts on a partial previous stage does not fail — it produces a quietly wrong answer.
`int-aggregate` will solve a target from 195 of 196 tiles and report `n_present` without
complaint, and that target then rescales every tile.

The price is real and worth knowing: one genuinely failed element parks everything downstream
in PEND with `Dependency condition never satisfied` (`bjobs -l <id> | grep -i depend`). The
generated script's header says so. It is a one-word edit in a generated file for anyone who
wants the other trade — which is the reason this generates a script instead of driving LSF
from Python.

### `emptiness` is measured at generation time, not submitted

It is first in `PIPELINES` and absent from the chain. `_stats_prep`'s fingerprint decides
whether a camera's finished background-quantile partials survive, it is computed in the
submitting process, and it reads `empty_threshold` — so the measurement has to be on disk
before anything goes out. `ensure_emptiness` skips the rescan when it already is. The
consequence is that `--cluster` does real work on the submit host the first time.

### `qstack` and `basic` get 8x the run limit

Both were run by hand before this existed, so neither has ever been measured under LSF, and
each processes a whole camera per element rather than one chunk range or one tile. The 60 min
default is ~17x the apply stage's worst case; against a whole-camera BaSiC fit it could be
*shorter* than a healthy run.

The asymmetry is what forces a generous value: overshooting costs a wedged host a few idle
hours, undershooting costs an **infinite requeue loop** — `-Q 140` has no job-level retry cap
— and in a chain nothing downstream ever starts. `WHOLE_CAMERA_RUNLIMIT = 8`; replace it with
a measurement when there is one. `basic` is one array element per camera, so the fits run
concurrently.

### `SPOTLIGHT_APPLY_BASIC`, because each stage is now its own process

`local.run_pipeline` overrides `apply_basic` in-process from the pipeline NAME. On the cluster
the stages are separate processes that would each auto-detect from whether the fields exist,
and the detection is wrong in exactly the case that prompted this: `--cluster intensity` on an
experiment with leftover fields detects True, measures all 196 tiles from flat-fielded voxels,
and only fails hours later in `correct._check_basic_mode`.

So the generated script states it in the environment on the two stages that read the flag
(`int-stats`, `int-aggregate`); `correct` sets its own from `--mode`. The env var **beats**
both the toml and the detection, so that a cluster run and a local run of the same pipeline
name reach the same answer.

Note this is not a bug in the `both` chain even without the env var — `basic` runs before
`int-stats`, so the fields exist by the time detection runs. It is `intensity` that needs it.

## A hung element is usually a hung HOST, not a hung code path

`mouse_hipp_3_channel`, job `153471857` (`int-apply`, `[3-6,65-560]`, 30 slots each).
Three elements — 144, 409, 419 — ran 12–14 h and were killed by hand. All 500 elements
that reported are in one of two states, with nothing in between:

| | elements | host(s) | run time | CPU time | peak RSS |
|---|---|---|---|---|---|
| healthy | 497 | 40 hosts | 99–209 s | ~2050 s | ~100 GB |
| hung | 3 | **e10u18, all three** | 42828–49522 s | 116–133 s | 0.8 / 0.8 / 6.2 GB |

`e10u18` ran exactly three elements and all three hung; it never completed one. Element
145 was dispatched in the same wave as 144 — 20:02:16, same stage, same dataset, one host
over — and finished in 158 s.

The hung elements did **1/15th of a healthy element's CPU work over 300× the wall clock**
and never allocated the ~100 GB working set, so they were blocked, not spinning or
thrashing. They also wedged at three *different* places, which is what rules out a code
path:

- **144** — no stdout at all, and no `KeyboardInterrupt` traceback on `bkill` either. It
  could not take a signal: at least one thread was in uninterruptible sleep.
- **409** — `fields.basic_model` → `stores.source_pyramid_shapes`, blocked in tensorstore
  opening the *source*.
- **419** — reached the shard loop (`0/27 in 11:53:45`) and its event loop was idle in
  `selector.select()`, waiting on a future that never resolved.

One dataset, one stage, three unrelated I/O sites, one host. The host's mount was gone.

Consequence: **there is no code fix for this, only a bound and a retry.** `scripts._bsub`
now emits `-W {lsf_runlimit_minutes} -Q "140"` on every stage: `-W` kills the wedged
element, `-Q` sends it back to PEND instead of leaving a hole in the output. Each is
useless alone — `-W` by itself loses the tile, `-Q` by itself never fires — which is why
`_watchdog()` emits them together or not at all.

`-W` is not a runtime estimate. Set it well above the slowest healthy element: 60 min is
~17× the 209 s worst case here. Raise it for a larger tile before assuming the ceiling is
the bug, and note there is **no job-level cap on requeue count** — `MAX_JOB_REQUEUE` is
`lsb.queues` / `lsb.applications` only — so an element that legitimately exceeds `-W`
requeues forever. A generous `-W` is the only guard against that.

### Why the requeue list is `140`, and only `140`

140 is the run-limit kill: LSF sends **SIGUSR2** on `-W` expiry, and 128+12 = 140. Python
installs no SIGUSR2 handler, so the default disposition (terminate) applies and the code
reaches LSF unmodified.

That is not true of the signals `bkill` sends. **Every `TERM_OWNER` kill in this
experiment's logs reported exit code 1** (12 of 13; the other reported 130), because Python
turns SIGINT into a `KeyboardInterrupt` traceback and exits 1 — the same 1 the TOML-BOM and
output-shape failures exited with. So requeueing on 1 would retry real errors forever, and
the exit code is only a reliable hang signal for the run-limit path.

Two forms deliberately **not** used, both in `scripts.REQUEUE_EXIT_CODES` if wanted:

- **`EXCLUDE(140)`** — exclusive requeue, the form that keeps the retry off the host that
  just wedged. LSF documents it as **not working for parallel jobs**, and every stage here
  is `-n > 1`. So a requeue can land back on the sick host and burn another `-W`; that is
  still better than a hand-edited index list, but it is not host avoidance. Valid for a
  1-slot stage.
- **`137`** — SIGKILL, which LSF sends ~10 min after SIGUSR2 if the job has not died.
  Element 144 is the reason to consider it: it could not take a signal at all, so it may
  never produce a 140. Left off because 137 is also a `bkill -s 9` and a memory-limit kill,
  and requeueing an OOM forever is worse than one manual resubmit.

The residual limit is element 144's: a thread in uninterruptible sleep cannot be signalled,
so `-W` bounds when LSF *gives up on* the element, not always when the process dies.

Diagnosing the next one: the missing-index trick tells you *which* elements are stuck, and
then **group them by execution host** — if they share one, stop reading the code.

```sh
for f in output/_ic_*.txt; do awk -v F="$f" '
  /^Sender: LSF System/ {b=0} /Subject: Job <JOBID>\[/ {b=1}
  b && /was executed on host/ {match($0,/[a-z][0-9]+u[0-9]+>/); h=substr($0,RSTART,RLENGTH-1)}
  b && /Run time :/ {print h, $4, F; b=0}' "$f"; done | sort -k2 -rn | head
```

Still unverified: that this program actually reports 140 under `-W` rather than converting
SIGUSR2 into something else. It cannot be checked off-cluster, and the 12 exit-1 `bkill`
results above are the reason to check it rather than assume — submit one element with
`-W 1` and read the reported code before trusting the requeue on a full array.

## Writing to a macOS mount: `file_io_locking`, not the obvious suspects

Local run against `//prfs.hhmi.org/tavakolilab` (smbfs, mounted at `/Volumes/tavakolilab`):

```
ValueError: UNKNOWN: Error opening "n5" driver: Error writing local file
".../spotlight/camera1/minima/s0/attributes.json": Failed to close file descriptor:
Input/output error [os_error_code='5'] [tensorstore/internal/os/file_descriptor.cc:66]
```

The message names `close()` and EIO and nothing else, so it reads like bad hardware or a
dropped SMB session. It is neither: it is **tensorstore's default `file_io_locking` mode
(`os`), which takes an OS-level lock per file that smbfs does not honour.** `close()` is
just where the failure surfaces.

Matrix, n5 uint16 640×640 zstd on that share, 2 runs per cell:

| `file_io_locking.mode` | `file_io_sync=true` (default) | `false` |
|---|---|---|
| **`os`** (default) | **EIO, EIO** | ok, ok |
| `lockfile` | ok, ok | ok, ok |
| `none` | ok, ok | ok, ok |
| `non_atomic` | ok, ok | ok, ok |

Confirmed at the real 1920×1920 (900 blocks), write + read-back compared: default 3/3 EIO,
`lockfile` 3/3 identical. Valid modes are `os`, `lockfile`, `none`, `non_atomic` — anything
else is rejected at `ts.Context` construction, which is the cheap way to enumerate them.

`stores._locking_mode()` returns `lockfile` when `sys.platform == "darwin"` **and** the cwd
resolves under `/Volumes/`; `SPOTLIGHT_IO_LOCKING` overrides either way. Linux resolves to
`None` by construction, so `/nrs` keeps the `os` locking it has been measured with — the
cluster path is untouched, which is the point. `lockfile` keeps `file_io_sync` on, so this
costs a sidecar lock file per key and no durability.

### The three wrong turns, so nobody repeats them

Each looked like the answer and each is disproved by a probe in this file's history:

- **`fsync`/`F_FULLFSYNC` on smbfs.** The first hypothesis, because `file_io_sync=true` is
  right there in the echoed spec. Probed directly: `fsync` and `F_FULLFSYNC` both return ok
  on that share, on a file fd *and* a directory fd. `file_io_sync=false` does mask the
  metadata failure, which is what makes this hypothesis so convincing — but it only defers
  the error to the chunk writes.
- **Concurrency.** `file_io_concurrency` 8/4/2/1 with sync off gave fail/fail/**ok**/fail —
  not monotonic, so not a limit to tune. Non-monotonic results mean you are looking at the
  wrong variable, not at a threshold.
- **The write pattern.** 900 concurrent small-file writes by hand — plain, and temp-file +
  rename + nested dirs in both close/rename orders — produced **0 errors in 1800 files**.
  The pattern is fine; only tensorstore's locking is not.

`attributes.json` was already on disk, complete and valid, after the failure. A close-time
EIO here does not mean the bytes are missing, so "the file is there" is not evidence the
write succeeded.

## Windows: local pipeline only, and the separator is the whole port

`win-64` is in `pixi.toml`'s `platforms` and resolves from the same lock — same numpy
1.26.4 / scipy 1.12.0 as the other two, only the build strings differ, so the pocketfft
argument for one lock still holds. Adding it changed nothing for `linux-64` / `osx-arm64`;
the lock diff was pure insertion.

**The cluster is always Linux.** Nothing under `scripts.py` (bsub, `runner()`,
`MALLOC_ARENA_MAX`) or `bench/` has to work on Windows, which is what makes this small.
`tests/test_bench_scripts.py` is `skipif(os.name == "nt")` for that reason — it drives
`bash` and a chmod'd stub, and there is nothing there to port.

### Normalise the ROOTS, not the ten kvstore paths

Every path this package hands tensorstore's `file` driver is a `/`-joined suffix on a
configured root (`formats._path`, `stores.write_group_metadata`,
`qstack.read_quantile_stack`, …), so a root of `C:\data\exp` produces the mixed
`C:\data\exp/setup0/timepoint0/s0`. Python's `open()` does not care; the driver has to
parse the key. Forward slashes are what both platforms accept.

So the fix is `config._slashes` inside `expand()` — one function, covering all four
*_path keys plus `results_root`, and a no-op wherever `os.sep != "\\"` because `\` is a
legal filename character on Linux and macOS.

Two things that are deliberate:

- **Not `Path.as_posix()` on the roots.** It resolves nothing here and would collapse a
  UNC root's leading `\\server\share` to a single separator. A plain `.replace()` keeps it
  as `//server/share`.
- **`stores.open_stats_array` is the one exception** and uses `.as_posix()`, because
  `stats_array_path` builds its path by `Path` joining rather than by `/`-joining a root,
  so `_slashes` never sees it.

That exception is only *testable* from a POSIX host via `monkeypatch.setattr(stores,
"Path", PureWindowsPath)` — `Path` is `PosixPath` here, so it cannot produce a backslash
to catch. Without that the mutation `.as_posix()` → `str()` survives, which is how it was
found.

### `_machine_memory()` had a silent Windows floor

`os.sysconf` does not exist on Windows, so every local Windows run fell to the 8 GiB
fallback and sized `_concurrency` against a 4 GiB budget — on any workstation, whatever it
has. Now falls through to `kernel32.GetPhysicallyInstalledSystemMemory` (ctypes, no
psutil) before the 8 GiB default, which stays for the VMs where that SMBIOS read fails.

### Unverified on Windows

Nothing here has been run on Windows — it is a port by inspection, and these are the
places to look first if it misbehaves:

- **`_locking_mode()` returns `None` on Windows**, i.e. tensorstore's default `os`
  locking, which is `LockFileEx`. That is right for local NTFS. A mapped drive or a UNC
  path to an SMB server is the case that broke on macOS smbfs, and it is untested here —
  if a write dies on close, try `SPOTLIGHT_IO_LOCKING=lockfile` before reading any code.
  Deliberately not extended to Windows on a guess: the darwin branch is there because it
  was measured 3/3 against 3/3, and a second unmeasured branch would dilute that.
- **The 260-character path limit.** Zarr chunk keys are deep. Python 3.6+ ships the
  `longPathAware` manifest, so this only bites without the `LongPathsEnabled` policy.
- **`os.replace` in `_atomic_write_json`** fails on Windows if the destination is open in
  another process. The local driver is single-process, so this is only a risk if someone
  has the store open in a viewer.

Checked and genuinely fine, so don't re-audit them: asyncio + `ThreadPoolExecutor`,
`progress.py` (`\r` and ASCII only), `os.chmod(0o755)` (a harmless near-no-op),
`$HOME` via `expanduser`.

## `apply_basic` is about READING, not about whether BaSiC gets applied

`run basic` prints `apply_basic=False` and then applies BaSiC. Both are correct; the flag
answers a different question than its name suggests — *do the stages read BaSiC-corrected
voxels?* — and the `correct` stage sets its own answer regardless:

| pipeline | mode | pipeline-level | what `correct` uses |
|---|---|---|---|
| `basic` | `basic` | False | **True** (`correct._view`, unconditional) |
| `intensity` | `intensity` | False | False |
| `both` | `both` | True | True |

False is not merely harmless for the `basic` pipeline, it is required: `stats` and `qstack`
build the qstacks the `basic` stage **fits the fields from**, so reading flat-fielded voxels
there would fit a flat field to already-flat-fielded data. And in that pipeline only
`correct` consults the flag at all — `emptiness`, `stats`, `qstack` and `basic` never read
`cfg["apply_basic"]`. The consumers are `int-stats`, `int-aggregate`, and
`fields.basic_model` via `correct`.

So the banner said "for every stage" about a flag four of five stages ignore and the fifth
overrides. It now names `mode`, which is what actually decides. Keep the substring
`apply_basic=<bool>` — `test_run_pipeline_overrides_a_stale_autodetected_value` greps it —
and keep it off the `pipeline:` line, which `test_stage_windows` parses by stage name.

## Gotchas that cost real time

- **Stale `.pyc` on same-size edits.** Mutation-testing `"n_cores_int_correct"` →
  `"n_cores_nonexistent"` (both 19 chars) left Python reusing the cached bytecode, because
  the cache validates on (mtime, size) and coarse mtime hid the change. A "CAUGHT" result
  was a false positive. Clear `__pycache__` between same-length source mutations.
- **`ru_maxrss` for `RUSAGE_CHILDREN` is a monotonic high-water mark** across every child
  ever waited on. Using `after - before` per arm reported the first arm's peak and then
  `0.0` for all the rest, which reads as "this arm used no memory". Read the child's own
  `VmHWM` from `/proc/<pid>/status` while it's alive.
- **`/proc/<pid>/task/<tid>/stat` field 2 is the executable name, unescaped** — it can
  contain spaces and parentheses, so `split()[2]` picks the wrong field for exactly the
  processes whose names are interesting. Parse the state char after the *last* `)`.
- **`LSB_DJOB_NUMPROC` unset is silent and poisonous.** A benchmark run outside LSF took
  `slots()`'s off-cluster default of 8 on a 64-core node and reported ratios against a
  ceiling of 16 that corresponded to no reservation. Anything whose whole output is a ratio
  against the allocation should refuse to guess the denominator.
- **`_load_toml_config` reads `Path.cwd()/LocalPreferences.toml`**, so every stage must run
  from the experiment directory. `Invalid statement (at line 1, column 1)` on a file that
  looks correct is characteristically a UTF-8 BOM — `tomllib` doesn't strip it.
- **Page cache warms across benchmark arms**, so later arms look faster on any source small
  enough to fit. `--shuffle` spreads it; irrelevant for a 70 GiB correction input.
- **LSF writes job logs at exit**, so a hung array element has no log at all. Diagnose by
  which indices are *missing* from the output directory, not by reading them.

## Testing discipline

- **Mutation-check every new test.** Apply the one-line source change the test claims to
  catch and confirm it fails. Six of seven survived this on the concurrency work; the
  seventh revealed an untested call site (below).
- **Don't assert that code is absent.** A test whose body is "the thing we removed is still
  removed" tests nothing that could break. Record rejected approaches here instead —
  that's what this file is for.
- **Pin measurements, not arithmetic.** `test_the_two_pools_are_sized_from_the_reservation_and_differ`
  exists because the asymmetry is a measurement; making the two limits symmetric again is a
  regression that would read like tidying.
- **Membership over literals for operator choices.** `tile_threshold`'s default moved
  `otsu` → `li` and left a test asserting the old literal. The invariant worth pinning is
  that the default is *in* `THRESHOLD_MODES` — a default outside it breaks every stage that
  reads it.

### Known coverage gap

`correct.py`'s two `stores.slots(_config.stage_cores(view, mode))` call sites are not
covered — reverting them to a literal `8` passes the whole suite. The failure mode is an
off-cluster local run building an 8-thread pool instead of 16–20, i.e. a performance nit,
not a correctness bug. Not worth a source-grep test.

## `spotfix`: repairing a local dimming defect in one already-corrected tile

A tile sometimes goes dark over part of its volume while the tiles overlapping it hold real
signal there. Neither flat/dark nor the per-tile gain can fix that — both are corrections a
tile applies to *itself*, and the evidence that something is missing lives in its
**neighbours**. So the expectation comes from the overlapping tiles resampled into this
tile's grid, never from a model of the tile itself. That one choice is what makes the stage
safe on dark anatomy: a genuinely dark structure is dark in the neighbours too, so nothing
is demanded of it.

    python -m spotlight run spotfix 126 158        # local, one tile at a time

Reads and writes `output_intensity_path` — the corrected dataset — because it repairs a
correction run's output. The previous version of the tile is **renamed** aside
(`s126-t0.zarr.prespotfix`), never deleted and never overwritten in place: a rename is
atomic on one filesystem, costs nothing for a 70 GB tile, leaves the old pyramid whole if
the run dies, and deleting is the operation that parks indefinitely on an smbfs mount.
Clearing old backups is left to a human who can decide how many to keep.

The rule, in order:

1. **expectation** — neighbour value where covered (a measurement, so it wins); else that
   z slice's neighbour level; else the column's own healthy-depth plateau, but only where
   the tile still shows signal. The plateau case exists because the slice level falls below
   what a column's plateau says when the specimen ended inside the covered strips but
   persists here; requiring signal stops a column at background demanding a 25× lift.
2. **gate** — the neighbours locally show specimen (`local_presence`) **OR** the tile has
   its own signal (`signal_floor`), each with a deadband. Both branches are load-bearing:
   dropping presence collapsed the two largest fills from 172/117 DN to 11/14.
3. **mask** — `r < EDGE_R` where `r = obs/expected`. That is the whole detection rule.
4. **gain** — `1 + (mask × gate) × (expected/obs − 1)`, floored at 1. The gate decides
   *whether*, `need` decides *how much*; folding them under-corrects every moderately
   attenuated cell. A non-finite `need` falls back to 1.0, which is what keeps a tile's
   unimaged z padding (`obs == 0`) from being amplified out of nothing.
5. **despeckle** — median filter on the coarse gain. It does two jobs: removes isolated
   cells *and* smooths gain magnitudes near the boundary, i.e. it contributes to the
   feather. Moving it to binary morphology on the mask took the boundary step from 9.3% to
   22.5% on one tile and 90.6% on another — morphology can move a hard edge, never soften
   one.
6. **apply** — trilinear to the output level, damped by the contrast weight, `np.rint`.

The contrast weight is not optional and was missing from the first packaged version, which
is why that run looked worse than the prototype on both tiles: every dark structure inside
the corrected region took the full gain instead of being damped. An anatomical hole and an
attenuated region are indistinguishable pointwise — measured, 7 DN either way — and only the
ratio to their surroundings separates them (local mean 191 for the hole, 7.8 for the
attenuation). Its numerator must be per-voxel; its denominator is a `spotfix_contrast_um`
lateral mean, computed at the ANALYSIS level and upsampled, because at level 0 the
full-resolution version needs the whole volume as float (140 GB) or a 384-voxel halo per
shard (~3x the reads). Validated against the prototype's per-voxel version: hole 0.0000 vs
0.0000, attenuation 0.9701 vs 0.9631, correlation 0.99913.

Two bugs it hit on the way in, both caught by `test_a_dark_structure_inside_the_defect_*`:
the step being absent altogether, and then the local mean being named `loc`, which
`local_presence` reassigns twenty lines later — leaving the weight present but identically
1.0, i.e. inert. Shadowing is worth watching for in `gain_field`: it holds five different
per-cell fields and they all want short names.

### `EDGE_R` is a feathering width, not a detection threshold

The mask ends where `r == EDGE_R`, so the gain just inside is `1/EDGE_R` and just outside is
exactly 1: **`EDGE_R` sets the gain cliff at the edge of the correction.** Measured median
boundary step against the owner's judgement:

| `EDGE_R` | step | verdict |
|---|---|---|
| 0.75 | 70–105% | rolling dark band, obvious |
| 0.90 | 17–22% | distinguishable from 0.95 |
| 0.95 | 9–11% | indistinguishable from 0.985 — **shipped** |
| 0.985 | 4–7% | fine |

So the visibility threshold is a ~10–20% step. Growing until the deficit is only a few
percent makes the correction *fade to nothing at its own edge*, which is the whole point.
Judged as a hypothesis test it admits ~27% of healthy cells — irrelevant, that **is** the
feather. Tuning it as a detection threshold walks straight back into the dark band.

The config key is `spotfix_edge_step`, the discontinuity itself, because that is
dimensionless and means the same on any dataset.

### Every parameter is a length in microns or a count of noise sigma

Voxel counts and ratios-of-background do not transfer. A 4:1 z:lateral experiment and a
6.4:1 one need different bin factors for the same physical cell, and the same
`core_r = 0.60` was 11.1σ below the healthy median on one tile and 6.9σ on another.

* lengths — `spotfix_cell_um`, `spotfix_smooth_z_um` / `_lat_um`, `spotfix_presence_um`,
  `spotfix_contrast_um`. Converted with the voxel size read from `<voxelSize>` × the level's
  OME multiscales scale. On the mouse experiment `spotfix_smooth_z_um = 22.6` gives 9 z
  cells; on the fly VNC one it gives 6, because a z cell there is 4.0 µm not 2.512.
* noise — `spotfix_floor_sigma` (the signal-floor ramp top, `bg + k·bg_std`). `3*bg` was
  4.0σ on one tile and 3.1σ on another.
* dimensionless — `spotfix_edge_step`, `spotfix_loc_t`, `spotfix_floor_t`.

The despeckle footprint is deliberately **anisotropic** (22.6 µm in z against 60.3 µm
laterally) and the docstring that claimed those "roughly match" was wrong — in physical
units they differ 2.7×. Sweeping it confirmed the values: more z smoothing destroys the
large fills (a 117 DN fill collapses to 47), less lateral smoothing re-brightens sites that
must stay dark (7 → 47 DN). It is anisotropic for the same reason z is unbinned — the defect
changes fast along z and slowly laterally.

### Scope: local dimming only, and the stage checks it

The input must already be flat-field and intensity corrected. A tile uniformly a few percent
dim is a per-tile **gain** error; correcting it here smears it into a large spatially varying
correction. Measured: a uniform 0.93× takes a tile from 12% to 29% of its grid masked.

`_precondition` therefore measures the tile's level against its neighbours where healthy and
**refuses** outside `±0.5 × (1 − EDGE_R)`. The bound is tied to the feather width because a
global offset that size drops half the tile below the threshold on its own, at which point
detection cannot separate local from global. Mouse tiles measure 1.0042 and 1.0155 (fine);
one fly VNC tile measured 0.9527 with a 5% feather — the offset *equalled* the feather.

### Dead ends — measured and rejected, don't redo these

* **Seeds, cores, connected-component propagation.** `spotfix` used to grow the mask from
  clearly-dead "core" cells through the deficit field. The plain threshold `r < EDGE_R`
  matches it on both tiles — all 14 labelled sites identical, both population medians to
  four decimals, the anatomical hole preserved — with a ~9% larger mask that the despeckle
  absorbs. `core_r` moved the mask by at most ±9% over a **19× range** and could never
  constrain anyway: connectivity lets one dead cell validate an arbitrarily large component.
* **A per-voxel nearest-neighbour gain ceiling.** Real bleed (a 1.15× cell receives 15.6×
  from trilinear interpolation between two 30× cells) but it *was* the visible 16-voxel
  squares — boundary step ratio 4.4–5.7 against 1.1 without it. No smooth version can
  exist: the gain is already clamped to each cell's `need`, and trilinear interpolation
  preserves that, so a trilinear ceiling is provably a no-op.
* **A gain slope cap** (gain may rise at most *r*× per cell). Fixes the too-bright site
  exactly and strangles the legitimate large fills, 189 → 40 DN. A slope bound cannot tell
  "next to healthy because the data cuts off" from "next to healthy because of bleed".
* **A spatially local expectation** (`expectation_3d`, fill an uncovered cell from the
  nearest evidence rather than the z-slice median). Does **not** make `local_presence`
  redundant, and makes judged sites worse on its own. The expectation answers "what level
  should be here"; presence answers "is there specimen here at all", and a
  coverage-normalised fill will interpolate a bright level into a region where the specimen
  has ended.
* **Physically-motivated seeding schemes** — a shadow constrained to run inward from the
  sheet-entry edge, per-(z,x) permission pooling along the sheet, per-column axial deficit.
  All three scored worse than the plain threshold. Four such reformulations lost in total;
  every simplification that *worked* was one where equivalence could be proved or measured,
  not argued from physics.

### Two metrics that are biased, in opposite directions

Neither should be tuned against.

* **Restricted to the voxels two variants disagree about** — favours MORE correction by
  construction, because the comparison set is chosen to contain the correction.
* **The median over all covered voxels** — favours LESS correction, because lifting a
  deficient population up to *match* its neighbour drags the population median rightward
  with no voxel overshooting. This is why the variant that corrected least once scored
  "best".

Split covered voxels by their state BEFORE correction instead (healthy / deficient /
already-brighter) and ask whether each moved. That is the framing that answers "did this
over-brighten healthy tissue", and under it the shipped config lifts already-healthy voxels
by ~0.6%.

### Neighbours must match channel, not merely overlap

`neighbours()` selects on world-space bounding-box overlap AND on matching
channel/angle/illumination. Overlap alone admits the same tile in another channel — it
occupies exactly the same space — which is a different fluorophore and so not an independent
measurement of this signal. It also fails outright, because in this dataset the
non-channel-0 setups carry only a single-scale `raw/0`:

    NOT_FOUND: ... s672-t0.zarr/4/zarr.json does not exist

on a run that asked for tile 126. A file with no `<attributes>` block behaves as
single-channel rather than matching nothing.

There is no per-neighbour fallback for a missing analysis level: same-channel tiles share a
pyramid, so `spotfix_level` is clamped once against the tile's own level count
(`min(spotfix_level, n_levels - 1)`) and the chosen level is printed.

### Not verified

The stage is tested end to end on a synthetic two-tile store (`tests/test_spotfix.py`), with
every test mutation-checked. What has **not** been compared voxel-for-voxel against the
prototype is a real tile. `_levels_in`, the pyramid rebuild, `write_group_metadata` and the
backup/restore path have only ever seen 16^3 arrays, and `spotfix_contrast_um` converts to a
768-voxel window at level 0 where the prototype only ever ran it at 96 — the conversion is
what makes the parameter portable, but that scale is untested.
