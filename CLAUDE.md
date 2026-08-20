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
