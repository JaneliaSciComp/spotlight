# Benchmarking spotlight against BigFlatFieldIlluminator.jl

Everything here runs on the cluster. Nothing needs to be edited first — the scripts read
the experiment's own `LocalPreferences.toml`.

## Run it

```bash
cd <your experiment directory>          # the one holding LocalPreferences.toml
python /path/to/spotlight/bench/make_bench_scripts.py \
    --julia-project /path/to/BigFlatFieldIlluminator.jl

./bsub_bench_stats.sh                   # wait for it to finish (bjobs)
./bsub_bench_correct.sh                 # then this one

python /path/to/spotlight/bench/collect.py bench_results
```

Paste back: the markdown `collect.py` prints, and `bacct -l <jobid>` for both arrays.

## Why one array job instead of two

The element index picks the implementation — even runs Julia, odd runs Python — over the
**same camera and the same chunk range**. Two separate submissions would land on
different hosts, at different times, against a differently-warm page cache; on a
heterogeneous cluster that difference is comparable to the one being measured, and
averaging does not remove it. Interleaving puts each pair in the same window on the same
host pool, so the per-index ratio means something on its own. `collect.py` flags any pair
that still ended up on different hosts rather than averaging it in.

Both arms pin `JULIA_NUM_THREADS` / `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`MKL_NUM_THREADS` to the same value. Both drive the same tensorstore underneath, so
leaving its concurrency at the default would benchmark tensorstore's defaults rather than
the port.

## Why `correct` sweeps a semaphore limit

The two implementations bound their in-flight reads differently, so wall clock alone
misses the interesting result. `apply_correction_chunked` in Julia queues **every**
chunk's read future up front, so a large setup can hold its whole volume at once — that
is the stall the Julia package's supervisor watchdog existed to requeue
around. The port caps
shards in flight with a semaphore.

So the Python arm runs at several limits (4, 16, 64 by default; `--sem-sweep`) and the
deliverable is a peak-RSS-against-wall-time curve rather than a single speedup number.
"Python finishes in 8 GB where Julia needed 60" is a real result and would not appear in
a timing table. If the Julia arm gets killed by the LSF memory limit at large chunk
counts, that is data — record it rather than retrying with a bigger reservation.

## Correctness, for free

```bash
python /path/to/spotlight/bench/compare_outputs.py \
    <julia output store> <python output store> <format> <setup> [setup ...]
```

Exact uint16 comparison, exiting non-zero at the first differing chunk and naming it.
A benchmark run has already read and written everything; this turns it into a real-data
parity check for the cost of one more job — arguably the more valuable half.

## Calibrating concurrency

`memory_budget()` derives the in-flight limit from the LSF allocation (cores × 15 GiB ×
0.5). That derivation is a model; this sweeps the real thing and tells you whether the
model lands near the optimum:

```bash
cd <your experiment directory>
pixi run --manifest-path <spotlight> python <spotlight>/bench/sweep_concurrency.py \
    stats 0 1 64 --values 4,8,18,32,64 --repeats 2 --out sweep.json
```

It runs the stage once per value in a fresh subprocess and prints wall clock, MiB/s and
peak RSS for each, then names the fastest AND the knee — the lowest value within 5% of
it, which is usually the one you want, since past the knee only memory keeps climbing.

Two confounds it handles, because both invent a knee that is not there:

* **Page cache.** The first value read from cold looks slowest whatever it is. Values run
  in shuffled order with repeats by default, so the cache advantage is spread rather than
  landing on one. `--warmup` measures warm-cache behaviour instead — honest, but not what
  a production job sees.
* **Memory.** `--budget` pins `SPOTLIGHT_MEMORY_BYTES` so every value sees the same
  budget and only the concurrency differs. Without it, a value that raises RSS is also
  changing what the derivation would have allowed.

### Running it without bsub

Nothing about the sweep needs LSF. From any machine that can see the data:

```bash
cd <your experiment directory>
pixi run --manifest-path <spotlight> python <spotlight>/bench/sweep_concurrency.py \
    stats 0 1 64 --values 4,8,18,32 --results-root /scratch/sweep
```

Two things change off-cluster:

* **The budget comes from the machine, not an allocation** -- total RAM x
  `SPOTLIGHT_MEMORY_FRACTION`, since there is no `LSB_DJOB_NUMPROC` to scale from. On a
  128 GiB workstation that is 64 GiB. Pin it with `--budget` if you are sharing the box.
* **`--results-root` redirects the output** to scratch, so the sweep does not rewrite the
  results you are comparing against -- it re-runs the stage once per value per repeat.
  It copies `intensity_stats` across so `empty_threshold` is still found; without that
  the pass skips the background profile and does less work than a real run.

A value that gets OOM-killed is not a failure of the sweep — it brackets the ceiling, and
the rest of the table still prints.

The harness is tested (`tests/test_sweep.py`) against a stubbed stage with a known
answer; what it cannot test is the measurement itself, which depends on the filesystem
and the node.
