# spotlight

Illumination correction for large microscopy datasets — flat-field/dark-field estimation
(BaSiC) and per-tile intensity matching, over OME-Zarr v2/v3 (sharded and unsharded) and
N5 stores, driven as LSF array jobs.

## Install

```bash
pixi install                 # both linux-64 and osx-arm64 resolve from one lock
```

No `pip install` step: the package is used from the checkout, and the generated `bsub`
scripts set `PYTHONPATH` themselves.

## Configure

Settings live in `LocalPreferences.toml` in each experiment's own directory, under a
`[spotlight]` table (and `[spotlight.basic]` for the BaSiC parameters). Every stage reads
it from its own working directory, so an experiment is selected by `cd`-ing into it — and
the LSF jobs must run from the same directory the scripts were generated in.

A pre-rename `[BigFlatFieldIlluminator]` table is still read, so existing experiment
directories work untouched; the first `set_config` call folds it into `[spotlight]`.

```python
import spotlight as sp

sp.set_config(
    input_basic_path  = "/path/to/data/dataset.ome.zarr",   # the store BaSiC reads
    output_basic_path = "/path/to/basic_corrected.ome.zarr",# the store `correct` writes
    results_root      = "/path/to/results",
    format            = "zarr3",       # zarr2 | zarr3 | zarr3_unsharded | n5
    last_setup        = 8,
    setups_per_camera = 9,
    chunk_size        = [128, 128, 64],
    shard_size        = [512, 512, 256],
    basic_stats_level = 0,             # pyramid level the quantile stack is built from
    lsf_project       = "myproject",
    output_stem       = "$HOME/output/output",
    error_stem        = "$HOME/output/error",
)
sp.set_basic_config(estimate_darkfield=True)
```

## Workflow

```python
sp.create_quartile_histograms()   # -> bsub_command.sh      (submit it, wait)
sp.save_qstack()                  # -> qstacks/camera{N}.tiff  (inspect these)
sp.run_basic()                    # -> results_root/camera{N}/{Flat,Dark}-field.tif
sp.write_correction_script()      # -> bsub_correction.sh   (submit it)
```

## Running it locally, without LSF

The `bsub` scripts above exist to spread the work across a cluster. On a workstation, or
for a dataset small enough to sit on one machine, run the whole thing in one process:

```bash
python -m spotlight run basic        # emptiness, stats, qstack, basic, correct
python -m spotlight run intensity    # emptiness, int-stats, int-aggregate, correct
python -m spotlight run both         # both corrections, applied in ONE pass
python -m spotlight run both --dry-run          # list the stages and units first
python -m spotlight run basic --stop-after qstack   # stop and look at the qstack
python -m spotlight run basic --start-at basic      # resume after inspecting it
```

Same functions the bsub scripts invoke — only the loop over units differs. Units run
sequentially on purpose: each stage is already concurrent inside itself (asyncio over
reads, a thread pool for the numpy) and sized against a memory budget derived from the
whole machine, so running two at once would double the memory while competing for the
same saturated I/O.

`--stop-after` matters because the natural checkpoints are inspection points: look at
`qstacks/camera{N}.tiff` before fitting, and at the fields before correcting a whole
dataset. Re-running from a stage re-does it; nothing is skipped just because its output
exists, apart from `emptiness`, which is expensive and idempotent.

Off-cluster the memory budget comes from the machine's own RAM rather than an LSF
allocation — see the Tuning table.

## Correction

There is ONE correction stage, `spotlight.correct`, and it does three things:

| mode | applies | reads / writes | needs |
|---|---|---|---|
| `basic` | `max((raw - dark) / flat, 0)` | `input_basic_path` → `output_basic_path` | `run_basic()`'s fields |
| `intensity` | the per-tile gain solved from tile overlaps | `input_intensity_path` → `output_intensity_path` | the `aggregate` stage's target |
| `both` | both, in one pass | `input_intensity_path` → `output_intensity_path` | both of the above |

`--mode auto` (the default) picks `both` when the fields and the target are both present,
otherwise whichever one is.

**Prefer `both` when you want both.** It reads the data once, writes it once, and rounds
to uint16 **once** — the two-pass route quantizes the flat-field-corrected data before the
intensity rescale ever sees it. It requires `input_intensity_path == input_basic_path`
(the RAW store), and refuses to run if that is not so, since pointing it at an
already-corrected store would apply the fields twice.

For the intensity modes, first run the three-stage pipeline that produces the target —
`sp.create_intensity_correction_script()` writes `bsub_int_stats.sh`,
`bsub_int_aggregate.sh` and `bsub_int_correct.sh`, to run in that order, waiting for each
stage's jobs to finish before submitting the next.

The correction arithmetic is folded into an affine multiply-add per voxel
(`raw*(1/flat) + (-dark/flat)`) so all three corrections compose into one pass. That
differs from the unfolded order by an ULP, moving a small fraction of voxels by one gray
level — far less error than the intermediate rounding it removes. Flat-field values under
`correct.FLAT_FLOOR` are floored, because `1/0` would make the folded form NaN rather
than saturate.

Every stage is also reachable directly, which is what the generated scripts invoke:

```
python -m spotlight stats <camera> <chunk_start> <chunk_stop>
python -m spotlight qstack
python -m spotlight basic [camera ...]
python -m spotlight correct <setup> [--mode auto|basic|intensity|both]
python -m spotlight emptiness
python -m spotlight int-{stats,aggregate} [setup]
python -m spotlight int-apply <setup>          # alias for `correct --mode auto`
python -m spotlight submit {stats,correct,intensity}
```

Camera and setup arguments are 0-based. On disk cameras stay 1-based (`camera1/`), as
before.

## Tuning

Every knob is an environment variable; the defaults are sized from `LSB_DJOB_NUMPROC` and
a memory budget, so a normal run needs none of them.

| variable | default | what it does |
|---|---|---|
| `SPOTLIGHT_STATS_CONCURRENCY` | all of a camera's setups, within the budget | setup reads in flight during `stats` |
| `SPOTLIGHT_STATS_MEMORY_BYTES` | 4 GiB | budget the above is derived from |
| `SPOTLIGHT_CORRECT_CONCURRENCY` | shards fitting the budget | shards in flight during `correct` |
| `SPOTLIGHT_CORRECT_MEMORY_BYTES` | 8 GiB | budget the above is derived from |
| `SPOTLIGHT_CACHE_BYTES` | 512 MiB | tensorstore cache pool, shared by every array |
| `SPOTLIGHT_IO_CONCURRENCY` | `LSB_DJOB_NUMPROC * 64` | tensorstore in-flight file opens |
| `SPOTLIGHT_BLOCK_KIB` | 1024 | correction kernel block size — read the commentary above `_BLOCK_VOXELS` before changing it |
| `SPOTLIGHT_OTSU_VOXELS` | 32M | ceiling on the pooled sample the `emptiness` threshold is taken from |

The concurrency limits are sized from **memory**, not core count, because the reads are
latency-bound: measured on an 18-setup camera, raising `stats` concurrency from 8 (the
core count) to 18 (all setups) cut the stage from 61 s to 26 s. A limit below the setup
count also stalls on head-of-line blocking, since results are consumed in setup order.

## How the quantile stack is built

Each frame pixel's Z-column, within one setup, is reduced to 21 order statistics; those
are averaged across the camera's setups and written as 23 N5 arrays (`minima`, `maxima`,
`q000`..`q100`). `save_qstack` stacks the quantile planes into the TIFF that BaSiC fits.

The statistic is not a percentile of the column. It is the mean of sorted blocks of `p`
consecutive slices, so `q000` is a mean of per-block minima and sits well above the
column's actual minimum. Anything reasoning about a qstack plane as a plain percentile
will be wrong by that gap. `spotlight/orderstats.py` documents the rest.

Cameras with fewer than 21 Z-slices per setup have no distribution to summarise. Those
skip the stats pass entirely and `save_qstack` writes all `Z * nsetups` raw slices, so
BaSiC sees the data directly.

## Tests

```bash
pixi run python -m pytest tests -q
```

The parity tests compare against reference values produced by the Julia implementation
itself (`tests/golden/`, regenerated by `tests/gen_golden.jl`). Order statistics,
quantiles and uint16 rounding are asserted **exactly**; the BaSiC fit, being float32 and
iterative, gets a tolerance. `tests/test_stores.py` builds small synthetic stores in all
four formats and needs no Julia.

Regenerating the goldens needs Julia plus `OnlineStats`, `FFTW`, `ImageTransformations`,
`JSON3`, `Colors`, `FixedPointNumbers`, `FileIO`, `TiffImages` — it `include`s
`src/basic.jl` directly behind a few stubs, so it does not need `PyTensorStore` or its
conda environment:

```bash
julia --project=<scratch env> tests/gen_golden.jl /path/to/BigFlatFieldIlluminator.jl
```

## Benchmarks

See [bench/README.md](bench/README.md). Both implementations run inside one LSF array
job, interleaved by element index over the same work, so the per-pair ratio is not a
comparison of which hosts each landed on.
