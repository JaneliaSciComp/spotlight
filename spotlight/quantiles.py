"""Stage: the per-camera quantile statistics pass.

Named `quantiles`, not `stats`, because `tilestats.py` is also a "stats" stage: this one
reduces a camera's Z-columns to order statistics for BaSiC, that one measures one tile's
foreground for the gain solve. Pairs with `qstack.py`, which consumes what this writes.

One LSF array element owns a slice of the frame's X/Y chunks. For each it reads that chunk
from every setup the camera covers, reduces each setup's Z-columns to `OrderStats`, merges
across setups and then across Z blocks, and writes 23 n5 arrays (`minima`, `maxima`, and
`q000`..`q100` in steps of 5) under `{results_root}/camera{N}/{stat}/s{level}`.

Where the Julia version threads over setups and loops per pixel, this reads them
concurrently through tensorstore's async API and reduces each chunk in one vectorised
sort. The merge is still a strict left fold in SETUP ORDER, not completion order -- float
addition is not associative, and matching Julia's order is what makes the two
implementations comparable rather than merely close.
"""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config as _config
from . import stores
from .progress import Progress
from .orderstats import LEVELS, N_QUARTILES, OrderStats, block_size, quantile_r7, to_uint16

__all__ = ["calculate_camera_stats", "BACKGROUND_PIXEL_STRIDE"]

# Only every Nth frame pixel in each direction contributes to the background profile. The
# result is a single 21-vector per camera, averaged over (empty columns x setups x Z
# blocks), so even a 1/64 sample leaves it averaged over millions of Z-samples -- while a
# full sweep would evaluate 21 quantiles at every (setup, pixel, Z block) triple in the
# hot loop.
BACKGROUND_PIXEL_STRIDE = 8


def _concurrency(cfg, n_setups, bytes_per_setup):
    """How many setup reads may be in flight at once.

    Sized from a MEMORY budget, not the core count. Reads here are latency-bound, not
    CPU-bound -- measured on an 18-setup camera over a sharded store, 9216 reads of ~1 MiB
    moved at 49 MiB/s aggregate -- so matching concurrency to cores starves the I/O.
    `stores._context` already reasons this way for tensorstore's own
    `file_io_concurrency`; this is the same argument one level up.

    Worse, a limit below `n_setups` causes head-of-line blocking. Results must be consumed
    in setup order (the merge is a left fold and float addition is not associative), so a
    slot is held from the moment a task starts until its turn comes round. With 8 slots
    and 18 setups, tasks 1..7 finish and sit on their slots while task 0 is still reading,
    and nothing new can start. Letting every setup of one Z block be in flight removes
    that stall and costs little: each holds its chunk plus an (X, Y, p) float64 buffer, ~7
    MiB for a 64x64 tile at block size 126.
    """
    env = os.getenv("SPOTLIGHT_STATS_CONCURRENCY")
    if env:
        return max(1, int(env))
    budget = int(os.getenv("SPOTLIGHT_STATS_MEMORY_BYTES", stores.memory_budget()))
    return max(1, min(n_setups, budget // max(bytes_per_setup, 1)))


def _z_read_blocks(cfg, z_total):
    """The contiguous Z ranges one XY chunk is read in, covering [0, z_total) exactly.

    Sharded zarr3 is read in shard-deep batches (`shard_size[2] * z_batch`) so each read
    lands on whole shards; every other format reads the column in one pass. The batches
    are CONCATENATED into the full Z-column before the quantile is taken, so -- unlike a
    block merge -- they need not be equal depth: the last is simply shorter. Concatenating
    the samples, rather than merging per-batch order statistics, is what makes the result
    a TRUE per-pixel quantile of the whole column instead of an average of per-block ones.
    """
    if cfg["input_format"] != "zarr3":
        return [(0, z_total)]
    sz = cfg["shard_size"][2] * cfg["z_batch"]
    return [(z, min(z + sz, z_total)) for z in range(0, z_total, sz)]


def empty_threshold(cfg, camera):
    """The dataset-wide intensity threshold separating background from specimen.

    Measured by the emptiness stage and merged into every tile's own stats JSON. None when
    that stage has not run, in which case the stats pass skips the background-profile
    measurement.
    """
    for setup in _config.camera_setups(cfg)[camera]:
        path = Path(cfg["results_root"]) / "intensity_stats" / f"setup{setup}.json"
        if not path.is_file():
            continue
        try:
            thr = json.loads(path.read_text()).get("empty_threshold")
        except (OSError, ValueError):
            continue
        if isinstance(thr, (int, float)):
            return float(thr)
    return None


class BackgroundQuantiles:
    """Running per-quantile background profile, plus the number of columns behind it.

    Taken from the very `OrderStats` whose average becomes the qstack plane, rather than
    measured separately, so it cannot disagree with them about block geometry: `q000` is a
    mean of per-block minima, not a column minimum. Measured on one dataset, a profile
    built from full-column percentiles spanned 173..239 counts where the real one spans
    183..221 -- subtracting the former would over-correct the spread by ~1.7x.
    """

    __slots__ = ("sum", "count")

    def __init__(self):
        self.sum = np.zeros(N_QUARTILES, dtype=np.float64)
        self.count = 0

    def accumulate(self, st, threshold):
        """Add every entirely-empty column in one setup's stats.

        A column counts as empty only when its MAXIMUM is under `threshold`: nothing in it
        ever saw specimen. `threshold` must be the dataset-wide one -- a per-tile
        threshold is not comparable between tiles, and on a tile with no separable
        background it bisects tissue, quietly making "background" mean "the dimmer half of
        the specimen".
        """
        s = BACKGROUND_PIXEL_STRIDE
        sel = st.vmax[::s, ::s] < threshold
        n = int(sel.sum())
        if n == 0:
            return
        v = np.sort(st.value[::s, ::s], axis=-1)[sel]     # (n_empty, p)
        for i, q in enumerate(LEVELS):
            self.sum[i] += float(quantile_r7(v, q / 100.0).sum())
        self.count += n


def background_quantile_dir(cfg, camera):
    return Path(cfg["results_root"]) / f"camera{camera + 1}" / "background_quantiles"


def write_background_quantile_partial(cfg, camera, job, acc):
    """Write one stats job's share of the background profile.

    Per job rather than per camera because the stats pass is an LSF array: each element
    owns a slice of the frame's chunks, and no single element sees enough of the frame to
    be relied on -- a job whose chunks all land on specimen finds no empty column at all.
    `create_quartile_histograms` clears the directory when it writes a submission script
    whose partials would not match the ones already there, so a rerun cannot blend two
    runs' partials, and finished cameras are left alone when they would.
    """
    d = background_quantile_dir(cfg, camera)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"job{job}.json"
    path.write_text(json.dumps({"sum": list(acc.sum), "count": acc.count,
                                "stride": BACKGROUND_PIXEL_STRIDE}))
    print(f"wrote background-quantile partial {path} ({acc.count} columns)")


# ─── the pass ─────────────────────────────────────────────────────────────────


async def _fit_setup(src, xs, ys, z_reads, p, sem, pool, loop, timing):
    """Read one setup's whole Z-column for this XY chunk and reduce it. THE SEMAPHORE IS
    NOT RELEASED HERE.

    The caller releases it after consuming the result, so at most `limit` finished-but-
    unmerged accumulators are alive at once. Releasing on completion instead would let
    every task pile up its buffer while the fold is still waiting on the first one, which
    bounds nothing.

    Sharded zarr3 arrives as several shard-deep reads (`z_reads`) issued together and
    CONCATENATED along Z; every other format is a single read. Either way the reduction
    below sees the full column and sorts it once.
    """
    await sem.acquire()
    t0 = time.perf_counter()
    if len(z_reads) == 1:
        z0, z1 = z_reads[0]
        block = await src[z0:z1, ys, xs].read(order="C")     # (Z, Y, X)
    else:
        parts = await asyncio.gather(
            *(src[z0:z1, ys, xs].read(order="C") for z0, z1 in z_reads))
        block = np.concatenate(parts, axis=0)                # full Z-column
    t1 = time.perf_counter()
    a = np.ascontiguousarray(block.T)                        # -> (X, Y, Z)
    st = await loop.run_in_executor(pool, OrderStats.fit, a, p)
    t2 = time.perf_counter()
    # Aggregate time spent in each phase across all tasks. These overlap, so they sum to
    # more than the wall clock -- the ratio is what says whether this job is I/O or CPU
    # bound, which is the thing worth knowing before optimising either.
    timing["t_read"] += t1 - t0
    timing["t_compute"] += t2 - t1
    timing["bytes_in"] += block.nbytes
    return st


async def _run(cfg, camera, start, stop):
    lvl = cfg["basic_stats_level"]
    # Every size here is at `basic_stats_level`, not level 0: the frame the stats arrays
    # cover, the X/Y job tiling that addresses them, and the Z extent the quantiles are
    # taken over all shrink together with the level. The submission script sizes its job
    # array off the same level, so the chunk indices agree.
    s0 = stores.source_size_xyz(cfg, scale=lvl)
    chunks = stores.xy_chunks(s0[:2], cfg["chunk_size"][:2])
    # The bsub array's last job overshoots on purpose (num_jobs = ceil(total /
    # chunks_per_job)), so its computed `stop` can exceed the true chunk count.
    stop = len(chunks) if stop is None else min(stop, len(chunks))

    setups = _config.camera_setups(cfg)[camera]
    cam = stores.camera_source_size_xyz(cfg, setups, scale=lvl)
    if cam[0] < s0[0] or cam[1] < s0[1]:
        raise RuntimeError(
            f"camera {camera + 1} has setups smaller in X/Y ({cam[:2]}) than the reference "
            f"setup ({s0[:2]}); the stats tiling assumes a common X/Y canvas")
    if cam[2] < N_QUARTILES:
        raise RuntimeError(
            f"camera {camera + 1} has only {cam[2]} Z-slice(s) per setup, fewer than "
            f"N_QUARTILES={N_QUARTILES}, so per-pixel quantiles are undefined. Skip this "
            "stats job: save_qstack() feeds BaSiC the raw slices directly.")

    z_reads = _z_read_blocks(cfg, cam[2])
    # One sorted block over the whole Z-column -> a TRUE per-pixel quantile. block_size
    # rounds the depth down to a multiple of N_QUARTILES, discarding a few trailing
    # slices exactly as the Julia pipeline did.
    p = block_size(cam[2])

    # Measured here because this pass already owns the `OrderStats` the qstack is made of,
    # so the profile cannot disagree with it about block geometry. Needs the emptiness
    # stage's dataset-wide threshold, which is why create_quartile_histograms runs that
    # stage before submitting this one.
    threshold = empty_threshold(cfg, camera)
    acc = None if threshold is None else BackgroundQuantiles()
    if threshold is None:
        print(f"warning: no empty_threshold found for camera {camera + 1}, so this pass "
              "will not measure the per-quantile background profile; basic_unmix_empty "
              "will have nothing to subtract. Run the emptiness stage and resubmit if you "
              "intend to un-mix.")

    ctx = stores.context()
    srcs = [stores.open_source(cfg, s, scale=lvl, ctx=ctx) for s in setups]
    stats_arrays = {name: stores.open_stats_array(cfg, camera, name, s0[:2], scale=lvl, ctx=ctx)
                    for name in ("minima", "maxima", *(f"q{q:03d}" for q in LEVELS))}

    # One in-flight setup holds its whole Z-column for this XY chunk (uint16, plus a
    # transient copy from the transpose/concat) and the float64 `value` buffer the
    # reduction builds from it. Reading the full column is the memory cost of a true
    # per-pixel quantile; for sharded zarr3 it means holding one column's worth of every
    # shard the column crosses -- accepted deliberately.
    tile = cfg["chunk_size"][0] * cfg["chunk_size"][1]
    bytes_per_setup = tile * (3 * cam[2] * 2 + p * 8)
    limit = _concurrency(cfg, len(setups), bytes_per_setup)
    sem = asyncio.Semaphore(limit)
    loop = asyncio.get_running_loop()
    timing = {"t_read": 0.0, "t_compute": 0.0, "t_write": 0.0, "bytes_in": 0,
              "n_reads": 0, "concurrency": limit, "block_size": p,
              "n_z_reads": len(z_reads), "n_setups": len(setups)}
    t_start = time.perf_counter()
    print(f"calculating statistics: camera {camera + 1}, {len(setups)} setups, "
          f"chunks {start}..{stop} of {len(chunks)}, cam_size {cam}, "
          f"1 sorted block of depth {p} read in {len(z_reads)} pass(es), "
          f"threshold {threshold}, level {lvl}")

    bar = Progress(stop - start + 1, f"stats camera {camera + 1}")
    with ThreadPoolExecutor(max_workers=cfg["n_cores_stats"]) as pool:
        for idx in range(start, stop + 1):
            xs, ys = chunks[idx - 1]
            tasks = [asyncio.ensure_future(
                         _fit_setup(src, xs, ys, z_reads, p, sem, pool, loop, timing))
                     for src in srcs]
            mstat = None
            try:
                for t in tasks:
                    st = await t
                    sem.release()
                    # Before the merge, which consumes the accumulator in place: the
                    # background profile needs each setup's OWN quantiles, not their
                    # average.
                    if acc is not None:
                        acc.accumulate(st, threshold)
                    mstat = st if mstat is None else mstat.merge(st)
            except BaseException:
                for t in tasks:
                    t.cancel()
                raise
            tw = time.perf_counter()
            await _save_stats(stats_arrays, xs, ys, mstat)
            timing["t_write"] += time.perf_counter() - tw
            timing["n_reads"] += len(srcs) * len(z_reads)
            bar.advance()
    bar.close()

    timing["t_total"] = time.perf_counter() - t_start
    # One machine-readable line the benchmark harness folds into its record.
    print("SPOTLIGHT_TIMING " + json.dumps(timing), flush=True)

    if acc is not None:
        # Keyed by this job's first chunk, which is unique across the LSF array.
        write_background_quantile_partial(cfg, camera, start, acc)


async def _save_stats(arrays, xs, ys, mstat):
    """Write one chunk's 23 statistics, all writes in flight together."""
    planes = {"minima": mstat.vmin, "maxima": mstat.vmax}
    for q, plane in zip(LEVELS, mstat.quantiles()):
        planes[f"q{q:03d}"] = plane
    await asyncio.gather(*(arrays[name][xs, ys].write(to_uint16(plane))
                           for name, plane in planes.items()))


def calculate_camera_stats(cfg, camera, start=1, stop=None):
    """Run the stats pass for one camera over chunks `start..stop` (1-based, inclusive)."""
    asyncio.run(_run(cfg, camera, start, stop))
