"""The per-camera quantile statistics pass.

One LSF array element owns a slice of the frame's X/Y chunks. For each of them it reads
that chunk from every setup the camera covers, reduces each setup's Z-columns to
`OrderStats`, merges across setups and then across Z blocks, and writes 23 n5 arrays
(`minima`, `maxima`, and `q000`..`q100` in steps of 5) under
`{results_root}/camera{N}/{stat}/s{level}`.

Where the Julia version threads over setups and loops per pixel, this reads them
concurrently through tensorstore's async API and reduces each chunk in one vectorised
sort. The merge is still a strict left fold in SETUP ORDER, not completion order --
float addition is not associative, and matching Julia's order is what makes the two
implementations comparable rather than merely close.
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config as _config
from . import stores
from .orderstats import LEVELS, N_QUARTILES, OrderStats, block_size, quantile_r7, to_uint16

__all__ = ["calculate_camera_stats", "BACKGROUND_PIXEL_STRIDE"]

# Only every Nth frame pixel in each direction contributes to the background profile. The
# result is a single 21-vector per camera, averaged over (empty columns x setups x Z
# blocks), so even a 1/64 sample leaves it averaged over millions of Z-samples -- while a
# full sweep would evaluate 21 quantiles at every (setup, pixel, Z block) triple in the
# hot loop.
BACKGROUND_PIXEL_STRIDE = 8


def _concurrency(cfg):
    """How many setup reads may be in flight at once.

    This is what actually bounds memory: each in-flight setup holds its chunk plus a
    (X, Y, p) float64 `value` buffer, and a camera can cover a hundred setups. Julia holds
    every setup's accumulator simultaneously; bounding it is one of the two reasons this
    port uses less memory than the original.
    """
    return int(os.getenv("SPOTLIGHT_STATS_CONCURRENCY",
                         os.getenv("LSB_DJOB_NUMPROC", "16")))


def _z_depth(cfg, read_depth):
    """The block depth `OrderStats` is sized from, for a chunk `read_depth` slices deep."""
    if cfg["input_format"] == "zarr3":
        configured = cfg["shard_size"][2] * cfg["z_batch"]
    else:
        configured = cfg["chunk_size"][2]
    return min(configured, read_depth)


def _z_blocks(cfg, z_total):
    """The Z blocks one chunk is read in.

    Only sharded zarr3 splits Z at all; everything else reads the whole column in one
    pass. Every block must be EXACTLY the same depth, because `OrderStats` accumulators
    sized off different depths cannot merge. If Z is not a multiple of the block size the
    last block is SLID BACK rather than truncated -- a few Z-slices get double-counted in
    the overlap, which is statistically negligible for per-pixel quantiles pooled over
    many tiles, and keeps every block mergeable.
    """
    if cfg["input_format"] != "zarr3":
        return [(0, z_total)]
    sz = cfg["shard_size"][2] * cfg["z_batch"]
    if z_total <= sz:
        return [(0, z_total)]
    starts = list(range(0, z_total, sz))
    starts[-1] = min(starts[-1], z_total - sz)
    starts = sorted(set(starts))
    return [(z, z + sz) for z in starts]


def empty_threshold(cfg, camera):
    """The dataset-wide intensity threshold separating background from specimen.

    Measured by the emptiness stage and merged into every tile's own stats JSON. None
    when that stage has not run, in which case the stats pass simply skips the
    background-profile measurement.
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
    measured separately, so it cannot disagree with them about block geometry: `q000` is
    a mean of per-block minima, not a column minimum. Measured on one dataset, a profile
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
        background it bisects tissue, which would quietly make "background" mean "the
        dimmer half of the specimen".
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
    be relied on (a job whose chunks all land on specimen finds no empty column at all).
    `create_quartile_histograms` clears the directory when it writes a new submission
    script, so a rerun cannot blend its partials with the previous run's.
    """
    d = background_quantile_dir(cfg, camera)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"job{job}.json"
    path.write_text(json.dumps({"sum": list(acc.sum), "count": acc.count,
                                "stride": BACKGROUND_PIXEL_STRIDE}))
    print(f"wrote background-quantile partial {path} ({acc.count} columns)")


# ─── the pass ─────────────────────────────────────────────────────────────────


async def _fit_setup(src, xs, ys, zr, p, sem, pool, loop):
    """Read one setup's chunk and reduce it. The SEMAPHORE IS NOT RELEASED HERE.

    The caller releases it after consuming the result, so at most `limit` finished-but-
    unmerged accumulators are alive at once. Releasing on completion instead would let
    every task pile up its buffer while the fold is still waiting on the first one, which
    bounds nothing.
    """
    await sem.acquire()
    block = await src[zr[0]:zr[1], ys, xs].read(order="C")   # (Z, Y, X)
    a = np.ascontiguousarray(block.T)                        # -> (X, Y, Z)
    return await loop.run_in_executor(pool, OrderStats.fit, a, p)


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

    z_blocks = _z_blocks(cfg, cam[2])
    p = block_size(_z_depth(cfg, z_blocks[0][1] - z_blocks[0][0]))

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

    limit = _concurrency(cfg)
    sem = asyncio.Semaphore(limit)
    loop = asyncio.get_running_loop()
    print(f"calculating statistics: camera {camera + 1}, {len(setups)} setups, "
          f"chunks {start}..{stop} of {len(chunks)}, cam_size {cam}, "
          f"{len(z_blocks)} z block(s) of depth {p}, threshold {threshold}, level {lvl}")

    with ThreadPoolExecutor(max_workers=cfg["n_cores_stats"]) as pool:
        for idx in range(start, stop + 1):
            xs, ys = chunks[idx - 1]
            mstat = None
            for zr in z_blocks:
                tasks = [asyncio.ensure_future(_fit_setup(src, xs, ys, zr, p, sem, pool, loop))
                         for src in srcs]
                partial = None
                try:
                    for t in tasks:
                        st = await t
                        sem.release()
                        # Before the merge, which consumes the accumulator in place: the
                        # background profile needs each setup's OWN quantiles, not their
                        # average.
                        if acc is not None:
                            acc.accumulate(st, threshold)
                        partial = st if partial is None else partial.merge(st)
                except BaseException:
                    for t in tasks:
                        t.cancel()
                    raise
                mstat = partial if mstat is None else mstat.merge(partial)
            await _save_stats(stats_arrays, xs, ys, mstat)
            print(f"chunk {idx}/{stop} written")

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
