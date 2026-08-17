"""The one correction stage: flat/dark, per-tile intensity, or both.

There used to be two implementations of this — a BaSiC-only one here and the intensity
pipeline's `apply` — which differed in their blocking strategy, their memory profile and
their output layout while computing overlapping arithmetic. This is the merged one.

Three modes, all through the same shard loop and the same kernel:

* `basic`     — `max((raw - dark) / flat, 0)`, reading `input_basic_path` and writing
                `output_basic_path`. Needs `run_basic()`'s fields; needs no stats.
* `intensity` — the per-tile gain solved from tile overlaps, reading
                `input_intensity_path` and writing `output_intensity_path`. Needs the
                `aggregate` stage's target file.
* `both`      — both at once, so the data is read once, written once, and rounded to
                uint16 ONCE rather than twice through an intermediate store. This is the
                reason to prefer it: the two-pass route quantizes the flat-field-corrected
                data before the intensity rescale ever sees it. Requires
                `input_intensity_path == input_basic_path` (the RAW store).

`auto` picks `both` when the BaSiC fields and the intensity target are both present,
otherwise whichever one is.

With `apply_basic` on, `int-stats` and `int-aggregate` correct the voxels THEY read too,
not just this stage. They have to: the Otsu split, the per-tile foreground mean/std and
the overlap medians the gain is solved from must all describe the same values this stage
will write. Flat-field vignetting varies across a tile's field of view, so gains solved on
raw voxels would partly chase vignetting instead of sensor gain. For the downsampled
levels those stages read, the fields are mean-downsampled by the same per-axis factor --
the fields vary slowly in-plane, so downsample-then-divide and divide-then-downsample
agree well within the noise. The stats and target files record which mode produced them,
and both refuse to mix modes rather than silently combining raw-derived statistics with
basic-corrected voxels.

The heavy lifting is `ShardCorrection` / `_correct_shard`, which
compose all three corrections into one pass over each shard and are tuned against real
shard geometry — see the commentary above `_BLOCK_VOXELS` there for why the CPU blocks
are ~1 MiB contiguous row-spans rather than chunks or whole planes.
"""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorstore as ts

from .formats import _SPEC, _in_order, _input_location, canonical_shape, canonical_view
from .kernel import ShardCorrection, _BLOCK_VOXELS, _blocks, _correct_shard
from .config import basic_field_paths, camera_of, target_path
from .stores import open_output_array, source_pyramid_factors, write_group_metadata
from .fields import _check_basic_mode, basic_model
from .progress import Progress
from .tiffout import (COMPRESSION, DEFAULT_SLAB_PLANES as TIFF_SLAB_PLANES,
                      tiff_tile_path, voxel_size_um, write_tile_ome_tiff)
from .tilestats import _classify, limits
from . import config as _config
from . import stores

__all__ = ["apply_correction_chunked", "apply_correction", "resolve_mode", "MODES"]

MODES = ("auto", "basic", "intensity", "both")

# Flat-field values at or below this are replaced by it. The flat field is normalised to
# mean 1, so anything three orders of magnitude under that is not a measurement -- it is
# a pixel a degenerate fit left at (or below) zero.
#
# A floor is required, not merely tidy: the kernel folds `(raw - dark)/flat` into
# `raw*(1/flat) + (-dark/flat)`, and at flat == 0 that is `inf + (-inf)` = NaN, which
# casts to an ARBITRARY uint16 instead of saturating. The floor must also not be tiny,
# or `1/flat` grows until the two terms cancel catastrophically in float32 -- at 1e-6
# they are ~6.7e8 apart and the difference keeps no significant digits.
FLAT_FLOOR = np.float32(1e-3)   # default for `flat_floor`; see the warning below


def apply_correction(image, flat, dark, nonneg_offset=0.0, multiplier=1.0):
    """The pure image math, as float32. `image` broadcasts against a (Y, X) field.

    Kept for callers that want the arithmetic on an array in hand. The stage itself goes
    through `_correct_shard`, which folds this into an affine multiply-add.
    """
    out = (np.asarray(image, dtype=np.float32) - dark) / flat + np.float32(nonneg_offset)
    if multiplier != 1.0:
        out *= np.float32(multiplier)
    return out


# ─── mode ─────────────────────────────────────────────────────────────────────


def _has_basic_fields(cfg, setup):
    try:
        cam = camera_of(cfg, setup)
    except RuntimeError:
        return False
    return all(p.exists() for p in basic_field_paths(cfg, cam))


def resolve_mode(cfg, setup, requested="auto"):
    """Which corrections this run applies, and a clear error when it cannot."""
    if requested not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {requested!r}")
    has_fields = _has_basic_fields(cfg, setup)
    has_target = target_path(cfg).exists()

    if requested == "auto":
        if has_fields and has_target:
            return "both"
        if has_fields:
            return "basic"
        if has_target:
            return "intensity"
        raise RuntimeError(
            f"nothing to correct for setup {setup}: no BaSiC fields under "
            f"{cfg['results_root']} (run `basic`) and no intensity target at "
            f"{target_path(cfg)} (run `int-aggregate`)")

    if requested in ("basic", "both") and not has_fields:
        raise RuntimeError(
            f"mode={requested} needs run_basic()'s fields for setup {setup} under "
            f"{cfg['results_root']}/camera*/; run `python -m spotlight basic` first")
    if requested in ("intensity", "both") and not has_target:
        raise RuntimeError(
            f"mode={requested} needs the intensity target at "
            f"{target_path(cfg)}; run the aggregate stage first")
    return requested


def _view(cfg, mode):
    """The config with the I/O paths this mode reads and writes.

    `basic` works on the BaSiC pair; `intensity` and `both` on the intensity pair. In
    `both` those must be the same input store — the raw data — or the flat/dark
    correction gets applied twice.
    """
    if mode == "basic":
        return {**_config.basic_view(cfg), "apply_basic": True}
    view = {**cfg, "apply_basic": mode == "both"}
    if mode == "both":
        raw, basic_in = view.get("input_intensity_path"), cfg.get("input_basic_path")
        if basic_in and raw != basic_in:
            raise RuntimeError(
                "mode=both applies the flat/dark correction while reading, so "
                f"input_intensity_path ({raw}) must be the RAW store, the same as "
                f"input_basic_path ({basic_in}). Pointing it at an already-corrected "
                "store would apply the fields twice.")
    return view


def _intensity_params(cfg, setup):
    """`(mode, thr, mean, scale, target_mean)` for the per-tile rescale.

    Lifted from the intensity pipeline's apply stage unchanged, including the reason the
    formula reduces to a pure per-tile gain: `corrected_mean`/`corrected_std` encode this
    tile's gain `g` (the aggregate stage writes `corrected_mean = g*M`,
    `corrected_std = g*S`), so the rescale is `out = raw/g`. Texture survives — it is one
    gain per tile, not a re-center to a common mean — and only the sensor gain the
    overlap solve isolated from content is removed.
    """
    tp = target_path(cfg)
    combined = json.loads(tp.read_text())
    _check_basic_mode(cfg, combined.get("apply_basic"), str(tp))
    target_mean, target_std = combined["target_mean"], combined["target_std"]

    s = combined["setups"].get(str(setup))
    if s is None:
        print(f"correct: setup {setup} has no stats -> no intensity rescale")
        return "none", 0, 0.0, 1.0, target_mean

    lim = limits(cfg)
    kind = _classify(s, lim)
    thr, _n_fg = s["threshold"], s["n_foreground"]
    if kind == "empty":
        print(f"correct: setup {setup} too empty -> no intensity rescale")
        return "none", 0, 0.0, 1.0, target_mean

    mean_i, std_i = s["corrected_mean"], s["corrected_std"]
    if kind == "uniform":
        thr = float("-inf")            # no clean background to protect: correct everything
    scale = min(target_std / std_i, lim["max_gain_scale"])
    print(f"correct: setup {setup} [{kind}] mean={mean_i:.2f} std={std_i:.2f} -> "
          f"target_mean={target_mean:.2f} target_std={target_std:.2f} "
          f"scale={scale:.4f} thr={thr:.1f}")
    return ("uniform" if thr == float("-inf") else "bimodal"), thr, mean_i, scale, target_mean


# ─── the stage ────────────────────────────────────────────────────────────────


def _concurrency(cfg, n_shards, shard_bytes):
    """How many shards may be in flight, sized from a memory budget.

    A plain count is the wrong knob when one shard can be 1.36 GiB — a shard spanning the
    whole plane costs three orders of magnitude more than a 64^3 chunk at the same limit.
    Each in flight holds its shard plus the transaction buffering the write.
    """
    env = os.getenv("SPOTLIGHT_CORRECT_CONCURRENCY")
    if env:
        return max(1, int(env))
    budget = int(os.getenv("SPOTLIGHT_CORRECT_MEMORY_BYTES", stores.memory_budget()))
    return max(1, min(n_shards, budget // max(2 * shard_bytes, 1)))


async def _write_tiff(view, setup, mode, src_c, zyx, dtype_name, shard_corr):
    """One BigTIFF OME-TIFF per tile, streamed in z order.

    Separate from the shard loop above because a TIFF is a single file with sequential
    IFDs: shards cannot be written out of order or concurrently. The reads still overlap
    the writes (`tiffout` prefetches one slab), and the correction is the same kernel --
    only the write side differs.

    No pyramid: level 0 only, so this is one pass over the data with no re-read.
    """
    Z, Y, X = zyx
    path = tiff_tile_path(view, setup)
    voxel = voxel_size_um(view, setup)
    pool = ThreadPoolExecutor(max_workers=int(os.getenv("LSB_DJOB_NUMPROC", "8")))
    timing = {"t_read": 0.0, "t_compute": 0.0, "t_write": 0.0, "bytes_in": 0,
              "n_shards": 0, "concurrency": 1, "mode": mode}
    t_start = time.perf_counter()
    print(f"correct: setup {setup} mode={mode} {view['input_format']} {zyx} -> "
          f"tiff @ {path}")
    print(f"correct: streaming {Z} planes of {Y}x{X} in "
          f"{int(view.get('tiff_slab_planes') or TIFF_SLAB_PLANES)}-plane slabs, "
          f"voxel {voxel if voxel else 'UNCALIBRATED (no dataset.xml voxelSize)'} um, "
          f"compression {view.get('tiff_compression') or COMPRESSION}")
    if voxel is None:
        print("correct: WARNING no voxel size found -- the OME-TIFF will carry no "
              "calibration, so anything measured from it in BigStitcher will be in "
              "pixels. Point `dataset_xml` at the SpimData2 xml to fix.")

    def read_slab(z0, z1):
        t0 = time.perf_counter()
        block = src_c[z0:z1, 0:Y, 0:X].read(order="C").result()
        t1 = time.perf_counter()
        if shard_corr is not None:
            block = _correct_shard(block, shard_corr, (0, Y), (0, X))
        timing["t_read"] += t1 - t0
        timing["t_compute"] += time.perf_counter() - t1
        timing["bytes_in"] += block.size * np.dtype(dtype_name).itemsize
        return block

    bar = Progress(Z, f"correct setup {setup}")
    try:
        await asyncio.get_running_loop().run_in_executor(
            pool, lambda: write_tile_ome_tiff(
                path, read_slab, (Z, Y, X), dtype_name, voxel=voxel,
                planes=int(view.get("tiff_slab_planes") or TIFF_SLAB_PLANES),
                compression=(view.get("tiff_compression") or COMPRESSION),
                progress=bar))
    finally:
        bar.close()
        pool.shutdown(wait=True)
    timing["t_total"] = time.perf_counter() - t_start
    # tifffile owns the write, so there is no inner timer to read: the write is whatever
    # the wall clock did not spend reading or correcting (compression included).
    timing["t_write"] = max(timing["t_total"] - timing["t_read"] - timing["t_compute"], 0.0)
    print("SPOTLIGHT_TIMING " + json.dumps(timing), flush=True)
    print(f"correct: done setup {setup}, mode={mode}, tiff level 0 only -> {path} "
          f"({path.stat().st_size / 2**30:.2f} GiB on disk)")


async def _run(cfg, setup, requested):
    mode = resolve_mode(cfg, setup, requested)
    view = _view(cfg, mode)
    ctx = stores.context()
    in_path, in_order = _input_location(view, setup, 0)

    src = ts.open({"driver": _SPEC[view["input_format"]]["driver"],
                   "kvstore": {"driver": "file", "path": in_path}},
                  context=ctx, create=False, open=True).result()
    # Canonical (Z, Y, X) at both ends, so the shard loop never transposes in numpy and
    # the kernel always sees C-contiguous data.
    src_c = canonical_view(src, in_order)
    zyx = tuple(src_c.domain.shape)
    dtype_name = str(src.dtype.name)

    basic = basic_model(view, setup, zyx[1:]) if mode in ("basic", "both") else None
    if basic is not None:
        cam = camera_of(view, setup)
        floor = np.float32(view.get("flat_floor") or FLAT_FLOOR)
        n_zero = int((basic.flat <= 0).sum())
        if n_zero:
            # A zero here is not merely a division by zero, because the kernel folds
            # `(raw - dark)/flat` into `raw*(1/flat) + (-dark/flat)`: at flat == 0 that is
            # `inf + (-inf)` = NaN, which casts to an ARBITRARY uint16 rather than
            # saturating. Floor the field instead, so both forms agree and the pixel
            # saturates the way the unfolded arithmetic would. The floor is far below any
            # physical value -- the flat field is normalised to mean 1 -- so it only ever
            # touches pixels a degenerate fit left at zero.
            #
            # Not silent: a saturated patch reads as real signal downstream.
            print(f"warning: the flat field for setup {setup} has {n_zero} non-positive "
                  f"pixel(s) of {basic.flat.size}; those are floored to {float(floor)} "
                  f"and will read as saturated wherever raw exceeds dark. Check "
                  f"{basic_field_paths(view, cam)[0]}")
            basic.flat = np.maximum(basic.flat, floor)
        print(f"correct: setup {setup} flat/dark from camera {cam + 1} "
              f"(flat mean={float(basic.flat.mean()):.4f}, dark mean={float(basic.dark.mean()):.2f})")

    if mode in ("intensity", "both"):
        imode, thr, mean_i, scale_i, target_mean = _intensity_params(view, setup)
    else:
        imode, thr, mean_i, scale_i, target_mean = "none", 0, 0.0, 1.0, 0.0

    # None only when there is nothing to do at all, in which case shards copy straight
    # through -- which `resolve_mode` has already ruled out.
    shard_corr = (None if basic is None and imode == "none"
                  else ShardCorrection(basic, imode, thr, mean_i, scale_i, target_mean))

    if view["output_format"] == "tiff":
        await _write_tiff(view, setup, mode, src_c, zyx, dtype_name, shard_corr)
        return

    # Below here is the zarr/n5 path only. `_SPEC` has no "tiff" row on purpose -- it maps
    # formats to tensorstore drivers, and there is no TIFF driver -- so this lookup has to
    # sit after the branch above, not beside the input one.
    out_order = _SPEC[view["output_format"]]["order"]
    out, shard, out_path = open_output_array(
        view, setup, 0, _in_order(zyx, out_order), dtype_name, ctx)
    out_c = canonical_view(out, out_order)

    sz, sy, sx = canonical_shape(shard, out_order)
    Z, Y, X = zyx
    origins = [(oz, oy, ox)
               for oz in range(0, Z, sz)
               for oy in range(0, Y, sy)
               for ox in range(0, X, sx)]
    itemsize = np.dtype(dtype_name).itemsize
    shard_bytes = sz * sy * sx * itemsize
    limit = _concurrency(cfg, len(origins), shard_bytes)
    sem = asyncio.Semaphore(limit)

    # The kernel must not run as plain synchronous code inside a coroutine: that blocks
    # the event loop and serializes every shard's compute onto one core no matter how
    # many were asked for. In a thread pool, numpy's GIL-releasing array ops overlap
    # across cores while asyncio keeps the reads flowing.
    n_cores = int(os.getenv("LSB_DJOB_NUMPROC", "8"))
    pool = ThreadPoolExecutor(max_workers=n_cores)
    timing = {"t_read": 0.0, "t_compute": 0.0, "t_write": 0.0, "bytes_in": 0,
              "n_shards": len(origins), "concurrency": limit, "mode": mode}
    t_start = time.perf_counter()
    print(f"correct: setup {setup} mode={mode} {view['input_format']} {zyx} -> "
          f"{view['output_format']} @ {out_path}")
    print(f"correct: {len(origins)} shard(s) of {(sz, sy, sx)} "
          f"({shard_bytes / 2**30:.2f} GiB), {limit} in flight, {n_cores} kernel threads, "
          f"{len(_blocks((sz, sy, sx)))} blocks/shard of "
          f"{_BLOCK_VOXELS * 4 // 1024} KiB (override: SPOTLIGHT_BLOCK_KIB)")

    async def write_shard(oz, oy, ox):
        z = (oz, min(oz + sz, Z)); y = (oy, min(oy + sy, Y)); x = (ox, min(ox + sx, X))
        async with sem:
            t0 = time.perf_counter()
            # Freshly allocated per read, so it is writable and the kernel can correct
            # it in place -- no second shard-sized buffer anywhere in this loop.
            canon = await src_c[z[0]:z[1], y[0]:y[1], x[0]:x[1]].read(order="C")
            t1 = time.perf_counter()
            if shard_corr is not None:
                canon = await asyncio.get_running_loop().run_in_executor(
                    pool, _correct_shard, canon, shard_corr, y, x)
            t2 = time.perf_counter()
            # One transaction per shard, so the shard file is written exactly once and
            # the inner-chunk writes batch into it -- no rewrite amplification.
            txn = ts.Transaction(atomic=False)
            await out_c[z[0]:z[1], y[0]:y[1], x[0]:x[1]].with_transaction(txn).write(canon)
            await txn.commit_async()
            t3 = time.perf_counter()
        timing["t_read"] += t1 - t0
        timing["t_compute"] += t2 - t1
        timing["t_write"] += t3 - t2
        timing["bytes_in"] += canon.size * itemsize
        bar.advance()

    bar = Progress(len(origins), f"correct setup {setup}")
    try:
        await asyncio.gather(*(write_shard(*o) for o in origins))
    finally:
        bar.close()
        pool.shutdown(wait=True)

    # The matching multiscale pyramid, mean-downsampled from the corrected level 0 with
    # the same cumulative factors the input dataset uses.
    factors = source_pyramid_factors(view, setup)
    level0 = ts.open({"driver": _SPEC[view["output_format"]]["driver"],
                      "kvstore": {"driver": "file", "path": out_path}},
                     context=ctx, open=True, read=True).result()
    for level in range(1, len(factors)):
        ds = ts.open({"driver": "downsample",
                      "base": level0.spec(),
                      "downsample_factors": _in_order(factors[level], out_order),
                      "downsample_method": "mean"},
                     context=ctx, open=True, read=True).result()
        lvl, _, _ = open_output_array(
            view, setup, level, list(ds.domain.shape), dtype_name, ctx)
        ltxn = ts.Transaction(atomic=False)
        print(f"correct: setup {setup} level {level} {tuple(ds.domain.shape)}")
        await lvl.with_transaction(ltxn).write(ds)
        await ltxn.commit_async()

    write_group_metadata(view, setup, factors)
    timing["t_total"] = time.perf_counter() - t_start
    print("SPOTLIGHT_TIMING " + json.dumps(timing), flush=True)
    print(f"correct: done setup {setup}, mode={mode}, {len(factors)} level(s)")


def apply_correction_chunked(cfg, setup, mode="auto"):
    """Correct one setup. `mode` is one of `MODES`."""
    asyncio.run(_run(cfg, setup, mode))
