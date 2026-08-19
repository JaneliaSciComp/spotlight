"""Stage: how much of the camera frame each tile leaves empty, and what is in it.

One job per dataset, not per setup: the intensity threshold it splits on is pooled across
every tile, and a per-tile threshold would not be comparable between tiles.

Produces three things other stages need, all merged into the per-tile stats JSONs and a
fraction map beside them:
  * `empty_threshold` -- the dataset-wide background/specimen split. The quantile stats
    pass needs it to measure its own per-quantile background profile.
  * `background_level` -- the scalar additive offset, for the darkfield override and for
    un-mixing a raw-slice stack.
  * `empty_area` per tile and the per-frame-pixel empty FRACTION map, which is what
    un-mixing a quantile stack inverts.

CLI: `python -m spotlight emptiness`.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import tensorstore as ts

from .config import camera_groups, empty_fraction_path, stats_path, tile_list
from .formats import (_input_location, _SPEC, canonical_shape, canonical_view,
                      write_plane_tiff)
from .stores import _atomic_write_json, _context, source_pyramid_shapes
from .tilestats import (MIN_FG_FRACTION, _merge_tile_stats, threshold_mode,
                        threshold_values)


# Percentile of the per-frame-pixel background means taken as the dataset's additive
# offset (see `_background_level`). A low percentile, not the mean, because the
# measurement is contaminated one-sidedly by tissue proximity.
#
# This is the SCALAR offset, used for the darkfield override (a darkfield is one
# pedestal) and for un-mixing a `raw_stack_mode` stack. It is NOT what un-mixing a
# quantile stack subtracts: there the background rises with quantile index, and the
# profile is measured on the Julia side by `_accumulate_background_quantiles!`, off the
# same `OrderStats` the stack is built from. The only thing this stage owes that
# measurement is `empty_threshold` below, which it writes into every tile's stats.
BACKGROUND_PERCENTILE = 5

# Ceiling on the pooled subsample the dataset-wide Otsu threshold is computed from.
# Otsu needs a representative intensity DISTRIBUTION, not every voxel, and its answer is
# stable well below this -- but the pooled array is held whole while it is concatenated,
# so without a cap it grows with the tile count. `_empty_areas` pass 2 already documents
# this hazard for its own partials; pass 1 has the same one. Override: SPOTLIGHT_OTSU_VOXELS.
OTSU_SAMPLE_VOXELS = int(os.getenv("SPOTLIGHT_OTSU_VOXELS", 32_000_000))


# ── `emptiness` stage: which tiles are too partly-empty to FIT the BaSiC field ─────
#
# A frame pixel counts as EMPTY when under this fraction of its z-column rises above
# the dataset-wide intensity threshold -- i.e. that position in the camera frame is
# essentially never looking at specimen in this tile.
EMPTY_OCCUPANCY_FLOOR = 0.02


# ─── emptiness stage: how much frame each tile leaves empty, and what is there ───


def _basic_input_cfg(cfg):
    """`cfg` with the input redirected to `input_basic_path`, so the reader helpers
    (`_input_location` and friends, which resolve against `input_intensity_path`) read
    the RAW store the BaSiC fit itself consumes.

    Deliberately the raw store, not a corrected one: choosing the fit set from
    BaSiC-corrected data would pick tiles using a field derived from the very tiles
    being judged."""
    out = dict(cfg)
    basic = cfg.get("input_basic_path", "").replace("$HOME", os.path.expanduser("~"))
    out["input_intensity_path"] = basic or cfg["input_intensity_path"]
    return out


def _select_level(cfg, setup, min_frame=96):
    """Coarsest input pyramid level whose frame is still at least `min_frame` across.

    Empty area is a coarse, whole-frame property, so there is no reason to read full
    resolution -- but the frame must stay big enough that a empty region spans many
    pixels. Verified on RID19 s15 that levels 3 and 4 (288 and 144 across) give the
    same verdict."""
    shapes = source_pyramid_shapes(cfg, setup)
    if not shapes:
        return 0
    for lvl in range(len(shapes) - 1, -1, -1):
        _, y, x = canonical_shape(shapes[lvl], _input_location(cfg, setup, lvl)[1])
        if min(y, x) >= min_frame:
            return lvl
    return 0


def _read_tile(cfg, setup, level):
    """One tile as a canonical (Z, Y, X) array at an existing pyramid `level`.

    Opens the level directly rather than going through `open_downsampled`, which is
    pinned to `stats_scale` and falls back to wrapping level 0 in a downsample driver
    -- that would read full resolution to answer a coarse question."""
    path, order = _input_location(cfg, setup, level)
    arr = ts.open({
        "driver": _SPEC[cfg["input_format"]]["driver"],
        "kvstore": {"driver": "file", "path": path},
    }, context=_context(), open=True, read=True).result()
    return np.asarray(canonical_view(arr, order).read(order="C").result())


def _emptiness_workers(n_tiles):
    """Thread count for the two `_empty_areas` read passes.

    `LSB_DJOB_NUMPROC` like the other stages, but with a smaller default: unlike `stats`
    and `apply`, this stage is normally launched by Julia's `measure_emptiness()` on
    whatever node the user is sitting on rather than through bsub, so the variable is
    usually absent and the default is what actually runs. Capped at the tile count
    because a thread per tile is the most that can help.
    """
    n = int(os.getenv("LSB_DJOB_NUMPROC", str(min(16, (os.cpu_count() or 8)))))
    return max(1, min(n, n_tiles))


def _empty_areas(cfg, setups):
    """`(empty, threshold, level)` -- per-setup fraction of the CAMERA FRAME that is
    effectively never occupied by specimen, plus the dataset-wide intensity threshold
    and pyramid level used.

    Why this statistic, and not a per-tile foreground fraction:

      * The bias being removed comes from emptiness that is STRUCTURED IN THE CAMERA
        FRAME. A tile uniformly 50% empty (sparse labelling everywhere) does not skew
        any frame pixel's quantiles; a tile empty across its upper y skews exactly the
        pixels in that band. So the thing to measure is how much of the frame a tile
        never illuminates -- not how much signal it holds.
      * The threshold is GLOBAL, from a pooled subsample across all tiles. A per-tile
        Otsu threshold (as `_compute_stats` computes, for its own different purpose) is
        not comparable between tiles: on a tile with no separable background it bisects
        *tissue*, so its "foreground fraction" is an arbitrary within-tissue split. On
        RID19 s15 that made the per-tile number rank partly-empty tiles as the FULLEST
        ones -- exactly backwards.
      * Tile occupancy also varies smoothly across a mosaic (s15 ranges 0.36 to 0.89
        top row to bottom), so there is no two-population structure to cluster on;
        attempts to find a boundary in it correctly report none. Empty area does not
        inherit that gradient, because "is any part of the frame unused" is a different
        question from "how much signal is there".
    """
    level = _select_level(cfg, setups[0])
    workers = _emptiness_workers(len(setups))
    # Both passes read every tile, so both are threaded. Threads rather than processes:
    # the time goes into tensorstore reads and whole-array numpy, both of which drop the
    # GIL, and a process pool would have to ship the decoded volumes back.
    #
    # No progress line: this stage is deliberately two lines per camera whatever the tile
    # count, and `cmd_emptiness` reports the thread count on the second of them.

    # Pass 1: pooled subsample -> one threshold for the whole dataset. Strided rather
    # than complete so this stays cheap; the threshold only needs to separate the dark
    # floor from specimen, which a coarse sample resolves fine.
    #
    # Every tile's share is capped so the pooled array does not grow with the tile count:
    # `list(pool.map(...))` holds all of them at once and `concatenate` then doubles it,
    # which is the same pile-up pass 2 below goes out of its way to avoid. A further
    # uniform stride keeps the subsample deterministic and unbiased in z.
    # `tile_threshold` governs this split too, so one setting means one thing everywhere:
    # `li` here and `otsu` per tile would judge emptiness on a different population than
    # the stats stage measures. "pooled" NAMES this value, so it resolves to otsu here --
    # anything else would be circular.
    mode = threshold_mode(cfg)
    pooled_mode = "otsu" if mode == "pooled" else mode

    if not isinstance(pooled_mode, str):
        # A fixed threshold needs no sample, so the whole first pass -- a strided read of
        # every tile -- is skipped. That is a real saving, not a shortcut: pass 1 exists
        # only to produce this number.
        threshold, how = float(pooled_mode), f"tile_threshold={float(pooled_mode):g}"
        print(dt.now(), f"emptiness: threshold {threshold:.1f} from {how} "
                        f"(no sampling pass needed)", flush=True)
    else:
        per_tile = max(1, int(cfg.get("otsu_sample_voxels")
                              or OTSU_SAMPLE_VOXELS) // len(setups))

        def _sample(s):
            v = _read_tile(cfg, s, level)[::4].ravel()[::7]
            if v.size > per_tile:
                v = v[::-(-v.size // per_tile)]
            return v

        with ThreadPoolExecutor(max_workers=workers) as pool:
            samples = list(pool.map(_sample, setups))
        pooled = np.concatenate(samples)
        del samples
        threshold, how = threshold_values(pooled_mode, pooled)
        print(dt.now(), f"emptiness: threshold {threshold:.1f} ({how}) from "
                        f"{pooled.size} pooled voxels at level {level} "
                        f"({per_tile} max/tile)", flush=True)
        del pooled

    # Pass 2: per-frame-pixel occupancy over the z column, then the empty fraction --
    # and, over the empty pixels only, the background level (see `_background_level`).
    #
    # Each worker folds its own (Y, X) partials into the shared accumulators under a lock
    # and then drops them, so live memory is bounded by `workers` partials rather than by
    # the tile count -- returning them for the caller to reduce would pile up ~0.7 MB per
    # tile, which is over a gigabyte on a 1600-tile mosaic.
    empty = {}
    acc = {"bg_sum": None, "bg_cnt": None, "phi": None}
    lock = threading.Lock()

    def _measure(s):
        v = _read_tile(cfg, s, level)
        occupancy = (v > threshold).mean(axis=0)          # (Y, X)
        is_empty = occupancy < float(cfg.get("empty_occupancy_floor")
                                     or EMPTY_OCCUPANCY_FLOOR)
        # Per-frame-pixel mean of sub-threshold voxels, kept only where the whole
        # column is unoccupied. Accumulated over every tile: a full tile has almost no
        # empty pixels and so contributes almost nothing, which is exactly right --
        # only tiles with real empty regions can witness the dark floor.
        below_mask = v < threshold
        below = np.where(below_mask, v, 0.0)
        n = below_mask.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(n > 0, below.sum(axis=0) / np.maximum(n, 1), np.nan)
        m = np.where(is_empty & np.isfinite(m), m, 0.0)
        c = (is_empty & (n > 0)).astype(np.float64)
        with lock:
            empty[s] = float(is_empty.mean())
            acc["bg_sum"] = m if acc["bg_sum"] is None else acc["bg_sum"] + m
            acc["bg_cnt"] = c if acc["bg_cnt"] is None else acc["bg_cnt"] + c
            acc["phi"] = (is_empty.astype(np.float64) if acc["phi"] is None
                          else acc["phi"] + is_empty)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # list() so an exception in any tile propagates instead of being swallowed.
        list(pool.map(_measure, setups))

    # phi[y,x]: the FRACTION of this camera's tiles that are empty at that frame
    # position. This is what makes the bias removable -- see `_write_empty_fraction`.
    # Exact regardless of completion order: it is a sum of booleans. `bg_sum` is a float
    # sum, so threading reorders it, but only in the last bits of a percentile input.
    phi = acc["phi"] / len(setups)
    return empty, threshold, level, _background_level(acc["bg_sum"], acc["bg_cnt"],
                                        float(cfg.get("background_percentile")
                                              or BACKGROUND_PERCENTILE)), phi


def _background_level(bg_sum, bg_cnt, percentile=BACKGROUND_PERCENTILE):
    """The dataset's additive background level in counts, or None if nothing observed it.

    Taken as the BACKGROUND_PERCENTILE of the per-frame-pixel background means, not
    their mean. The measurement is contaminated one-sidedly: a frame pixel counts as
    empty if under EMPTY_OCCUPANCY_FLOOR of its column clears the threshold, so pixels
    near the specimen still admit dim tissue and scattered light, which only ever pushes
    the value UP. Measured on RID19 s15 that contamination rises steadily with y --
    124 counts at the top of the frame where the specimen is absent, ~180 at the deepest
    rows that still qualify -- so the mean (137) over-estimates while a low percentile
    (121) tracks the uncontaminated top. A low percentile also keeps the corrected
    background positive: subtracting 137 drove it to -6..-9 counts, which then clips.

    Not a per-pixel map: only ~46% of the frame is ever empty in any tile, so there is
    nothing to measure over the rest, and the apparent y-gradient is the specimen
    arriving rather than the offset changing -- extrapolating it would subtract signal
    as if it were offset. A no-illumination dark acquisition is the only way to get
    honest full-frame structure."""
    if bg_sum is None or bg_cnt is None:
        return None
    seen = bg_cnt > 0
    if not seen.any():
        return None
    vals = bg_sum[seen] / bg_cnt[seen]
    return float(np.percentile(vals, percentile))


def _write_empty_fraction(path, phi):
    """Write the per-frame-pixel empty fraction as a float32 TIFF for `save_qstack`.

    `phi` is measured on canonical (Z, Y, X) volumes, so it arrives (Y, X) and is written
    exactly so -- like every other plane beside a camera. `empty_fraction_map` swaps it into
    the qstack's order on read.

    This is the map that lets the emptiness bias be REMOVED rather than avoided.
    `mstat_by_chunk` merges per-setup `OrderStats` by averaging their sorted buffers,
    so a tile that is empty at a frame position contributes the background level at
    EVERY quantile, and the pooled stack reads

        q_observed[k] = phi*bg[k] + (1 - phi)*q_true[k]

    which inverts exactly. Note `bg[k]`, not a scalar: the same averaging makes an empty
    column's contribution a mean of per-block ORDER STATISTICS, so it rises with quantile
    index. That profile is measured on the Julia side (`_accumulate_background_quantiles!`)
    because its width depends on the stats pass's block geometry; this stage supplies only
    phi and the scalar offset. Subsetting avoids the bias by discarding tiles; un-mixing
    removes it while keeping them, and keeps the darkfield's evidence in the fit.

    Only phi's VARIATION across the frame can bias a flat field, so a mosaic whose
    emptiness is spatially uniform once pooled has nothing here to correct.

    Written at the coarse pyramid level the measurement ran at; `save_qstack` upsamples
    it to the qstack's frame size. That is fine because phi varies on the scale of tile
    overlap, not per pixel -- and unlike the background profile, occupancy survives
    downsampling, which is why the two are measured at different levels."""
    try:
        write_plane_tiff(path, phi)
    except ImportError:
        print(f"  (no {path.name}: tifffile not installed; qstack un-mixing unavailable)")
        return False
    return True


def cmd_emptiness(cfg):
    """Measure how much of each camera's frame the tiles leave empty, and what the
    detector reads there.

    Writes, per camera:

      * `empty_area` and `background_level` into each tile's own
        `intensity_stats/setup{N}.json`, merged with whatever the `stats` stage has
        already put there. `_classify` reads `empty_area` straight out of the stats it
        already loads, and Julia reads `background_level` for the darkfield override and
        the qstack un-mixing.
      * `camera{N}/empty_fraction.tif` -- the per-frame-pixel empty fraction, used
        by `save_qstack` to un-mix the quantile stack (`unmix_empty_fraction!`).

    Must run BEFORE the `stats` stage's results are needed and before the BaSiC quantile
    pass, since both consume its output; `_merge_tile_stats` is what makes that ordering
    safe in either direction.

    Reads the raw input directly at a coarse pyramid level, so it needs neither the
    `stats` stage nor its `apply_basic` bookkeeping.
    """
    bcfg = _basic_input_cfg(cfg)
    for cam, setups in enumerate(camera_groups(cfg), start=1):
        setups = list(setups)
        empty, threshold, level, background_level, phi = _empty_areas(bcfg, setups)
        # Two lines per camera, whatever the tile count. The per-tile numbers all go to
        # each tile's own stats JSON, which is what every consumer reads; a dataset can
        # carry thousands of tiles and a per-tile log is unreadable at that size.
        vals = np.array([empty[s] for s in setups], dtype=float)
        bg = "NOT MEASURABLE" if background_level is None else f"{background_level:.1f} counts"
        print(f"camera {cam}: {len(setups)} tiles | empty area min {100 * vals.min():.1f}% "
              f"median {100 * np.median(vals):.1f}% max {100 * vals.max():.1f}% | "
              f"background {bg}")

        for s in setups:
            _merge_tile_stats(cfg, s, {"empty_area": empty[s],
                                       "background_level": background_level,
                                       "empty_threshold": threshold,
                                       "empty_level": level})
        phi_path = empty_fraction_path(cfg, cam - 1)
        phi_path.parent.mkdir(parents=True, exist_ok=True)
        wrote_phi = _write_empty_fraction(phi_path, phi)
        print(f"  thr {threshold:.0f} (level {level}, {_emptiness_workers(len(setups))} "
              f"threads) | merged into {len(setups)} tile stats"
              + (f" | phi map max {phi.max():.3f} -> {phi_path.name}" if wrote_phi else ""))
        if background_level is None:
            print("  WARNING no tile has an empty region here, so the additive offset "
                  "cannot be observed. basic_unmix_empty and override_darkfield both "
                  "need it.")
