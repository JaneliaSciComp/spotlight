"""Stage: per-tile intensity statistics, and what they say about a tile.

One job per setup. Reads the tile at a downsampled level, splits background from
foreground with Otsu, and writes `{results_root}/intensity_stats/setup{N}.json` for
`aggregate` to reduce.

Also owns `_classify`, which turns those numbers into the three tile kinds the correction
branches on -- "bimodal" (real background to protect), "uniform" (no clean split, correct
every voxel), "empty" (nothing worth correcting) -- and the thresholds that decide
between them. The thresholds live beside the classifier that reads them, with the
measurements that set them: a tuning constant separated from its benchmark is worse than
no constant, because the next person re-tunes it blind.

Always per-tile: each tile is thresholded on its own. `aggregate` then uses those
per-tile thresholds to compare OVERLAPPING tiles and solve a gain -- which is what keeps
a tile's own content out of the gain estimate.

`_classify` sorts a tile into the three strategies the correction branches on:

* **bimodal** -- a clean background/foreground split. Mask the background out and rescale
  the foreground only.
* **uniform** -- no clean split but real signal throughout (whole-tile std above the noise
  floor). Rescale every voxel.
* **empty** -- no clean split and std near the noise floor: an all-noise tile. Passed
  through unmodified, and dropped from the gain solve entirely.

These thresholds are in GRAY LEVELS, and they stay valid under joint BaSiC correction
because the flat field is normalised around 1, so correcting preserves the data's overall
scale.

CLI: `python -m spotlight int-stats <setup>`.
"""

import json
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import tensorstore as ts
from skimage.filters import threshold_li, threshold_otsu

from .config import camera_of, stats_path
from .fields import basic_model
from .formats import _input_location, _SPEC, canonical_view
from .stores import _context, source_pyramid_factors


#     A tile below this is passed through uncorrected and excluded from the gain
#     solve and target.
#   - `_pair_gain_constraint`: overlap representativeness -- foreground / overlap
#     voxels. A background-dominated overlap with only a stray bright speck fails
#     this, so its (misregistration-sensitive) ratio never enters the solve.
# A fraction, not an absolute count, because absolute foreground scales with tile
# size / `stats_scale` -- a fixed count is meaningless across resolutions.
# Empirically the real tiles here sit above ~2% foreground while empty/speck tiles
# form an isolated spike below ~0.1%. Whole-tile use needs `n_voxels` in the stats
# dict (written by `_compute_stats`; `cmd_aggregate` backfills it from the on-disk
# tile shape for older stats files, so no stats re-run is required).
MIN_FG_FRACTION = 0.001

# Absolute foreground VOXEL floor for a per-pair overlap median to be statistically
# stable (a median's standard error scales like 1/sqrt(n)). This is a sample-count
# concern, independent of overlap size, so it stays an absolute count and gates
# `_pair_gain_constraint` ALONGSIDE the MIN_FG_FRACTION representativeness check.
MIN_FOREGROUND = 256

# Safety net on the per-tile rescale: never amplify by more than this. `scale =
# target_std / std_i` is normally exactly 1/gain (see `_apply`), so this only binds if
# a tile solves to an absurd gain (under 1/8) -- e.g. an all-noise tile whose Otsu
# split bisected the noise, leaving a tiny std that would blow the tile from black to
# white. `_classify` sends those to "empty" via MIN_UNIFORM_STD, so this is a backstop
# for anything that slips through.
MAX_SCALE = 8.0

# Whole-tile std separates a genuinely empty/all-noise tile from one that is real
# signal everywhere with no separable black region -- `_classify` uses `empty_area` for
# the background question, but both of those cases have little background, so std is
# what tells them apart -- empirically, empty/noise tiles sit at ~0.3-0.6
# while the weakest real-signal tile in this dataset sits at ~40, a >60x gap.
# Above this, treat the whole tile as foreground; below it, pass through
# unmodified as an empty tile.
MIN_UNIFORM_STD = 10.0


# Empty frame area at or above which `_classify` calls a tile "bimodal" (it has real
# background to protect) rather than "uniform" (correct every pixel in it). On RID19
# s15 the two populations sit at 0.0000-0.0006 and 0.294-0.462, so anything in between
# separates them; 0.02 keeps ~30x margin above the fullest tiles.
MIN_BACKGROUND_AREA = 0.02

# Percentile of the per-frame-pixel background means taken as the dataset's additive
# offset (see `_background_level`). A low percentile, not the mean, because the
# measurement is contaminated one-sidedly by tissue proximity.


# ─── stats stage ─────────────────────────────────────────────────────────────────


def open_downsampled(cfg, setup):
    """Open a downsampled view of a setup's input array.

    Uses the prebuilt pyramid level `stats_scale` if it exists on disk; otherwise
    opens scale 0 and wraps it in TensorStore's mean-downsample driver so stats
    stay cheap without requiring a prebuilt pyramid.
    """
    fmt = cfg["input_format"]
    spec = _SPEC[fmt]
    scale = cfg["stats_scale"]
    context = _context()

    def _open(path):
        return ts.open({
            "driver": spec["driver"],
            "kvstore": {"driver": "file", "path": path},
        }, context=context, open=True, read=True).result()

    path, order = _input_location(cfg, setup, scale)
    if (Path(path) / spec["meta"]).exists():
        return _open(path)

    path0, order0 = _input_location(cfg, setup, 0)
    arr0 = _open(path0)
    f = 2 ** scale  # in-plane factor; Z left alone so thin stacks survive
    if order0 == "tczyx":
        factors = [1, 1, 1, f, f]
    elif order0 == "zyx":
        factors = [1, f, f]
    else:
        factors = [f, f, 1]   # xyz
    return ts.open({
        "driver": "downsample",
        "base": arr0.spec(),
        "downsample_factors": factors,
        "downsample_method": "mean",
    }, context=context, open=True, read=True).result()


THRESHOLD_MODES = ("otsu", "li", "pooled")


def threshold_mode(cfg):
    """`tile_threshold`, validated once: a float, or one of THRESHOLD_MODES.

    Validated in one place so every stage rejects the same typos with the same message.
    A silent fallback to "otsu" is the worst outcome, because it is indistinguishable
    from a threshold that was already right.
    """
    mode = cfg.get("tile_threshold")
    if mode is None or mode == "":
        return "otsu"
    if not isinstance(mode, str):
        return float(mode)
    if mode not in THRESHOLD_MODES:
        raise ValueError(f"tile_threshold={mode!r} must be one of {THRESHOLD_MODES} "
                         f"or a number")
    return mode


def threshold_values(mode, values):
    """Apply `mode` to a flat array of intensities. Returns `(threshold, how)`.

    The shared core of the per-tile split and the emptiness stage's pooled one, so the
    setting cannot mean one thing in one stage and something else in the other. `mode`
    here is only "otsu", "li", or a number -- "pooled" is resolved by the caller, since
    it names the emptiness stage's OWN result and would be circular there.
    """
    if not isinstance(mode, str):
        return float(mode), f"tile_threshold={float(mode):g}"
    if mode == "li":
        return float(threshold_li(_histogrammable(values))), "li"
    return float(threshold_otsu(values)), "otsu"


THRESHOLD_METHODS = ("otsu", "li")


def threshold_catalogue(vol, chosen_mode=None, chosen=None):
    """Every method's threshold for one tile, so later stages need not re-read it.

    The stats stage is the only place that holds a whole tile in memory, and each method
    costs tens of milliseconds against a read of seconds. Computing them all here means
    `gain_floor = "li"` can pick a different split from `tile_threshold` without opening
    a single array -- and without the two settings having to agree.

    `chosen_mode`/`chosen` avoid recomputing the one `resolve_threshold` already did.
    """
    out = {}
    for m in THRESHOLD_METHODS:
        out[m] = float(chosen) if m == chosen_mode else threshold_values(m, vol)[0]
    return out


def resolve_threshold(cfg, setup, vol):
    """The background/foreground split for one tile, honouring `tile_threshold`.

    `"otsu"` (default) is per-tile Otsu. It is the right choice when every tile holds a
    similar mix of tissue and background, and the wrong one when they do not: Otsu splits
    whatever it is given, so a tile packed with bright structure gets a HIGH threshold and
    its sparse neighbour a low one. Measured on a 3-tile worm: 245, 580, 1621 -- a 6.6x
    spread on one specimen imaged one way.

    That spread is not cosmetic, because this one number drives three things:
      * `n_foreground`, hence `_classify` -- a tile whose threshold is too high falls
        under MIN_FG_FRACTION, is called "empty", and is dropped from the gain solve and
        passed through UNCORRECTED.
      * the gain solve's common floor (see `aggregate._common_floor`).
      * the apply stage's mask -- a "bimodal" tile is rescaled only above its threshold,
        so a threshold set too high leaves most of the tile untouched. This is the usual
        cause of a seam that survives correction.

    WHY Otsu misses on sparse data, since the fix follows from it: Otsu maximizes
    `w0*w1*(u0-u1)^2`, which assumes two classes of COMPARABLE SIZE. On a tile that is
    ~99.9% background with a long bright tail, `w1` is negligible for any threshold in
    the tail while `(u0-u1)^2` keeps growing quadratically as the threshold climbs it, so
    the product keeps rising and the optimum lands deep in the tail -- measured at the
    99.6th-99.9th percentile of all three tiles of that worm. It therefore tracks how
    BRIGHT each tile's brightest structure is (max 22168/13296/6118 against thresholds
    1621/580/245), which is exactly the quantity that differs between tiles and exactly
    what must not be used to compare them. This is a property of the histogram, not of
    the binning: 256 bins and 65536 bins give the identical answer.

    Alternatives:
      * `"li"` -- minimum cross-entropy (Li & Lee 1993; iterative form Li & Tam 1998).
        Minimizes the KL divergence between the image and its two-level reconstruction,
        whose fixed point `t = (u1-u0)/(ln u1 - ln u0)` is the LOGARITHMIC mean of the
        two class means -- where isodata's is the ARITHMETIC mean. The logarithmic mean
        is always the smaller, and the gap widens as the classes separate (274.6 vs
        628.2 on that worm's tile 0), which is exactly why Li resists a heavy tail: its
        criterion grows logarithmically with class separation where Otsu's grows
        quadratically. Verified against the identity on real tiles to within 0.4 counts.
      * a NUMBER -- one floor for every tile, so the spread is 1.0x by construction.
        Blunt, and usually the right answer once you have looked at the data: comparing
        tiles needs cross-tile consistency more than per-tile optimality.
      * `"pooled"` -- the dataset-wide `empty_threshold` from the emptiness stage.
        Consistent across tiles by construction, but it is still an OTSU split, so it
        inherits the failure above and is dominated by the brightest tile in the pool
        (1114 on that worm, still far too high). Not a fix for sparse data.
    """
    mode = threshold_mode(cfg)
    if mode != "pooled":
        return threshold_values(mode, vol)
    from .quantiles import empty_threshold
    thr = empty_threshold(cfg, camera_of(cfg, setup))
    if thr is None:
        raise RuntimeError(
            "tile_threshold='pooled' needs the emptiness stage's empty_threshold; run "
            "`python -m spotlight emptiness` first, or set tile_threshold to a number")
    return float(thr), "pooled empty_threshold"


def _histogrammable(vol):
    """An integer view of `vol`, for `threshold_li` only.

    skimage's `threshold_li` picks its implementation off the dtype: integer input gets a
    ONE-PASS histogram and then iterates over ~65k bin centres, while floating input
    re-scans the whole array every iteration (`image[image > t].mean()`), allocating a
    fresh copy of the selected voxels each time. Measured on a 19.9M-voxel tile: 64 ms
    integer vs 1000 ms float, and 190 MiB of peak allocation against a 76 MiB array. It
    scales linearly, so a 268M-voxel tile pays seconds and gigabytes.

    That matters here because `_read_tile_volume` returns float32 whenever `apply_basic`
    is on (BaSiC's `correct()` casts), so the expensive path is reached by a config flag
    that has nothing to do with thresholding. Rounding to uint16 costs nothing real: these
    are photon counts, and sub-count precision cannot move a background/foreground split.
    Only Li needs this -- `threshold_otsu` histograms either way.
    """
    if vol.dtype.kind in "iu":
        return vol
    out = np.clip(vol, 0, 65535)
    np.rint(out, out=out)
    return out.astype(np.uint16)


def _compute_stats(vol, thr=None):
    """Otsu background/foreground split + whole-volume stats for one tile's voxels.

    `thr` overrides the per-tile Otsu split; see `resolve_threshold`."""
    # Whole-tile (unmasked) stats -- used by the apply stage's "uniform" branch
    # for tiles with no clean background/foreground split (see MIN_UNIFORM_STD).
    vol64 = vol.astype("float64")
    all_mean = float(vol64.mean())
    all_std = float(vol64.std())

    thr = float(threshold_otsu(vol)) if thr is None else float(thr)
    fg = vol[vol > thr].astype("float64")
    bg = vol[vol <= thr].astype("float64")
    n_fg = int(fg.size)
    mean = float(fg.mean()) if n_fg else float("nan")
    std = float(fg.std()) if n_fg else float("nan")
    bg_mean = float(bg.mean()) if bg.size else float("nan")
    bg_std = float(bg.std()) if bg.size else float("nan")
    # How many background sigmas the foreground sits above the background. For a
    # genuinely bimodal tile this is large; for an all-noise tile Otsu splits the
    # noise and the two class means are ~1 sigma apart -> small separation.
    separation = ((mean - bg_mean) / (bg_std + 1e-6)
                  if np.isfinite(mean) and np.isfinite(bg_mean) and np.isfinite(bg_std)
                  else float("nan"))

    return {
        "mean": mean,
        "std": std,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "all_mean": all_mean,
        "all_std": all_std,
        "separation": separation,
        "threshold": thr,
        "n_foreground": n_fg,
        "n_voxels": int(vol.size),   # tile total, for the MIN_FG_FRACTION emptiness gate
    }


def _merge_tile_stats(cfg, setup, fields):
    """Merge `fields` into a tile's stats JSON, preserving whatever is already there.

    Two stages write this file and neither owns all of it: `stats` writes the Otsu split
    and the tile's own moments, `emptiness` writes `empty_area` and `background_level`.
    They also run in that reverse order -- `emptiness` first, because the BaSiC pass
    needs its measurements -- so a wholesale overwrite by either one silently drops the
    other's fields. Merging is what lets them be rerun independently."""
    out = stats_path(cfg, setup)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception as err:
            print(f"could not read {out} to merge into ({err}); rewriting it")
    existing.update(fields)
    out.write_text(json.dumps(existing))
    return out


def _write_stats(cfg, setup, stats):
    return _merge_tile_stats(cfg, setup, {
        "setup": int(setup), "stats_scale": cfg["stats_scale"],
        "apply_basic": cfg["apply_basic"], **stats})


def _read_tile_volume(cfg, setup):
    """One setup's downsampled volume, canonical (Z, Y, X), BaSiC-corrected when
    `apply_basic` -- so the Otsu split and the foreground stats describe the same
    values `apply` will write (see the module docstring)."""
    _, order = _input_location(cfg, setup, cfg["stats_scale"])
    arr = canonical_view(open_downsampled(cfg, setup), order)
    vol = arr.read(order="C").result()
    model = basic_model(cfg, setup, vol.shape[1:])
    return vol if model is None else model.correct(vol)


def cmd_stats(cfg, setup):
    print(dt.now(), f"stats: opening setup {setup} ({cfg['input_format']}, "
                    f"apply_basic={cfg['apply_basic']})", flush=True)
    vol = _read_tile_volume(cfg, setup)
    print(dt.now(), f"stats: read volume {vol.shape}", flush=True)

    thr, how = resolve_threshold(cfg, setup, vol)
    stats = _compute_stats(vol, thr)
    stats["threshold_source"] = how
    stats["thresholds"] = threshold_catalogue(vol, how if how in THRESHOLD_METHODS else None,
                                              thr)
    out = _write_stats(cfg, setup, stats)
    print(dt.now(), f"stats: setup {setup} mean={stats['mean']:.2f} std={stats['std']:.2f} "
                    f"all_mean={stats['all_mean']:.2f} all_std={stats['all_std']:.2f} "
                    f"thr={stats['threshold']:.1f} ({how}; "
                    f"{', '.join(f'{m}={v:.0f}' for m, v in stats['thresholds'].items())}) "
                    f"n_fg={stats['n_foreground']} "
                    f"sep={stats['separation']:.1f} -> {out}", flush=True)


# ─── apply stage ─────────────────────────────────────────────────────────────────




# Every module constant above is a DEFAULT, not a fixed law: `limits(cfg)` lets a toml
# override each one. They stay module constants so the functions below are callable
# without a config (tests, notebooks) and so the default is stated once.
_LIMIT_KEYS = {
    "min_tile_fraction": "MIN_FG_FRACTION",
    "min_tile_foreground": "MIN_FOREGROUND",
    "min_uniform_std": "MIN_UNIFORM_STD",
    "min_background_area": "MIN_BACKGROUND_AREA",
    "max_gain_scale": "MAX_SCALE",
}


def limits(cfg=None):
    """The tile-classification thresholds, with any `cfg` overrides applied.

    Returned as a plain dict rather than read from `cfg` at each site so that
    `_classify` and `_enough_foreground` stay callable with no config at all -- they are
    used from tests and from `bench/compare_thresholds.py`, neither of which has one.
    """
    out = {k: globals()[const] for k, const in _LIMIT_KEYS.items()}
    if cfg:
        for k in _LIMIT_KEYS:
            v = cfg.get(k)
            if v is not None:
                out[k] = float(v)
    return out


def _enough_foreground(s, lim=None):
    """Whole-tile emptiness test: is the foreground at least MIN_FG_FRACTION of the
    tile's voxels? Uses `n_voxels` when present (written by `_compute_stats`, or
    backfilled by `cmd_aggregate` from the tile shape); falls back to the legacy
    absolute MIN_FOREGROUND count only for old stats files lacking `n_voxels`."""
    lim = limits() if lim is None else lim
    nv = s.get("n_voxels")
    if nv:
        return s["n_foreground"] >= lim["min_tile_fraction"] * nv
    return s["n_foreground"] >= lim["min_tile_foreground"]


def _classify(s, lim=None):
    """Pick the apply stage's correction strategy for one tile, from its empty frame
    area (`empty_area`, injected by `cmd_aggregate` from the `emptiness` stage).

    - "bimodal": the tile really does contain background (`empty_area` at or above
      MIN_BACKGROUND_AREA), so mask the background out and rescale the foreground only.
    - "uniform": real signal fills the whole tile with no separable background, so
      treat every pixel as foreground (`thr = -inf` in `_apply`) and rescale using
      whole-tile mean/std.
    - "empty": too few candidate foreground pixels, or a whole-tile std near the noise
      floor -- an actually empty/all-noise tile. Passed through unmodified.

    The bimodal-vs-uniform question is "does this tile contain genuine background", and
    `empty_area` answers it directly. This used to be decided by testing Otsu's
    class-mean gap against a fixed 50-count bar, which cannot answer it: where a tile has no background
    Otsu bisects *tissue*, and a within-tissue split easily puts the class means far
    apart. Measured on RID19 s15, all 20 tiles cleared that bar -- including 15 with
    `empty_area` under 0.0006 that are plainly "uniform" -- and the two groups' gap
    ranges OVERLAP (77-90 for tiles with real background, 52-103 for those without), so
    no threshold on that statistic separates them. `empty_area` separates them by 469x.

    Getting this wrong is not cosmetic: a "bimodal" tile is masked at its own Otsu
    threshold, so those 15 tiles were receiving no intensity rescale at all across
    67-87% of their voxels.
    """
    lim = limits() if lim is None else lim
    empty_area = s.get("empty_area")
    if empty_area is None or not np.isfinite(empty_area):
        raise RuntimeError(
            f"tile {s.get('setup')} has no empty_area; run the emptiness stage "
            f"(`python -m spotlight emptiness`) before aggregate -- classification "
            f"depends on it")
    if not _enough_foreground(s, lim) or not (
            np.isfinite(s["mean"]) and np.isfinite(s["std"]) and s["std"] > 0):
        return "empty"
    all_std = s.get("all_std", float("nan"))
    if not (np.isfinite(all_std) and all_std >= lim["min_uniform_std"]):
        return "empty"
    return "bimodal" if empty_area >= lim["min_background_area"] else "uniform"
