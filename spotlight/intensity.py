"""Per-setup ("tile") intensity correction for OME-ZARR / N5 data.

Optionally JOINT with the BaSiC flat-field/dark-field correction (config key
`apply_basic`, auto-enabled when `run_basic()`'s fields are present -- see the
"joint BaSiC correction" section at the bottom of this docstring), so a dataset
that needs both corrections is read and rewritten exactly ONCE instead of twice.

Three stages:

  stats <setup>          read a downsampled version of the setup, find the
                          background via an Otsu threshold, and write the
                          foreground mean/std to
                          {results_root}/intensity_stats/setup{N}.json
                          One LSF array job per setup, always per-tile: each
                          tile is thresholded on its own. `aggregate` then uses
                          these per-tile thresholds to compare overlapping tiles
                          and solve a per-tile gain (see below).

  aggregate               single reduce job (no array). Solves a per-TILE gain
                          from tile overlaps, then writes the global target:
                          1. Reads every setup's own per-tile stats (Otsu
                             threshold, mean/std) and classifies via `_classify`.
                          2. Discovers every physically-overlapping tile pair
                             from the SpimData2 `dataset.xml` (config key
                             `dataset_xml`) -- within-camera AND cross-camera
                             (`_all_overlap_pairs`).
                          3. Per pair, a robust log-gain constraint from the two
                             tiles' foreground medians in the shared region
                             (`_pair_gain_constraint`) -- two overlapping tiles
                             image the SAME tissue there, so a mismatch is sensor
                             gain, not content. `gain_estimator` = "intersection"
                             (default) compares the matched voxels both tiles call
                             foreground (unbiased when registered); "independent"
                             compares each tile's own distribution (tolerates
                             misregistration, but a threshold/population mismatch
                             can bias it).
                          4. Solves a gain g_s by regularized global least squares
                             (`_solve_tile_gains`): correction is raw/g_s.
                             `gain_grouping` = "camera" (default) shares one gain
                             across a camera's tiles -- it robustly removes the
                             sensor-level step between cameras (many overlaps per
                             camera pin it, so it needs no shrinkage); "tile"
                             solves one per tile (opt-in, keep `gain_lambda` >~ 0.1
                             to damp drift). Regularization is `gain_lambda`, whose
                             lam -> 0 limit is a pure gauge anchor -- camera
                             defaults to a tiny lam, tile to 0.1. Overlap-driven
                             gains preserve real texture -- neighbours whose shared
                             region agrees keep equal gains.
                          5. Writes {results_root}/tile_gains.json (gains, grouping/
                             estimator, per-camera summary) and
                             {results_root}/intensity_target.json: each setup's
                             stats plus "corrected_mean"/"corrected_std" that
                             encode g_s so the (unchanged) `apply` computes
                             raw/g_s. target_mean/target_std = median over tiles
                             of the gain-equalized mean/std.

  apply <setup>           read the target and rescale this setup by its per-tile
                          gain (out = raw/g_s, via the stored corrected_mean/std),
                          writing the result to {output_intensity_path}/...
                          Three strategies, picked per tile by `_classify`: a clean
                          background/foreground split masks the background out and
                          rescales the foreground only; a tile with no clean split but
                          real signal throughout (whole-tile std above the noise floor)
                          rescales every pixel; a tile with no clean split and std near
                          the noise floor is an empty/all-noise tile and is passed
                          through unmodified. One LSF array job per setup.

Input and output formats are independent (config keys `input_format` /
`output_format`, each one of: n5, zarr2, zarr3_unsharded, zarr3, zarr3_zyx,
zarr3_raw). The data is normalised to a canonical (Z, Y, X) volume internally, so
any input format can be written to any output format. Config is read from
LocalPreferences.toml in the current working directory -- run this script from
the experiment directory that holds the toml you want. The streaming sharded I/O
processes one output shard at a time (read, correct, write under its own
transaction), with `asyncio.gather` + a semaphore keeping several shards
in flight concurrently -- not a single whole-array virtual_chunked write,
which serializes the copy and leaves cores idle (see `_apply`).

Joint BaSiC correction (`apply_basic`)
--------------------------------------
When enabled, every stage first applies the per-camera BaSiC flat-field /
dark-field correction to the voxels it reads:

    basic(raw) = max((raw - dark_field) / flat_field, 0)

using `{results_root}/camera{N}/{Flat,Dark}-field.tif` -- exactly the fields
`run_basic()` writes and exactly the math (including `nonneg_offset = 0` and the
clamp at 0) that `apply_correction_chunked` in src/BigFlatFieldIlluminator.jl
applies, so a joint run matches running the two corrections back to back. Camera
membership comes from `camera_groups` (mirrors Julia's `camera_setups`), and the
fields live in 1-based `camera{N}` directories, matching what `run_basic()` wrote.

With this on, `input_intensity_path` must point at the RAW dataset (the same
store as `input_basic_path`), NOT at the output of a previous Julia
`apply_correction()` run -- otherwise the flat/dark correction is applied twice.
The point is to skip that intermediate dataset entirely: one read of the raw
data, one write of the final result, and only ONE rounding to uint16 (the
two-pass route quantizes the flat-field-corrected data to uint16 before the
intensity rescale ever sees it).

`stats` and `aggregate` correct the voxels they read too, not just `apply`. They
have to: the Otsu split, the per-tile foreground mean/std, and the overlap
medians the per-tile gain is solved from must all describe the same values
`apply` will write. Flat-field vignetting differs across a tile's field of view,
so gains solved on raw voxels would partly chase vignetting instead of sensor
gain. For the downsampled levels `stats`/`aggregate` read, the fields are
mean-downsampled by the same per-axis factor (the fields vary slowly in-plane,
so downsample-then-divide and divide-then-downsample agree to well within the
noise). The stats and target files record which mode produced them, and
`aggregate`/`apply` refuse to mix modes rather than silently combining
raw-derived stats with basic-corrected voxels.

The intensity thresholds below (MIN_UNIFORM_STD, ...) are in gray
levels and stay valid under joint correction because BaSiC's flat field is
normalized around 1, so the correction preserves the data's overall scale.

`zarr3_raw` (input only) reads TensorSwitch's actual on-disk OME-ZARR layout: a
3-D array declared with the non-standard axes order [x,y,z] (x first, not last),
nested under a "raw/" subgroup with "s"-prefixed scale dirs (s0, s1, ...) -
distinct from this script's own flat "0", "1", ... scale dirs. `zarr3_zyx`
(typically the output format) writes the standard 3-D NGFF spatial order
[z,y,x] (x last) instead of this script's default 5-D (t,c,z,y,x), so tools
downstream of this correction step that assume that standard order need no
changes to read the result.

Usage:
    python intensity_correction.py stats <setup>
    python intensity_correction.py aggregate
    python intensity_correction.py apply <setup>
If the setup/camera argument is omitted it falls back to
ENV["LSB_JOBINDEX"] - 1 (0-indexed).
"""

import asyncio
import json
import os
import sys
import threading
import time
import tomllib
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import tensorstore as ts
from scipy import sparse
from scipy.sparse.linalg import lsqr
from skimage.filters import threshold_otsu

# A tile's (or an overlap's) foreground must be at least this FRACTION of its
# voxels to count as real signal rather than empty/speck. Used in two places:
#   - `_classify`: whole-tile emptiness -- n_foreground / tile voxels.
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

# ── `emptiness` stage: which tiles are too partly-empty to FIT the BaSiC field ─────
#
# A frame pixel counts as EMPTY when under this fraction of its z-column rises above
# the dataset-wide intensity threshold -- i.e. that position in the camera frame is
# essentially never looking at specimen in this tile.
EMPTY_OCCUPANCY_FLOOR = 0.02

# Empty frame area at or above which `_classify` calls a tile "bimodal" (it has real
# background to protect) rather than "uniform" (correct every pixel in it). On RID19
# s15 the two populations sit at 0.0000-0.0006 and 0.294-0.462, so anything in between
# separates them; 0.02 keeps ~30x margin above the fullest tiles.
MIN_BACKGROUND_AREA = 0.02

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


FORMATS = ("n5", "zarr2", "zarr3_unsharded", "zarr3", "zarr3_zyx", "zarr3_raw")

# OME-NGFF axes for the 5-D (T, C, Z, Y, X) zarr layout. Spatial axes are in
# micrometers; the scale transforms (see _ngff_datasets) use unit voxel size.
_AXES = [
    {"type": "time", "name": "t", "unit": "millisecond"},
    {"type": "channel", "name": "c"},
    {"type": "space", "name": "z", "unit": "micrometer"},
    {"type": "space", "name": "y", "unit": "micrometer"},
    {"type": "space", "name": "x", "unit": "micrometer"},
]

# OME-NGFF axes for the 3-D (Z, Y, X) zarr layout ("zarr3_zyx" output format) -
# the standard NGFF spatial axis order (x last), unlike "zarr3_raw"'s [x,y,z].
_AXES_ZYX = [
    {"type": "space", "name": "z", "unit": "micrometer"},
    {"type": "space", "name": "y", "unit": "micrometer"},
    {"type": "space", "name": "x", "unit": "micrometer"},
]


def _ngff_datasets(factors, order="tczyx"):
    """OME-NGFF `datasets` list from cumulative (fz, fy, fx) factors per level.

    scale = cumulative downsample factor; translation = half-pixel offset
    max(scale/2 - 0.5, 0) per axis (the standard OME-NGFF mean-downsample offset).
    `order` picks the scale/translation array length: 5-D (t,c,z,y,x) or 3-D (z,y,x).
    """
    out = []
    for level, (fz, fy, fx) in enumerate(factors):
        scale = ([float(fz), float(fy), float(fx)] if order == "zyx"
                  else [1.0, 1.0, float(fz), float(fy), float(fx)])
        translation = [max(s / 2.0 - 0.5, 0.0) for s in scale]
        out.append({
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": scale},
                {"type": "translation", "translation": translation},
            ],
        })
    return out

# ─── config ────────────────────────────────────────────────────────────────────


def load_config(require_intensity_io=True):
    """Read the [spotlight] table from LocalPreferences.toml in the
    current working directory (so each experiment's own toml can be used just by
    running from that experiment's directory, independent of where this script
    itself lives on disk).

    Falls back to the pre-rename `[BigFlatFieldIlluminator]` table so experiment
    directories written by the Julia package keep working untouched; `set_config`
    only ever writes `[spotlight]`.

    `require_intensity_io=False` tolerates a toml with no `input_intensity_path` /
    `output_intensity_path`, for stages that read neither. The `emptiness` stage is the
    case that matters: it is driven from the BaSiC side (`create_quartile_histograms`
    invokes it), and a BaSiC-only experiment has no reason to have configured the
    per-setup intensity pipeline's I/O at all. Demanding those keys there turned a
    working BaSiC run into a KeyError."""
    with open(Path.cwd() / "LocalPreferences.toml", "rb") as f:
        _tables = tomllib.load(f)
    cfg = _tables.get("spotlight") or _tables["BigFlatFieldIlluminator"]

    def expand(s):
        return s.replace("$HOME", os.path.expanduser("~"))

    for key in ("input_intensity_path", "output_intensity_path"):
        if key in cfg:
            cfg[key] = expand(cfg[key])
        elif require_intensity_io:
            raise KeyError(f"{key} is required in LocalPreferences.toml for this stage")
        else:
            cfg[key] = ""
    cfg["results_root"] = expand(cfg["results_root"])
    cfg.setdefault("format", "zarr2")
    cfg.setdefault("stats_scale", 2)
    cfg["input_format"] = cfg.get("input_format", cfg["format"])
    cfg["output_format"] = cfg.get("output_format", cfg["format"])
    for key in ("input_format", "output_format"):
        if cfg[key] not in FORMATS:
            raise ValueError(f"{key}={cfg[key]!r} must be one of {FORMATS}")

    # SpimData2/BigDataViewer dataset.xml -- consumed by the `aggregate` stage's gain
    # calibration, to find each camera border's physically-overlapping tile pairs, and
    # by `emptiness` for its tile map. Sits alongside the zarr/n5 store (the *_path keys
    # are the store itself, e.g. ".../dataset.ome.zarr", so dataset.xml is one level
    # up). Falls back to input_basic_path so a BaSiC-only toml still resolves it.
    store = cfg["input_intensity_path"] or expand(cfg.get("input_basic_path", ""))
    if store:
        cfg.setdefault("dataset_xml", str(Path(store).parent / "dataset.xml"))

    # Joint BaSiC flat/dark-field correction (see the module docstring). Default
    # on when run_basic()'s fields exist for the first camera -- if they're there,
    # the intent is to correct with them, and doing it here means the data is
    # rewritten once rather than twice. Set `apply_basic = false` in the toml to
    # run intensity-only (e.g. when input_intensity_path already points at a
    # Julia apply_correction() output, which is flat/dark-corrected already).
    if "apply_basic" not in cfg:
        cfg["apply_basic"] = all(p.exists() for p in basic_field_paths(cfg, 0))
    cfg["apply_basic"] = bool(cfg["apply_basic"])
    return cfg


def tile_list(cfg):
    """All setups to process. setup_ids (flattened) if given, else 0..last_setup."""
    ids = cfg.get("setup_ids", [])
    if ids:
        return [s for group in ids for s in group]
    return list(range(cfg["last_setup"] + 1))


def camera_groups(cfg):
    """Setups grouped by camera: setup_ids as-is if given (one group per camera,
    e.g. `[[171,...,194], [201,...,204]]`), else contiguous `setups_per_camera`-sized
    chunks of `0..last_setup`. Mirrors `camera_setups()` in src/BigFlatFieldIlluminator.jl
    so the two pipelines agree on which setups belong to which camera."""
    ids = cfg.get("setup_ids", [])
    if ids:
        return [list(group) for group in ids]
    per_cam = cfg.get("setups_per_camera", 1)
    last = cfg["last_setup"]
    return [list(range(start, min(start + per_cam, last + 1)))
            for start in range(0, last + 1, per_cam)]


# ─── joint BaSiC flat-field / dark-field correction ─────────────────────────────
#
# Applies `run_basic()`'s per-camera fields to the voxels every stage reads, so
# the flat/dark correction and the per-tile intensity correction happen in one
# read/write of the data instead of two (see the module docstring). Off unless
# `apply_basic` (auto-detected in `load_config`).


def basic_field_paths(cfg, camera):
    """(flat, dark) TIFF paths for a 0-based camera index. `run_basic()` writes
    1-BASED `camera{N}` directories (Julia's `camera_setups(config)[camera]` is
    indexed from 1), while `camera_groups` here is 0-based -- hence the +1."""
    d = Path(cfg["results_root"]) / f"camera{camera + 1}"
    return d / "Flat-field.tif", d / "Dark-field.tif"


def camera_of(cfg, setup):
    """0-based camera index owning `setup`, from `camera_groups`."""
    for cam, group in enumerate(camera_groups(cfg)):
        if setup in group:
            return cam
    raise RuntimeError(f"setup {setup} belongs to no camera group; check setup_ids / "
                       f"setups_per_camera / last_setup")


def _read_field_tiff(path):
    """One BaSiC field as a 2-D float32 array. `save_basic_field` writes a
    single-page Gray{Float32} TIFF, so any of these readers gives the same
    (rows, cols) array Julia's `load` returns."""
    try:
        import tifffile
        img = np.asarray(tifffile.imread(str(path)))
    except ImportError:
        from skimage.io import imread
        img = np.asarray(imread(str(path)))
    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D field, got shape {img.shape}")
    return img.astype("float32")


def _oriented_field(field, full_yx, path, in_plane_order):
    """A field in canonical (Y, X).

    `save_qstack_camera` writes the qstack -- and so BaSiC its fields -- in
    whatever in-plane order that camera's stats used: (Y, X) for the zarr
    formats, (X, Y) for n5. `in_plane_order` is that expectation, derived from
    the input format rather than guessed from the shape, so a square tile
    (Y == X, where both orientations "fit") still resolves correctly. The
    transpose is accepted with a warning only when the shape rules the expected
    order out -- e.g. fields produced by a differently-formatted earlier run.
    """
    Y, X = full_yx
    expected = (Y, X) if in_plane_order == "yx" else (X, Y)
    if field.shape == expected:
        return field if in_plane_order == "yx" else np.ascontiguousarray(field.T)
    if field.shape == expected[::-1]:
        print(dt.now(), f"WARNING: {path} shape {field.shape} is the transpose of the "
                        f"expected {in_plane_order} order {expected}; transposing", flush=True)
        return np.ascontiguousarray(field.T) if in_plane_order == "yx" else field
    raise ValueError(f"{path}: field shape {field.shape} matches neither the tile's "
                     f"expected {in_plane_order} plane {expected} nor its transpose")


def _block_mean_2d(a, fy, fx, out_yx):
    """Mean-downsample a 2-D field by (fy, fx), cropped/edge-padded to `out_yx`.

    `np.add.reduceat` rather than a reshape-mean so a ragged final block (when
    the plane isn't a multiple of the factor) is averaged over its real extent
    instead of forcing an exact division. The crop/pad then absorbs the
    off-by-one between our ceil-division block count and whatever rounding the
    on-disk pyramid level used -- edge padding, since a flat field's border
    value is the best estimate for a border row that only exists on one side.
    """
    if fy == 1 and fx == 1:
        m = a.astype("float32", copy=True)
    else:
        ys = np.arange(0, a.shape[0], fy)
        xs = np.arange(0, a.shape[1], fx)
        s = np.add.reduceat(np.add.reduceat(a.astype("float64"), ys, axis=0), xs, axis=1)
        cy = np.diff(np.append(ys, a.shape[0]))[:, None]
        cx = np.diff(np.append(xs, a.shape[1]))[None, :]
        m = (s / (cy * cx)).astype("float32")
    Y, X = out_yx
    m = m[:Y, :X]
    if m.shape != (Y, X):
        m = np.pad(m, ((0, max(0, Y - m.shape[0])), (0, max(0, X - m.shape[1]))), mode="edge")
    return m


class BasicModel:
    """One camera's BaSiC fields at one pyramid level, canonical (Y, X)."""

    __slots__ = ("flat", "dark")

    def __init__(self, flat, dark):
        self.flat, self.dark = flat, dark

    def sub(self, y_range, x_range):
        """The model restricted to a (y, x) sub-block of this level's plane."""
        return BasicModel(self.flat[y_range[0]:y_range[1], x_range[0]:x_range[1]],
                          self.dark[y_range[0]:y_range[1], x_range[0]:x_range[1]])

    def correct(self, canon):
        """max((raw - dark) / flat, 0) as float32 for a canonical (Z, Y, X) block.

        The (Y, X) fields broadcast over the leading Z axis. `nonneg_offset = 0`
        plus the clamp at 0 matches `apply_correction_chunked`'s call in
        src/BigFlatFieldIlluminator.jl, so a joint run reproduces the two-pass
        result (up to the intermediate uint16 rounding the joint route skips).
        """
        c = np.asarray(canon).astype("float32")
        c -= self.dark
        c /= self.flat
        np.clip(c, 0, None, out=c)
        return c


_BASIC_CACHE = {}


def basic_model(cfg, setup, level_yx):
    """`BasicModel` for `setup` at a level whose in-plane shape is `level_yx`, or
    None when `apply_basic` is off.

    The (fy, fx) downsample factor is derived from the level's own shape against
    the full-res shape, so this is correct both for a prebuilt pyramid level and
    for `open_downsampled`'s on-the-fly downsample driver (whose in-plane factor
    is 2**stats_scale but which leaves Z alone). Cached per (camera, level shape)
    -- the fields are per-camera, so an array job's every shard shares one.
    """
    if not cfg["apply_basic"]:
        return None
    cam = camera_of(cfg, setup)
    shapes = source_pyramid_shapes(cfg, setup)
    if not shapes:
        raise RuntimeError(f"no level-0 array found for setup {setup} under "
                           f"{cfg['input_intensity_path']} ({cfg['input_format']})")
    full_yx = tuple(shapes[0][1:])
    level_yx = tuple(int(v) for v in level_yx)
    key = (cfg["results_root"], cam, full_yx, level_yx)
    hit = _BASIC_CACHE.get(key)
    if hit is not None:
        return hit

    flat_p, dark_p = basic_field_paths(cfg, cam)
    for p in (flat_p, dark_p):
        if not p.exists():
            raise RuntimeError(f"apply_basic is on but {p} is missing; run run_basic() "
                               f"for camera {cam + 1}, or set apply_basic = false")
    fy = max(round(full_yx[0] / level_yx[0]), 1)
    fx = max(round(full_yx[1] / level_yx[1]), 1)
    # The fields were written by the Julia stats/BaSiC run over this same store
    # (joint mode reads the raw dataset), so the input format also fixes the
    # in-plane order those fields are stored in.
    ipo = "xy" if cfg["input_format"] == "n5" else "yx"

    def prep(p):
        return _block_mean_2d(_oriented_field(_read_field_tiff(p), full_yx, p, ipo),
                              fy, fx, level_yx)

    model = BasicModel(prep(flat_p), prep(dark_p))
    _BASIC_CACHE[key] = model
    return model


def _check_basic_mode(cfg, recorded, what):
    """Refuse to combine stats/target files with voxels corrected differently.

    Per-tile stats (Otsu threshold, foreground mean/std) and the solved gains are
    only meaningful for the pixel values they were measured on, so a target built
    from raw stats must not drive a joint `apply` (or vice versa). Files written
    before `apply_basic` existed carry no flag; they were raw-derived, so treat a
    missing flag as False."""
    if bool(recorded) != cfg["apply_basic"]:
        raise RuntimeError(
            f"{what} was written with apply_basic={bool(recorded)} but this run has "
            f"apply_basic={cfg['apply_basic']}; re-run the earlier stage(s) so the "
            f"stats describe the same voxel values that will be written")


# ─── overlap-based per-tile gain compensation ───────────────────────────────────
#
# Estimates a per-TILE multiplicative gain from every pair of tiles that
# physically overlap -- within a camera AND across camera borders (panorama
# "exposure compensation", Brown & Lowe-style gain solving). Two overlapping
# tiles image the SAME physical tissue in their shared region, so a mismatch in
# each tile's own foreground distribution there is pure sensor gain, not content.
# One regularized global least-squares solve turns all pairwise ratios into one
# gain g_i per tile; the correction is raw/g_i (see `cmd_aggregate`).
#
# This is deliberately per-tile, not per-camera: a single gain per camera can't
# fix a seam where two adjacent border tiles each deviate differently from their
# own camera's average, and can't capture a gain that drifts across a camera's
# field of view (both observed in this data). Per-tile gains driven by overlap
# ratios preserve real tile-to-tile texture -- neighbours whose shared region
# agrees get equal gains, so genuine content differences survive -- while a
# per-tile-to-target match (each tile forced to one target mean) would erase it.
# No camera-adjacency assumption: overlaps are discovered from the geometry.


def _parse_dataset_xml(cfg):
    return ET.parse(cfg["dataset_xml"]).getroot()


def _view_setup_sizes(xml_root):
    """setup id -> (sizeX, sizeY, sizeZ) in pixels, from <ViewSetups>/<ViewSetup>."""
    sizes = {}
    for vs in xml_root.findall(".//ViewSetups/ViewSetup"):
        setup = int(vs.findtext("id"))
        sx, sy, sz = (int(v) for v in vs.findtext("size").split())
        sizes[setup] = (sx, sy, sz)
    return sizes


def _view_registration_transforms(xml_root):
    """setup id -> composed 4x4 (pixel -> world) affine.

    Each <ViewRegistration> lists its <ViewTransform>s outermost-first (e.g.
    "Stitching Transform", then "Translation to Regular Grid", then
    "calibration" last) -- BDV/BigStitcher convention composes them as
    M_total = M_0 @ M_1 @ ... @ M_last, i.e. the LAST-listed transform
    (calibration: pixel index -> physical units) is applied FIRST to a raw
    pixel coordinate. Points are (x, y, z, 1) column vectors, matching the
    <affine> string's row-major (x-row, y-row, z-row) layout.
    """
    transforms = {}
    for vr in xml_root.findall(".//ViewRegistrations/ViewRegistration"):
        if vr.get("timepoint") != "0":
            continue
        setup = int(vr.get("setup"))
        total = np.eye(4)
        for vt in vr.findall("ViewTransform"):
            if vt.get("type") != "affine":
                continue
            vals = [float(v) for v in vt.findtext("affine").split()]
            mat = np.eye(4)
            mat[:3, :] = np.array(vals).reshape(3, 4)
            total = total @ mat
        transforms[setup] = total
    return transforms


def _pixel_corners(size_xyz):
    """8 corners of the (0,0,0)..size_xyz pixel-space box, shape (8, 3)."""
    sx, sy, sz = size_xyz
    return np.array([[x, y, z]
                      for x in (0, sx) for y in (0, sy) for z in (0, sz)])


def _setup_world_bbox(setup, sizes, transforms):
    """(mins_xyz, maxs_xyz) world-space axis-aligned bbox for one setup."""
    corners = _pixel_corners(sizes[setup])
    homo = np.hstack([corners, np.ones((8, 1))])
    world = (transforms[setup] @ homo.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def _all_overlap_pairs(cfg):
    """(pairs, sizes, transforms) where pairs is every physically-overlapping
    tile pair across the WHOLE dataset -- `(setup_a, setup_b, world_bbox)` for
    each pair whose world-space bounding boxes intersect, within-camera and
    cross-camera alike. Within-camera overlaps matter: they are what let the
    per-tile solve tell gain from content (a tile dim from its own gain reads
    low against its same-camera neighbours on shared tissue; one dim from
    content agrees with them). Overlap is discovered purely from the geometry,
    so no camera-adjacency assumption is made.

    The pairwise test is O(N^2) but vectorized per row (a few seconds for a few
    thousand tiles); only genuinely-overlapping pairs are materialized."""
    xml_root = _parse_dataset_xml(cfg)
    sizes = _view_setup_sizes(xml_root)
    transforms = _view_registration_transforms(xml_root)
    setups = [s for s in tile_list(cfg) if s in sizes and s in transforms]

    bb = {s: _setup_world_bbox(s, sizes, transforms) for s in setups}
    mins = np.array([bb[s][0] for s in setups])
    maxs = np.array([bb[s][1] for s in setups])

    pairs = []
    for i in range(len(setups)):
        lo = np.maximum(mins[i], mins[i + 1:])   # (N-i-1, 3)
        hi = np.minimum(maxs[i], maxs[i + 1:])
        ok = np.all(hi > lo, axis=1)
        for k in np.nonzero(ok)[0]:
            pairs.append((setups[i], setups[i + 1 + k], (lo[k], hi[k])))
    return pairs, sizes, transforms


def _world_bbox_to_pixels(setup, world_bbox, sizes, transforms):
    """World-space (mins, maxs) -> that setup's own full-res pixel index
    (mins, maxs), clipped to [0, size)."""
    mins, maxs = world_bbox
    corners = np.array([[x, y, z]
                         for x in (mins[0], maxs[0])
                         for y in (mins[1], maxs[1])
                         for z in (mins[2], maxs[2])])
    homo = np.hstack([corners, np.ones((8, 1))])
    inv = np.linalg.inv(transforms[setup])
    pixel = (inv @ homo.T).T[:, :3]
    px_min = np.clip(pixel.min(axis=0), 0, None)
    px_max = np.clip(pixel.max(axis=0), None, np.array(sizes[setup]))
    return px_min, px_max


def _overlap_ranges(setup, world_bbox, sizes, transforms, factor, shape):
    """The setup's downsampled (z, y, x) ranges for a world-space bbox, from a
    precomputed (fz, fy, fx) `factor` (level `stats_scale`) and downsampled
    `shape` -- no array re-opening / pyramid re-read per call. Used by the
    many-pair aggregate solve, where re-opening per pair would dominate the
    runtime."""
    px_min, px_max = _world_bbox_to_pixels(setup, world_bbox, sizes, transforms)
    fz, fy, fx = factor
    x_range = (int(px_min[0] // fx), int(np.ceil(px_max[0] / fx)))
    y_range = (int(px_min[1] // fy), int(np.ceil(px_max[1] / fy)))
    z_range = (int(px_min[2] // fz), int(np.ceil(px_max[2] / fz)))
    Z, Y, X = shape
    return ((max(0, z_range[0]), min(Z, z_range[1])),
            (max(0, y_range[0]), min(Y, y_range[1])),
            (max(0, x_range[0]), min(X, x_range[1])))


def _pair_gain_constraint(setup_a, setup_b, world_bbox, sizes, transforms, cache, order,
                          estimator="intersection"):
    """One log-gain constraint for an overlapping tile pair. Returns
    `(a, b, log(med_a) - log(med_b), weight)` -- the target for `log g_a - log
    g_b` (so raw/g equalizes the two on shared tissue) -- or None if the overlap
    has too little foreground to trust. `cache[s]` carries the tile's open array,
    downsample factor, shape, and Otsu threshold.

    `estimator` selects how the two medians are taken over the shared region:
      * "intersection" (default): median over the voxels BOTH tiles call
        foreground -- matched voxels, so the ratio is the true gain even when the
        two tiles' Otsu thresholds differ. Unbiased when registration is good; a
        few-voxel misregistration does bias it, which is why "independent" exists.
      * "independent": each tile's foreground distribution above the common floor,
        compared independently -- robust to registration error (no pixel-for-pixel
        match), at the cost of a possible threshold/population bias. On this data,
        switching independent -> intersection recovered the true camera step
        (9/10 ratio 1.51 -> 1.66) while the common floor below already removed the
        threshold-mismatch bias, so intersection is the default.

    Both tiles are gated at a COMMON floor `max(thr_a, thr_b)`, NOT each at its
    own Otsu threshold. The two per-tile thresholds routinely differ by
    hundreds of gray levels (Otsu places the split per tile, so a tile with more
    mid-intensity tissue lands lower); gating each tile at its own threshold then
    compares medians over DIFFERENT intensity populations of the SAME tissue --
    the lower-threshold tile sweeps in a band of dimmer voxels its neighbor
    excludes, dragging its median down and manufacturing a spurious gain
    difference (observed: a tile reading ~identically to its neighbors got a
    0.92 gain purely from a ~200-level threshold gap). The shared floor makes
    both medians span the same population, so the ratio measures gain, not the
    threshold mismatch."""
    ca, cb = cache[setup_a], cache[setup_b]
    za, ya, xa = _overlap_ranges(setup_a, world_bbox, sizes, transforms, ca["factor"], ca["shape"])
    zb, yb, xb = _overlap_ranges(setup_b, world_bbox, sizes, transforms, cb["factor"], cb["shape"])
    va = ca["arr"][za[0]:za[1], ya[0]:ya[1], xa[0]:xa[1]].read(order="C").result()
    vb = cb["arr"][zb[0]:zb[1], yb[0]:yb[1], xb[0]:xb[1]].read(order="C").result()
    # BaSiC-correct the overlap voxels when joint (cache["basic"] is None
    # otherwise): each tile images the shared tissue through a different part of
    # its own field of view, so an uncorrected ratio would carry the two tiles'
    # vignetting difference into the gain solve.
    if ca["basic"] is not None:
        va = ca["basic"].sub(ya, xa).correct(va)
        vb = cb["basic"].sub(yb, xb).correct(vb)
    thr = max(ca["thr"], cb["thr"])
    # Hybrid gate: enough absolute samples for a stable median (MIN_FOREGROUND) AND
    # foreground is a representative fraction of the overlap (MIN_FG_FRACTION), so a
    # background-dominated overlap with a stray speck doesn't contribute a ratio.
    if estimator == "intersection":
        sh = tuple(min(p, q) for p, q in zip(va.shape, vb.shape))
        va, vb = va[:sh[0], :sh[1], :sh[2]], vb[:sh[0], :sh[1], :sh[2]]
        mask = (va > thr) & (vb > thr)
        fg_a, fg_b = va[mask], vb[mask]
    elif estimator == "independent":
        fg_a, fg_b = va[va > thr], vb[vb > thr]
    else:
        raise ValueError(f"gain_estimator={estimator!r} must be 'intersection' or 'independent'")
    if (fg_a.size < MIN_FOREGROUND or fg_b.size < MIN_FOREGROUND
            or fg_a.size < MIN_FG_FRACTION * va.size
            or fg_b.size < MIN_FG_FRACTION * vb.size):
        return None
    med_a, med_b = float(np.median(fg_a)), float(np.median(fg_b))
    if not (med_a > 0 and med_b > 0):
        return None
    return (setup_a, setup_b, float(np.log(med_a) - np.log(med_b)),
            float(min(fg_a.size, fg_b.size)))


def _solve_tile_gains(constraints, setups, lam=0.1, group_of=None):
    """Multiplicative gains from all pairwise overlap constraints, by regularized
    global least squares in log space:

        min  Σ_(a,b) w_ab (log g_A - log g_B - d_ab)^2  +  lam Σ_g (log g_g)^2

    where d_ab = log(med_a) - log(med_b), w_ab is the overlap foreground size
    (normalized by its median so `lam` is scale-free), and A/B are the GROUPS
    tiles a/b belong to (`group_of[a]`, `group_of[b]`). `group_of=None` solves
    one gain per tile (group = the tile itself); passing a setup->camera map
    solves one gain per CAMERA (every tile in a camera shares it). Either way the
    return maps every setup to its solved gain.

    The regularization pulls log-gains toward 0 -- it fixes the otherwise-free
    global gauge (overlaps constrain only differences), damps groups with
    few/noisy constraints, and pins a group with no constraints to exactly 1.0.
    In the lam -> 0 limit it is a pure gauge anchor (the minimum-norm, sum-zero
    solution): per-camera it is safe to take lam tiny (each camera has many
    constraints, so the data dominates -- lam 1e-6 and a hard gauge agree to
    RMS 0); per-tile keep lam >~ 0.01 to damp the smooth drift null-mode. Solved
    as one sparse system via `scipy.sparse.linalg.lsqr`."""
    if group_of is None:
        group_of = {s: s for s in setups}
    groups = sorted({group_of[s] for s in setups})
    idx = {g: i for i, g in enumerate(groups)}
    n = len(groups)
    # within-group pairs (same camera) carry no relative-gain info; drop them
    used = [(group_of[a], group_of[b], d, w) for a, b, d, w in constraints
            if group_of[a] != group_of[b]]
    w_med = float(np.median([w for *_, w in used])) if used else 1.0

    rows, cols, data, rhs = [], [], [], []
    r = 0
    for ga, gb, d, w in used:
        sw = np.sqrt(w / w_med)
        rows += [r, r]; cols += [idx[ga], idx[gb]]; data += [sw, -sw]
        rhs.append(sw * d); r += 1
    for g in groups:  # regularization + gauge
        rows.append(r); cols.append(idx[g]); data.append(np.sqrt(lam))
        rhs.append(0.0); r += 1

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(r, n))
    sol = lsqr(mat, np.array(rhs), atol=1e-10, btol=1e-10)[0]
    return {s: float(np.exp(sol[idx[group_of[s]]])) for s in setups}



def _context():
    n = int(os.getenv("LSB_DJOB_NUMPROC", "24"))
    # File opens are latency-bound, not CPU-bound: an n5 source is unsharded with
    # many small blocks, so one shard-sized read touches hundreds of files.
    # Keep many in flight to hide network-FS open latency even when #cores is low;
    # data_copy stays ~#cores (CPU-bound). Override via BFF_IO_CONCURRENCY.
    io = int(os.getenv("BFF_IO_CONCURRENCY", str(n*64)))
    return {
        "data_copy_concurrency": {"limit": n},
        "file_io_concurrency": {"limit": io},
        "cache_pool": {"total_bytes_limit": 512 * 2**20},
    }


def stats_path(cfg, setup):
    return Path(cfg["results_root"]) / "intensity_stats" / f"setup{setup}.json"


# ─── format abstraction ──────────────────────────────────────────────────────────
#
# Internally everything is a canonical (Z, Y, X) volume. Three stored orders:
#   "tczyx" - 5-D OME-ZARR (T, C, Z, Y, X), the zarr2/zarr3/zarr3_unsharded formats.
#   "xyz"   - 3-D (X, Y, Z), n5's convention and also "zarr3_raw" - the actual
#             on-disk layout TensorSwitch writes today (axes declared [x,y,z],
#             nested under a "raw/" subgroup with "s"-prefixed scale dirs - see
#             https://github.com/JaneliaSciComp/tensorswitch). Input-only: this is
#             the non-standard order the rest of the pipeline (BigDataViewer/
#             BigStitcher) doesn't handle without a patch.
#   "zyx"   - 3-D (Z, Y, X), the standard NGFF spatial axis order (x last),
#             "zarr3_zyx" - already canonical, so `canonical_view` is the
#             identity. Use this as the output format so downstream tools
#             that assume the standard convention need no changes.
_SPEC = {
    "n5":              dict(driver="n5",    meta="attributes.json", order="xyz",
                             path="{base}/setup{setup}/timepoint0/s{scale}"),
    "zarr2":           dict(driver="zarr2", meta=".zarray",         order="tczyx",
                             path="{base}/s{setup}-t0.zarr/{scale}"),
    "zarr3_unsharded": dict(driver="zarr3", meta="zarr.json",       order="tczyx",
                             path="{base}/s{setup}-t0.zarr/{scale}"),
    "zarr3":           dict(driver="zarr3", meta="zarr.json",       order="tczyx",
                             path="{base}/s{setup}-t0.zarr/{scale}"),
    "zarr3_zyx":       dict(driver="zarr3", meta="zarr.json",       order="zyx",
                             path="{base}/s{setup}-t0.zarr/{scale}"),
    "zarr3_raw":       dict(driver="zarr3", meta="zarr.json",       order="xyz",
                             path="{base}/s{setup}-t0.zarr/raw/s{scale}"),
}


def _path(fmt, root, setup, scale):
    return _SPEC[fmt]["path"].format(base=root, setup=setup, scale=scale)


def _output_path(fmt, root, setup, scale):
    """Path for one level of an OUTPUT array under `root` (`output_intensity_path`).
    Always the plain bare-integer-scale convention -- output is always written
    by this codebase's own writers, never read from a pre-existing layout, so
    there's no non-standard convention to resolve here (unlike `_input_location`)."""
    if fmt == "n5":
        return f"{root}/setup{setup}/timepoint0/s{scale}"
    return f"{root}/s{setup}-t0.zarr/{scale}"


def _resolve_zarr3(root, setup, scale):
    """Resolve (path, order) for one level of a zarr3-driven INPUT array from
    its OME-NGFF group metadata (`s{setup}-t0.zarr/zarr.json`'s
    `multiscales[0].datasets[scale].path` and the resolved array's own
    `dimension_names`), instead of assuming a fixed scale-directory name or
    axis order -- so a non-standard on-disk layout (e.g. TensorSwitch's
    dataset nested under a `raw/` subgroup, axes declared [x, y, z] -- see
    https://github.com/JaneliaSciComp/tensorswitch) is discovered, not
    special-cased. Returns None if there's no OME group metadata, or no such
    level -- the caller then falls back to the static `_SPEC`/`_path`
    convention. Output paths are unaffected (`open_output_array` always
    writes its own known layout, never reads a pre-existing one).
    """
    group_dir = f"{root}/s{setup}-t0.zarr"
    try:
        with open(f"{group_dir}/zarr.json") as f:
            group = json.load(f)
        multiscale = group["attributes"]["ome"]["multiscales"][0]
        datasets = multiscale["datasets"]
    except (FileNotFoundError, KeyError, IndexError):
        return None
    if scale >= len(datasets):
        return None
    path = f'{group_dir}/{datasets[scale]["path"]}'
    with open(f"{path}/zarr.json") as f:
        arr_meta = json.load(f)
    names = arr_meta.get("dimension_names")
    if names is None and len(arr_meta["shape"]) == 3:
        # This codebase's own writers (`_write_ngff_metadata`) don't stamp
        # `dimension_names` on the array itself for 3-D outputs -- fall back
        # to the group's declared OME `axes` order.
        names = [a["name"] for a in multiscale["axes"]]
    if names is None:
        order = "tczyx"
    else:
        names = tuple(n.lower() for n in names)
        if names == ("z", "y", "x"):
            order = "zyx"
        elif names == ("x", "y", "z"):
            order = "xyz"
        elif len(names) == 5:
            order = "tczyx"
        else:
            raise ValueError(f"Unsupported zarr3 axis order {names} at {path}")
    return path, order


def _input_location(cfg, setup, scale):
    """(path, order) for one level of the configured INPUT array. Resolved
    from OME-NGFF group metadata for zarr3-driven formats (see
    `_resolve_zarr3`); falls back to the static `_SPEC`/`_path` convention
    when no such metadata is found, or for n5/zarr2 (which don't carry it in
    this codebase)."""
    fmt = cfg["input_format"]
    spec = _SPEC[fmt]
    if spec["driver"] == "zarr3":
        resolved = _resolve_zarr3(cfg["input_intensity_path"], setup, scale)
        if resolved is not None:
            return resolved
    return f'{_path(fmt, cfg["input_intensity_path"], setup, "")}{scale}', spec["order"]


def canonical_shape(shape, order):
    """Stored shape -> (Z, Y, X)."""
    if order == "tczyx":
        return tuple(shape[-3:])
    if order == "zyx":
        return tuple(shape)
    return (shape[2], shape[1], shape[0])   # xyz -> zyx


def _in_order(zyx, order):
    """A canonical (Z, Y, X) triple expressed in the stored output order (5-D or
    3-D). Config [X, Y, Z] blocks/factors pass their reverse, `xyz[::-1]`."""
    z, y, x = zyx
    if order == "tczyx":
        return [1, 1, z, y, x]
    if order == "zyx":
        return [z, y, x]
    return [x, y, z]


def canonical_view(arr, order):
    """A TensorStore view of `arr` whose own index order is canonical (Z, Y, X),
    so it can be indexed `[z0:z1, y0:y1, x0:x1]` and `.read(order="C")` hands back
    a C-contiguous canonical array -- no numpy transpose, and no `to_canonical`.

    This matters for more than tidiness. `np.array(view.transpose(...))` defaults
    to `order="K"`, which PRESERVES the transposed layout: for an xyz-stored
    source (n5, zarr3_raw) that yields an F-contiguous "canonical" array, on which
    every elementwise pass over a (Y, X) coefficient plane is strided -- measured
    ~7x slower in `_correct_shard` than the same data C-ordered. Transposing it in
    numpy instead costs a slow single-threaded strided copy. Handing the transpose
    to TensorStore is much cheaper than either: it fuses into chunk decoding
    (no extra pass) and runs across `data_copy_concurrency`.

    Symmetrically for writes: `canonical_view(out, ...)[z, y, x].write(canon)`
    lets TensorStore absorb the transpose into encoding, instead of handing it a
    strided `from_canonical` view.
    """
    if order == "tczyx":
        return arr[0, 0]        # drop the T=1, C=1 singletons
    if order == "zyx":
        return arr
    return arr.T                # xyz -> zyx


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
            "context": context,
        }, open=True, read=True).result()

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
        "context": context,
    }, open=True, read=True).result()


def _compute_stats(vol):
    """Otsu background/foreground split + whole-volume stats for one tile's voxels."""
    # Whole-tile (unmasked) stats -- used by the apply stage's "uniform" branch
    # for tiles with no clean background/foreground split (see MIN_UNIFORM_STD).
    vol64 = vol.astype("float64")
    all_mean = float(vol64.mean())
    all_std = float(vol64.std())

    thr = float(threshold_otsu(vol))
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

    stats = _compute_stats(vol)
    out = _write_stats(cfg, setup, stats)
    print(dt.now(), f"stats: setup {setup} mean={stats['mean']:.2f} std={stats['std']:.2f} "
                    f"all_mean={stats['all_mean']:.2f} all_std={stats['all_std']:.2f} "
                    f"thr={stats['threshold']:.1f} n_fg={stats['n_foreground']} "
                    f"sep={stats['separation']:.1f} -> {out}", flush=True)


# ─── apply stage ─────────────────────────────────────────────────────────────────


def target_path(cfg):
    return Path(cfg["results_root"]) / "intensity_target.json"


def _enough_foreground(s):
    """Whole-tile emptiness test: is the foreground at least MIN_FG_FRACTION of the
    tile's voxels? Uses `n_voxels` when present (written by `_compute_stats`, or
    backfilled by `cmd_aggregate` from the tile shape); falls back to the legacy
    absolute MIN_FOREGROUND count only for old stats files lacking `n_voxels`."""
    nv = s.get("n_voxels")
    if nv:
        return s["n_foreground"] >= MIN_FG_FRACTION * nv
    return s["n_foreground"] >= MIN_FOREGROUND


def _classify(s):
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
    empty_area = s.get("empty_area")
    if empty_area is None or not np.isfinite(empty_area):
        raise RuntimeError(
            f"tile {s.get('setup')} has no empty_area; run the emptiness stage "
            f"(intensity_correction.py emptiness) before aggregate -- classification "
            f"depends on it")
    if not _enough_foreground(s) or not (
            np.isfinite(s["mean"]) and np.isfinite(s["std"]) and s["std"] > 0):
        return "empty"
    all_std = s.get("all_std", float("nan"))
    if not (np.isfinite(all_std) and all_std >= MIN_UNIFORM_STD):
        return "empty"
    return "bimodal" if empty_area >= MIN_BACKGROUND_AREA else "uniform"


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
        "context": _context(),
    }, open=True, read=True).result()
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
    def _sample(s):
        return _read_tile(cfg, s, level)[::4].ravel()[::7]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        samples = list(pool.map(_sample, setups))
    threshold = float(threshold_otsu(np.concatenate(samples)))
    del samples

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
        is_empty = occupancy < EMPTY_OCCUPANCY_FLOOR
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
    return empty, threshold, level, _background_level(acc["bg_sum"], acc["bg_cnt"]), phi


def _background_level(bg_sum, bg_cnt):
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
    return float(np.percentile(vals, BACKGROUND_PERCENTILE))


def _write_empty_fraction(path, phi):
    """Write the per-frame-pixel empty fraction as a float32 TIFF for `save_qstack`.

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
        import tifffile
    except ImportError:
        print(f"  (no {path.name}: tifffile not installed; qstack un-mixing unavailable)")
        return False
    tifffile.imwrite(str(path), phi.astype("float32"))
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
      * `basic_empty_fraction_camera{N}.tif` -- the per-frame-pixel empty fraction, used
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
        phi_path = Path(cfg["results_root"]) / f"basic_empty_fraction_camera{cam}.tif"
        phi_path.parent.mkdir(parents=True, exist_ok=True)
        wrote_phi = _write_empty_fraction(phi_path, phi)
        print(f"  thr {threshold:.0f} (level {level}, {_emptiness_workers(len(setups))} "
              f"threads) | merged into {len(setups)} tile stats"
              + (f" | phi map max {phi.max():.3f} -> {phi_path.name}" if wrote_phi else ""))
        if background_level is None:
            print("  WARNING no tile has an empty region here, so the additive offset "
                  "cannot be observed. basic_unmix_empty and override_darkfield both "
                  "need it.")


def cmd_aggregate(cfg):
    """Reduce step: solve a per-tile gain from all tile overlaps, then write the
    global target (see the module docstring's `aggregate` section for the why).

    1. Discover every physically-overlapping tile pair from the SpimData2
       `dataset.xml` (`_all_overlap_pairs` -- within-camera and cross-camera).
    2. For each pair, a robust log-gain constraint from the two tiles' own
       foreground medians in the shared region (`_pair_gain_constraint`;
       registration-tolerant, using each tile's Otsu threshold from `stats`).
       Run across a thread pool -- the tensorstore reads release the GIL, so
       this uses the cores the bsub job requests.
    3. Solve a gain g_s by regularized global least squares (`_solve_tile_gains`),
       grouped per `gain_grouping`: "camera" (default) shares one gain across a
       camera's tiles (robust, corrects the sensor-level step), "tile" solves one
       per tile. `gain_estimator` ("intersection" default) and `gain_lambda`
       (tiny for camera, ~0.1 for tile) tune the constraint and regularization.
    4. Encode g_s for the (unchanged) `apply` stage: with global equalized
       centre/spread M, S (median over usable tiles of own_mean/g, own_std/g)
       and target_mean=M, target_std=S, storing corrected_mean=g_s*M,
       corrected_std=g_s*S makes apply's `(raw-mean_i)*(target_std/std_i)+
       target_mean` compute exactly `raw/g_s` -- pure per-tile gain, texture
       preserved. Writes {results_root}/intensity_target.json (every tile's
       stats + corrected_mean/std, so apply reads one file) and
       {results_root}/tile_gains.json (per-tile gains + per-camera summary).
    """
    setups_all = tile_list(cfg)
    _, order = _input_location(cfg, setups_all[0], cfg["stats_scale"])
    n_cores = int(os.getenv("LSB_DJOB_NUMPROC", "20"))
    # gain grouping: "camera" (default) shares one gain across a camera's tiles --
    # robust (many constraints per node) and matches the sensor-level step we correct;
    # "tile" is the old per-tile mode (opt in). Regularization is just `gain_lambda`;
    # the gauge is its lam->0 limit, so per-camera defaults to a tiny lam and per-tile
    # to 0.1 (needs damping). Estimator "intersection" (default) matches voxels.
    grouping = cfg.get("gain_grouping", "camera")
    if grouping not in ("camera", "tile"):
        raise ValueError(f"gain_grouping={grouping!r} must be 'camera' or 'tile'")
    estimator = cfg.get("gain_estimator", "intersection")
    if estimator not in ("intersection", "independent"):
        raise ValueError(f"gain_estimator={estimator!r} must be 'intersection' or 'independent'")
    lam = float(cfg.get("gain_lambda", 1e-6 if grouping == "camera" else 0.1))

    # per-tile stats (Otsu threshold, classification, own mean/std) for every setup
    stat_cache, missing = {}, []
    for s in setups_all:
        p = stats_path(cfg, s)
        if p.exists():
            stat_cache[s] = json.loads(p.read_text())
            _check_basic_mode(cfg, stat_cache[s].get("apply_basic"), str(p))
        else:
            missing.append(s)

    # `empty_area` arrives in the per-tile stats files themselves, written by the
    # `emptiness` stage -- no separate read. It cannot be computed by the per-setup
    # `stats` jobs, which never see another tile: it needs one intensity threshold
    # pooled dataset-wide.
    missing_ea = sorted(s for s, st in stat_cache.items() if st.get("empty_area") is None)
    if missing_ea:
        raise RuntimeError(
            f"no empty_area in the stats for setup(s) {missing_ea}; run the emptiness "
            f"stage (intensity_correction.py emptiness) -- `_classify` needs it to tell "
            f"a tile with real background from one that is signal everywhere")
    setups = [s for s in setups_all if s in stat_cache]
    if not setups:
        raise RuntimeError("no per-setup stats found; run the stats stage first")

    # overlap graph over all tiles that have stats
    pairs, sizes, transforms = _all_overlap_pairs(cfg)
    pairs = [(a, b, bb) for (a, b, bb) in pairs if a in stat_cache and b in stat_cache]

    # precompute per-tile array handle / downsample factor / shape / threshold once
    # (parallel; opens are I/O-bound) so the per-pair ratio does no re-opening
    def _prep(s):
        arr = canonical_view(open_downsampled(cfg, s), order)
        shape = tuple(arr.domain.shape)
        return s, {"arr": arr,
                   "factor": source_pyramid_factors(cfg, s)[cfg["stats_scale"]],
                   "shape": shape,
                   "basic": basic_model(cfg, s, shape[1:]),
                   "thr": stat_cache[s]["threshold"]}
    with ThreadPoolExecutor(max_workers=n_cores) as pool:
        cache = dict(pool.map(_prep, setups))

    # backfill n_voxels (tile total) from the on-disk shape for any stats file that
    # predates it, so the MIN_FG_FRACTION emptiness gate works without a stats re-run
    for s in setups:
        stat_cache[s].setdefault("n_voxels", int(np.prod(cache[s]["shape"])))

    # drop empty tiles (MIN_FG_FRACTION) from the solve entirely: they carry no real
    # signal, so they neither constrain a neighbour's gain nor get a gain of their
    # own -- they pass through uncorrected. Only pairs between two non-empty tiles
    # contribute constraints.
    nonempty = {s for s in setups if _classify(stat_cache[s]) != "empty"}
    pairs = [(a, b, bb) for (a, b, bb) in pairs if a in nonempty and b in nonempty]
    print(dt.now(), f"aggregate: {len(setups)} tiles ({len(nonempty)} non-empty), "
                    f"{len(pairs)} overlapping pairs, {n_cores} threads, "
                    f"apply_basic={cfg['apply_basic']}, grouping={grouping}, "
                    f"estimator={estimator}, lambda={lam:g}", flush=True)

    # robust log-gain constraint per pair (parallel)
    with ThreadPoolExecutor(max_workers=n_cores) as pool:
        raw = list(pool.map(
            lambda p: _pair_gain_constraint(p[0], p[1], p[2], sizes, transforms, cache,
                                            order, estimator),
            pairs))
    constraints = [c for c in raw if c is not None]
    print(dt.now(), f"aggregate: {len(constraints)}/{len(pairs)} usable overlap constraints",
          flush=True)

    # per-camera groups every tile in a camera onto one gain; per-tile leaves each free
    groups = camera_groups(cfg)
    cam_of = {s: cam for cam, group in enumerate(groups) for s in group}
    group_of = cam_of if grouping == "camera" else None
    gains = _solve_tile_gains(constraints, setups, lam, group_of)

    # global equalized centre/spread from usable tiles' own stats / gain
    eq_means, eq_stds = [], []
    for s in setups:
        st = stat_cache[s]
        kind = _classify(st)
        if kind == "bimodal":
            m, sd = st["mean"], st["std"]
        elif kind == "uniform":
            m, sd = st["all_mean"], st["all_std"]
        else:
            continue
        eq_means.append(m / gains[s]); eq_stds.append(sd / gains[s])
    if not eq_means:
        raise RuntimeError("no usable (bimodal/uniform) tiles; run the stats stage first")
    M = float(np.median(eq_means))
    S = float(np.median(eq_stds))
    target_mean, target_std = M, S   # pure per-tile gain equalization (apply -> raw/g)

    setups_out = {}
    for s in setups:
        st = dict(stat_cache[s])
        st["camera"] = cam_of.get(s)
        st["gain"] = gains[s]
        if _classify(st) != "empty":
            st["corrected_mean"] = gains[s] * M
            st["corrected_std"] = gains[s] * S
        setups_out[str(s)] = st

    # per-camera gain summary (diagnostic only; correction itself is per-tile)
    cam_summary = {}
    for cam, group in enumerate(groups):
        gs = [gains[s] for s in group if s in gains]
        if gs:
            cam_summary[str(cam)] = {"median_gain": float(np.median(gs)),
                                     "min_gain": float(min(gs)), "max_gain": float(max(gs)),
                                     "n_tiles": len(gs)}
    _atomic_write_json(Path(cfg["results_root"]) / "tile_gains.json", {
        "grouping": grouping,
        "estimator": estimator,
        "lambda": lam,
        "n_pairs": len(pairs),
        "n_constraints": len(constraints),
        "global_equalized_mean": M,
        "global_equalized_std": S,
        "tile_gains": {str(s): gains[s] for s in setups},
        "camera_summary": cam_summary,
    })

    combined = {
        "target_mean": target_mean,
        "target_std": target_std,
        "apply_basic": cfg["apply_basic"],
        "min_foreground": MIN_FOREGROUND,
        "min_fg_fraction": MIN_FG_FRACTION,
        "n_total": len(setups_all),
        "n_present": len(setups_out),
        "n_pairs": len(pairs),
        "n_constraints": len(constraints),
        "setups": setups_out,
    }
    _atomic_write_json(target_path(cfg), combined)
    if missing:
        print(dt.now(), f"aggregate: no stats for {len(missing)} setups (skipped): "
                        f"{missing[:20]}{' ...' if len(missing) > 20 else ''}", flush=True)
    gmin, gmax = min(gains.values()), max(gains.values())
    n_groups = len(set(gains.values())) if grouping == "camera" else len(gains)
    print(dt.now(), f"aggregate: solved {len(constraints)} constraints -> {grouping} gains "
                    f"({n_groups} distinct) [{gmin:.3f}, {gmax:.3f}]; "
                    f"target_mean={target_mean:.2f} target_std={target_std:.2f} "
                    f"-> {target_path(cfg)}", flush=True)


def _atomic_write_json(path, obj):
    """Write JSON atomically (temp + os.replace) so concurrent array jobs writing
    the shared top-level group file can't corrupt it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, p)


def write_group_metadata(cfg, setup, factors):
    """Write multiscale group metadata for the output dataset/setup.

    zarr3 / zarr3_unsharded / zarr3_zyx -> OME-NGFF 0.5 (multiscales under
    attributes.ome in the group zarr.json); zarr2 -> OME-NGFF 0.4 (multiscales in
    .zattrs); n5 -> n5 `downsamplingFactors` group attributes. `factors` is the
    per-level list of cumulative (fz, fy, fx) downsample factors (level 0 ==
    (1, 1, 1)). zarr3_zyx writes 3-D (z,y,x) axes/scale/translation arrays - the
    standard NGFF spatial order - instead of the 5-D (t,c,z,y,x) the other zarr
    formats use, so downstream readers that assume that standard order (rather
    than reading axes by name) work without any changes.
    """
    fmt = cfg["output_format"]
    order = _SPEC[fmt]["order"]
    axes = _AXES_ZYX if order == "zyx" else _AXES
    datasets = _ngff_datasets(factors, order)

    if fmt in ("zarr3", "zarr3_unsharded", "zarr3_zyx"):
        ds_dir = cfg["output_intensity_path"]
        setup_dir = f"{ds_dir}/s{setup}-t0.zarr"
        ome = {
            "version": "0.5",
            "multiscales": [{
                "name": "/",
                "axes": axes,
                "datasets": datasets,
                "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(axes)}],
            }],
        }
        _atomic_write_json(f"{ds_dir}/zarr.json", {"zarr_format": 3, "node_type": "group"})
        _atomic_write_json(f"{setup_dir}/zarr.json",
                           {"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}})
    elif fmt == "zarr2":
        ds_dir = cfg["output_intensity_path"]
        setup_dir = f"{ds_dir}/s{setup}-t0.zarr"
        multiscales = [{
            "version": "0.4",
            "name": "/",
            "axes": axes,
            "datasets": datasets,
            "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(axes)}],
        }]
        _atomic_write_json(f"{ds_dir}/.zgroup", {"zarr_format": 2})
        _atomic_write_json(f"{setup_dir}/.zgroup", {"zarr_format": 2})
        _atomic_write_json(f"{setup_dir}/.zattrs", {"multiscales": multiscales})
    else:  # n5: group-level downsamplingFactors (x, y, z), the n5 multiscale convention
        ds_dir = cfg["output_intensity_path"]
        grp = f"{ds_dir}/setup{setup}/timepoint0"
        xyz = [[float(fx), float(fy), float(fz)] for (fz, fy, fx) in factors]
        _atomic_write_json(f"{ds_dir}/attributes.json", {"n5": "2.0.0"})
        _atomic_write_json(f"{grp}/attributes.json",
                           {"downsamplingFactors": xyz, "scales": xyz})


def _output_metadata(fmt, shape, chunk, shard, dtype_name):
    """(driver, metadata) for one output array in the given format."""
    if fmt in ("zarr3", "zarr3_zyx"):   # same sharded array layout, different axis order
        return "zarr3", {
            "data_type": dtype_name,
            "shape": shape,
            "fill_value": 0,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shard}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": chunk,
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "zstd", "configuration": {"level": 3}},
                    ],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }],
        }
    if fmt == "zarr3_unsharded":
        return "zarr3", {
            "data_type": dtype_name,
            "shape": shape,
            "fill_value": 0,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 3}},
            ],
        }
    if fmt == "zarr2":
        return "zarr2", {
            "dtype": ">u2",
            "shape": shape,
            "chunks": chunk,
            "dimension_separator": "/",
            "compressor": {"id": "zstd", "level": 3},
        }
    return "n5", {                       # n5
        "dimensions": shape,
        "blockSize": chunk,
        "dataType": dtype_name,
        "compression": {"type": "gzip"},
    }


def open_output_array(cfg, setup, level, shape, dtype_name, context):
    """Create one output level array (shape given in the output's stored order)."""
    fmt = cfg["output_format"]
    order = _SPEC[fmt]["order"]
    path = _output_path(fmt, cfg["output_intensity_path"], setup, level)
    chunk = _in_order(cfg["chunk_size"][::-1], order)   # inner chunk
    shard = _in_order(cfg["shard_size"][::-1], order)   # outer chunk / file
    driver, meta = _output_metadata(fmt, list(shape), chunk, shard, dtype_name)
    arr = ts.open({
        "driver": driver,
        "kvstore": {"driver": "file", "path": path},
        "metadata": meta,
        "context": context,
    }, create=True, open=True).result()
    return arr, shard, path


_SHAPE_CACHE = {}


def source_pyramid_shapes(cfg, setup):
    """Canonical (Z, Y, X) shape of every input pyramid level present on disk.

    Cached per (store, format, setup): walking the pyramid means one array open
    per level, and both `source_pyramid_factors` and `basic_model` need it (the
    latter once per setup per level, which would otherwise re-open the whole
    pyramid for every shard).
    """
    key = (cfg["input_intensity_path"], cfg["input_format"], setup)
    hit = _SHAPE_CACHE.get(key)
    if hit is not None:
        return hit

    spec = _SPEC[cfg["input_format"]]
    context = _context()
    shapes = []
    level = 0
    while True:
        path, order = _input_location(cfg, setup, level)
        if not (Path(path) / spec["meta"]).exists():
            break
        arr = ts.open({
            "driver": spec["driver"],
            "kvstore": {"driver": "file", "path": path},
            "context": context,
        }, open=True, read=True).result()
        shapes.append(canonical_shape(arr.domain.shape, order))   # (Z, Y, X)
        level += 1

    _SHAPE_CACHE[key] = shapes
    return shapes


def source_pyramid_factors(cfg, setup):
    """Cumulative (fz, fy, fx) downsample factors for each input pyramid level.

    Derived generically from the on-disk level shapes (factor = round(shape0 /
    shapeL) per axis), so it works for n5 / zarr2 / zarr3 alike. Level 0 is
    always (1, 1, 1). Falls back to a single level if no pyramid is present.
    """
    shapes = source_pyramid_shapes(cfg, setup)
    if not shapes:
        return [(1, 1, 1)]
    z0, y0, x0 = shapes[0]
    return [(max(round(z0 / z), 1), max(round(y0 / y), 1), max(round(x0 / x), 1))
            for (z, y, x) in shapes]


# One shard's correction runs on ONE core (it's handed to a thread in a pool that
# already has every other core busy on another shard), so the per-voxel work has
# to be as lean as possible. Three things get it there -- see `ShardCorrection`
# and `_correct_shard`:
#   1. Per-pixel affine coefficient PLANES precomputed once per setup, so the
#      inner loop is a multiply-add and the flat-field division never happens per
#      voxel (float32 division is several times the cost of a multiply).
#   2. `np.where` instead of boolean gather/scatter for the foreground-only
#      rescale. Boolean fancy indexing dominated this kernel; a vectorized select
#      over two precomputed buffers replaces it. It has to be a plain select --
#      `np.copyto(..., where=)` and `where=`-masked ufuncs both benchmarked
#      SLOWER than the gather they'd replace.
#   3. Working in blocks of ~`_BLOCK_VOXELS`, always over a CONTIGUOUS span of
#      the (Z, Y, X) array -- whole Z-planes when a plane is small enough,
#      contiguous row-blocks of one plane when it isn't (this data's shard plane
#      is 2560x4096, i.e. 40 MiB as float32). This keeps the several passes over
#      the working set cheap without ever materializing a shard-sized float
#      buffer, so scratch stays a couple of MiB per thread.
# Correcting both flat-field and intensity this way costs less CPU per voxel than
# the intensity rescale alone did before, on top of saving a whole read/write of
# the dataset.
#
# Two failure modes bound the block size from both sides, and they -- not the
# cache hierarchy -- are what the size is chosen against (benchmarked on this
# dataset's real shard geometry, where blocking by the 64^3 chunk shape and
# blocking by whole 40 MiB Z-planes both measured slower than the value below):
#   * TOO SMALL is much worse than too large, and only shows up under load: each
#     numpy call needs the GIL, so per-call overhead is a SERIALIZED resource
#     across the correction pool. Blocks of ~64 KiB cost ~13% on one thread but
#     lose a factor of 4 with the pool busy.
#   * NON-CONTIGUOUS is worse still: a 64^3 sub-block of a (Z, Y, X) shard is
#     thousands of short strided runs, which gives up SIMD and prefetching and
#     multiplies the call count. Tile rows, not cubes.
_BLOCK_VOXELS = int(os.getenv("BFF_BLOCK_KIB", "1024")) * 1024 // 4

_SCRATCH = threading.local()


def _scratch(shape):
    """Two per-thread float32 scratch buffers shaped like `shape`, reused across
    blocks and shards -- a pool thread corrects thousands of blocks, so reusing
    the pages beats re-faulting them every time. Kept as flat buffers and
    reshaped so a short trailing block borrows the same allocation and still
    gets a contiguous view."""
    n = int(np.prod(shape))
    bufs = getattr(_SCRATCH, "bufs", None)
    if bufs is None or bufs[0].size < n:
        bufs = (np.empty(n, "float32"), np.empty(n, "float32"))
        _SCRATCH.bufs = bufs
    return bufs[0][:n].reshape(shape), bufs[1][:n].reshape(shape)


def _blocks(zyx):
    """Contiguous (z_start, z_stop, y_start, y_stop) blocks of ~`_BLOCK_VOXELS`
    covering a canonical (Z, Y, X) array; x is always taken whole, so every
    block is a contiguous span of the underlying buffer (and the (Y, X)
    coefficient planes slice to it contiguously too).

    Whole Z-planes when a plane fits the budget, otherwise row-blocks of a single
    plane -- for a 2560x4096 plane that's 64 rows at a time rather than one
    40 MiB plane.
    """
    Z, Y, X = zyx
    plane = max(Y * X, 1)
    if plane <= _BLOCK_VOXELS:
        step = max(1, _BLOCK_VOXELS // plane)
        return [(z, min(z + step, Z), 0, Y) for z in range(0, Z, step)]
    dy = max(1, _BLOCK_VOXELS // max(X, 1))
    return [(z, z + 1, y, min(y + dy, Y))
            for z in range(Z) for y in range(0, Y, dy)]


def _win(plane, y_range, x_range):
    """A coefficient plane restricted to a shard's (y, x) window (scalars and
    None pass through)."""
    if plane is None or np.isscalar(plane):
        return plane
    return plane[y_range[0]:y_range[1], x_range[0]:x_range[1]]


class ShardCorrection:
    """Everything one setup's shards need, precomputed once per apply job.

    `mode` is the intensity strategy from `_classify` -- "bimodal" (rescale the
    Otsu foreground only), "uniform" (rescale every voxel), or "none" (an empty
    tile: no rescale, but still flat/dark-corrected when joint).

    With a `BasicModel`, the flat/dark correction `(raw - dark) / flat` is
    rewritten as the per-pixel affine `raw * basic_a + basic_b` with
    `basic_a = 1/flat`, `basic_b = -dark/flat`, computed once here. For "uniform"
    the intensity rescale folds into the SAME two planes:

        max(raw*a + b, 0) * S + K  ==  max(raw*(a*S) + (b*S + K), K)

    (valid because S > 0), so that whole branch is one multiply-add and a
    scalar maximum. `S = scale_i`, `K = target_mean - mean_i*scale_i`.

    Folding trades exactness for speed at the last bit: `raw*(1/flat)` and
    `(raw - dark)/flat` differ by an ULP or so, which flips a tiny fraction of
    voxels by one gray level -- less than the intermediate-uint16 rounding the
    joint route already removes, so it's only accepted where joint correction is
    active. WITHOUT a basic model this class keeps the original operation order
    (`(c - mean_i)*scale_i + target_mean`) exactly, so an intensity-only run
    still produces bit-identical output to previous releases -- it only gains
    `np.where` and the slab loop.
    """

    __slots__ = ("basic_a", "basic_b", "fg_a", "fg_b", "floor",
                 "thr", "mode", "mean", "scale", "target_mean", "S", "K")

    def __init__(self, basic, mode, thr, mean_i, scale_i, target_mean):
        self.mode, self.thr = mode, thr
        self.mean = np.float32(mean_i)
        self.scale = self.S = np.float32(scale_i)
        self.target_mean = np.float32(target_mean)
        self.K = np.float32(target_mean - mean_i * scale_i)
        if basic is None:
            self.basic_a = self.basic_b = self.fg_a = self.fg_b = self.floor = None
        else:
            a = (1.0 / basic.flat).astype("float32")
            b = (-basic.dark / basic.flat).astype("float32")
            self.basic_a, self.basic_b = a, b
            self.fg_a = (a * self.S).astype("float32")
            self.fg_b = (b * self.S + self.K).astype("float32")
            self.floor = self.K       # the max(basic, 0) clamp, folded (see above)


def _correct_shard(canon, corr, y_range, x_range):
    """Apply `corr` to one shard's canonical (Z, Y, X) block, in place.

    Pure numpy -- no I/O, no asyncio. Run via ThreadPoolExecutor (not a process
    pool): numpy's C loops release the GIL for arrays this size, so threads get
    real multi-core overlap without paying to pickle/copy a multi-hundred-MB
    shard across a process boundary.

    Every branch ends with the same tail -- clip to the uint16 range, round
    half-to-even (`np.rint`, matching Julia's `round(UInt16, x)`), cast back into
    `canon` -- so the data is quantized exactly ONCE no matter how many
    corrections were composed on the way.

    `canon` must be C-contiguous for `_blocks` to hand out contiguous work (which
    `canonical_view` guarantees for every caller here); anything else is corrected
    via a C-ordered copy rather than silently running strided, which measured ~7x
    slower on an xyz-stored source.
    """
    if not canon.flags["C_CONTIGUOUS"]:
        tmp = np.ascontiguousarray(canon)
        _correct_shard(tmp, corr, y_range, x_range)
        canon[...] = tmp
        return canon
    a_b_full = _win(corr.basic_a, y_range, x_range)
    b_b_full = _win(corr.basic_b, y_range, x_range)
    a_f_full = _win(corr.fg_a, y_range, x_range)
    b_f_full = _win(corr.fg_b, y_range, x_range)
    joint = a_b_full is not None
    rows = lambda p, y0, y1: p if p is None or np.isscalar(p) else p[y0:y1]

    for z0, z1, y0, y1 in _blocks(canon.shape):
        blk = canon[z0:z1, y0:y1]
        a_b, b_b = rows(a_b_full, y0, y1), rows(b_b_full, y0, y1)
        a_f, b_f = rows(a_f_full, y0, y1), rows(b_f_full, y0, y1)
        c, c2 = _scratch(blk.shape)

        if corr.mode == "uniform":
            if joint:
                np.multiply(blk, a_f, out=c)      # flat/dark AND rescale, folded
                c += b_f
                np.maximum(c, corr.floor, out=c)  # the max(basic, 0) clamp
            else:
                np.subtract(blk, corr.mean, out=c)
                c *= corr.scale
                c += corr.target_mean
        elif corr.mode == "bimodal":
            if joint:
                np.multiply(blk, a_b, out=c)      # c = (raw - dark)/flat
                c += b_b
                if corr.thr < 0:
                    # thr >= 0 (every real tile) makes the max(basic, 0) clamp a
                    # no-op here: it can't change which voxels clear the
                    # threshold, foreground values are already above it, and the
                    # final clip floors the background anyway. Only a negative
                    # threshold needs it done explicitly.
                    np.clip(c, 0, None, out=c)
                mask = c > corr.thr
                np.multiply(c, corr.S, out=c2)    # == (c - mean_i)*S + target_mean
                c2 += corr.K
            else:
                mask = blk > corr.thr             # compare on the source dtype
                np.subtract(blk, corr.mean, out=c2)
                c2 *= corr.scale
                c2 += corr.target_mean
                c = blk
            # Vectorized select, NOT c[mask] = ... : boolean gather/scatter
            # dominated this kernel (see the note above `_BLOCK_VOXELS`).
            c = np.where(mask, c2, c)
        else:                                     # "none": flat/dark only
            np.multiply(blk, a_b, out=c)
            c += b_b

        np.clip(c, 0, 65535, out=c)
        np.rint(c, out=c)
        blk[...] = c
    return canon


async def _apply(cfg, setup):
    context = _context()
    in_path, in_order = _input_location(cfg, setup, 0)
    out_order = _SPEC[cfg["output_format"]]["order"]

    tp = target_path(cfg)
    if not tp.exists():
        raise RuntimeError(f"target file {tp} not found; run the aggregate stage first")
    combined = json.loads(tp.read_text())
    _check_basic_mode(cfg, combined.get("apply_basic"), str(tp))
    target_mean, target_std = combined["target_mean"], combined["target_std"]

    s = combined["setups"].get(str(setup))
    if s is None:
        copy = True
        kind = "empty"
        mean_i = std_i = scale_i = 0.0
        thr_i = n_fg = 0
        print(dt.now(), f"apply: setup {setup} has no stats -> no intensity rescale", flush=True)
    else:
        kind = _classify(s)
        thr_i, n_fg = s["threshold"], s["n_foreground"]
        copy = kind == "empty"
        if kind in ("bimodal", "uniform"):
            # corrected_mean/std encode this tile's per-tile gain g (written by
            # `cmd_aggregate`: corrected_mean = g*M, corrected_std = g*S, with
            # target_mean=M, target_std=S). The formula below then reduces to
            # out = raw/g -- a pure per-tile multiplicative gain, solved from
            # overlap ratios. Texture survives (it's one gain per tile, not a
            # per-tile re-center to a common mean); only real sensor gain, which
            # the overlap solve isolated from content, is removed.
            mean_i, std_i = s["corrected_mean"], s["corrected_std"]
            if kind == "uniform":
                thr_i = float("-inf")  # no clean background to protect -- correct every pixel
        else:
            mean_i = std_i = 0.0
        scale_i = min(target_std / std_i, MAX_SCALE) if not copy else 1.0

    if copy and s is not None:
        print(dt.now(), f"apply: setup {setup} too empty -> no intensity rescale", flush=True)
    elif not copy:
        print(dt.now(), f"apply: setup {setup} [{kind}] mean={mean_i:.2f} std={std_i:.2f} "
                        f"-> target_mean={target_mean:.2f} target_std={target_std:.2f} "
                        f"scale={scale_i:.4f} thr={thr_i:.1f}", flush=True)

    src = ts.open({
        "driver": _SPEC[cfg["input_format"]]["driver"],
        "kvstore": {"driver": "file", "path": in_path},
        "context": context,
    }, create=False, open=True).result()

    # Canonical (Z, Y, X) views of both ends, so the shard loop never transposes
    # in numpy and `_correct_shard` always gets C-contiguous data (see
    # `canonical_view`).
    src_c = canonical_view(src, in_order)
    zyx = tuple(src_c.domain.shape)                     # full-res (Z, Y, X)
    dtype_name = str(src.dtype.name)

    # Per-camera BaSiC fields at full resolution (None unless `apply_basic`); each
    # shard takes its own (y, x) window out of these. Loaded once per job -- the
    # fields are one plane, so keeping them resident costs nothing next to a shard.
    basic = basic_model(cfg, setup, zyx[1:])
    if basic is not None:
        cam = camera_of(cfg, setup)
        print(dt.now(), f"apply: setup {setup} joint BaSiC correction from camera {cam + 1} "
                        f"fields {basic.flat.shape} (flat mean={float(basic.flat.mean()):.4f}, "
                        f"dark mean={float(basic.dark.mean()):.2f})", flush=True)

    # Coefficient planes for every shard of this setup, built once (see
    # `ShardCorrection`). None only when there is nothing to do at all -- no
    # flat/dark fields and an empty tile -- in which case shards are copied
    # straight through.
    # NB: not named `corr` -- `write_shard` binds that for its correct-phase
    # duration, and a nested assignment would make this local to it too.
    mode = "none" if copy else ("uniform" if thr_i == float("-inf") else "bimodal")
    shard_corr = (None if basic is None and mode == "none"
                  else ShardCorrection(basic, mode, thr_i, mean_i, scale_i, target_mean))
    out, shard, out_path = open_output_array(
        cfg, setup, 0, _in_order(zyx, out_order), dtype_name, context)
    out_c = canonical_view(out, out_order)
    print(dt.now(), f"apply: {cfg['input_format']} {src.domain.shape} -> "
                    f"{cfg['output_format']} {out.domain.shape} @ {out_path}", flush=True)

    # Stream the volume one output shard at a time, with several shards in flight.
    # A single whole-array `out.write(virt)` under one transaction serializes the
    # copy, so the many small n5 block reads never overlap (effective read depth ~1
    # -> cores idle, ~40 MiB/s). Instead, treat each shard as an independent work
    # unit (BigStitcher's model): read its region once, correct it, and write it
    # under its OWN transaction so the shard file is written exactly once (the
    # transaction batches the inner-chunk writes -> no shard rewrite amplification).
    # `asyncio.gather` + a semaphore then keeps N shards genuinely concurrent, so
    # their read waits overlap and the cores stay busy.
    sz, sy, sx = canonical_shape(shard, out_order)     # shard extent (Z, Y, X)
    Z, Y, X = zyx
    origins = [(oz, oy, ox)
               for oz in range(0, Z, sz)
               for oy in range(0, Y, sy)
               for ox in range(0, X, sx)]
    n_shards = len(origins)
    concurrency = int(os.getenv("LSB_DJOB_NUMPROC", "8")) // 1.5
    sem = asyncio.Semaphore(concurrency)

    # CPU-bound masking/rescale (_correct_shard) must not run as plain
    # synchronous code inside a coroutine -- that blocks the whole event loop,
    # serializing every shard's compute onto one core regardless of how many
    # were requested. Farming it out to a thread pool lets numpy's GIL-releasing
    # array ops actually overlap across cores while asyncio keeps handling I/O.
    n_cores = int(os.getenv("LSB_DJOB_NUMPROC", "8"))
    correct_pool = ThreadPoolExecutor(max_workers=n_cores)

    diag = {"done": 0, "bytes": 0, "t0": dt.now(), "last": dt.now(),
            "r": 0.0, "c": 0.0, "w": 0.0}   # cumulative read/correct/write seconds
    itemsize = np.dtype(str(src.dtype.name)).itemsize

    async def write_shard(oz, oy, ox):
        z = (oz, min(oz + sz, Z)); y = (oy, min(oy + sy, Y)); x = (ox, min(ox + sx, X))
        async with sem:
            # Per-phase timing. Phases overlap across concurrent shards, so the
            # cumulative r/c/w can exceed wall-clock -- read them as RELATIVE cost.
            # The per-shard r/c/w is wall-clock for that shard (includes FS waits
            # and contention from other in-flight shards), which is what we want.
            t0 = time.perf_counter()
            # Canonical (Z, Y, X) and C-contiguous straight out of TensorStore --
            # freshly allocated per read, so it is writable and needs no copy.
            canon = await src_c[z[0]:z[1], y[0]:y[1], x[0]:x[1]].read(order="C")
            t1 = time.perf_counter()
            if shard_corr is not None:
                canon = await asyncio.get_running_loop().run_in_executor(
                    correct_pool, _correct_shard, canon, shard_corr, y, x)
            t2 = time.perf_counter()
            txn = ts.Transaction(atomic=False)
            await out_c[z[0]:z[1], y[0]:y[1], x[0]:x[1]].with_transaction(txn).write(canon)
            await txn.commit_async()
            t3 = time.perf_counter()
        r, corr, w = t1 - t0, t2 - t1, t3 - t2
        diag["done"] += 1
        diag["bytes"] += canon.size * itemsize
        diag["r"] += r; diag["c"] += corr; diag["w"] += w
        gib = canon.size * itemsize / 2**30
        print(dt.now(), f"apply: setup {setup} shard ({oz},{oy},{ox}) "
                        f"{gib:.2f} GiB: read={r:.1f}s correct={corr:.1f}s write={w:.1f}s "
                        f"(read {gib/r if r else 0:.2f} GiB/s, write {gib/w if w else 0:.2f} GiB/s)",
              flush=True)
        now = dt.now()
        if (now - diag["last"]).total_seconds() >= 10.0:
            elapsed = (now - diag["t0"]).total_seconds() or 1e-9
            print(dt.now(), f"apply: setup {setup} progress: {diag['done']}/{n_shards} "
                            f"shards, {diag['bytes']/2**30:.2f} GiB, "
                            f"{diag['bytes']/2**20/elapsed:.1f} MiB/s | "
                            f"cumulative read={diag['r']:.0f}s correct={diag['c']:.0f}s "
                            f"write={diag['w']:.0f}s", flush=True)
            diag["last"] = now

    print(dt.now(), f"apply: writing setup {setup} level 0: {n_shards} shards "
                    f"(z,y,x)=({sz},{sy},{sx}), {concurrency} in flight, "
                    f"{n_cores} correction threads, "
                    f"{len(_blocks((sz, sy, sx)))} correction blocks/shard of "
                    f"{_BLOCK_VOXELS * 4 // 1024} KiB (override: BFF_BLOCK_KIB)", flush=True)
    try:
        await asyncio.gather(*(write_shard(*o) for o in origins))
    finally:
        correct_pool.shutdown(wait=True)

    # Build the matching multiscale pyramid by mean-downsampling the corrected
    # level 0 with the same cumulative factors the input dataset uses.
    factors = source_pyramid_factors(cfg, setup)
    level0 = ts.open({
        "driver": _SPEC[cfg["output_format"]]["driver"],
        "kvstore": {"driver": "file", "path": out_path},
        "context": context,
    }, open=True, read=True).result()
    for level in range(1, len(factors)):
        ds = ts.open({
            "driver": "downsample",
            "base": level0.spec(),
            "downsample_factors": _in_order(factors[level], out_order),
            "downsample_method": "mean",
            "context": context,
        }, open=True, read=True).result()
        lvl, _, _ = open_output_array(
            cfg, setup, level, list(ds.domain.shape), dtype_name, context)
        ltxn = ts.Transaction(atomic=False)
        print(dt.now(), f"apply: writing setup {setup} level {level} {tuple(ds.domain.shape)}", flush=True)
        await lvl.with_transaction(ltxn).write(ds)
        await ltxn.commit_async()

    write_group_metadata(cfg, setup, factors)
    print(dt.now(), f"apply: done setup {setup} ({len(factors)} levels)", flush=True)


def cmd_apply(cfg, setup):
    asyncio.run(_apply(cfg, setup))


# ─── cli ─────────────────────────────────────────────────────────────────────────


STAGES = ("stats", "emptiness", "aggregate", "apply")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: python intensity_correction.py {{{'|'.join(STAGES)}}} <setup-or-camera>")
    stage = sys.argv[1]
    # `emptiness` reads no pixel data beyond a coarse level, and writes no store, so
    # it does not need the intensity I/O paths -- see load_config's require_intensity_io.
    cfg = load_config(require_intensity_io=(stage != "emptiness"))

    if stage == "aggregate":      # single reduce job, no setup argument
        cmd_aggregate(cfg)
        return

    if stage == "emptiness":      # single job, no setup argument; see cmd_emptiness
        cmd_emptiness(cfg)
        return

    if len(sys.argv) >= 3:
        arg = int(sys.argv[2])
    else:
        arg = int(os.environ["LSB_JOBINDEX"]) - 1  # 0-indexed setup or camera

    if stage == "stats":
        cmd_stats(cfg, arg)
    else:
        cmd_apply(cfg, arg)


if __name__ == "__main__":
    main()
