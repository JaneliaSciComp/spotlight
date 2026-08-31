"""Per-experiment configuration.

Same file as before -- `LocalPreferences.toml` in the current working directory, so an
experiment is selected by `cd`-ing into its directory -- but the table is now
`[spotlight]` / `[spotlight.basic]`. Reads fall back to the pre-rename
`[BigFlatFieldIlluminator]` table so directories written by the Julia package keep
working; writes only ever produce `[spotlight]`.

`load_config` layers the BaSiC-side keys and their defaults over `_load_toml_config`,
which parses the shared table and validates the formats. Also owns the grouping helpers
(`camera_groups`, `camera_of`) and the result paths (`stats_path`, `target_path`,
`basic_field_paths`) -- the questions "which camera owns this setup" and "where do results
live", which every stage asks and none should answer for itself.
"""

import os
from pathlib import Path

import tomli_w
import tomllib

from .formats import FORMATS, OUTPUT_FORMATS

__all__ = [
    "load_config", "set_config", "set_basic_config", "basic_params",
    "camera_setups", "num_cameras", "basic_view", "config_path", "stage_cores",
]

TABLE = "spotlight"
LEGACY_TABLE = "BigFlatFieldIlluminator"

# Keys the BaSiC side owns, with the defaults from Julia's `load_config`. The keys
# `_load_toml_config` already handles (results_root, formats, stats_scale,
# last_setup, setup_ids, apply_basic, dataset_xml, the *_intensity_path pair) are
# deliberately absent -- one owner each.
DEFAULTS = {
    "input_basic_path": "",
    "output_basic_path": "",
    "qstacks_dir": "qstacks",
    "setups_per_row": 1,
    "setups_per_camera": 1,
    "chunk_size": [128, 128, 64],
    "shard_size": [512, 512, 256],
    "basic_unmix_empty": False,
    "basic_stats_level": 0,
    "output_stem": "",
    "error_stem": "",
    "lsf_project": "",
    # `bsub -W`: a wall-clock ceiling per element, in minutes. 0 disables it.
    #
    # This is a HOST watchdog, not a runtime estimate. Three `int-apply` elements of the
    # 560-element mouse_hipp_3_channel array (144/409/419) wedged for 12-14 h on a single
    # host, e10u18 -- see CLAUDE.md. Every other element on every other host finished in
    # 99-209 s. Set this well above the slowest healthy element and low enough that a
    # wedged mount costs minutes, not a night.
    "lsf_runlimit_minutes": 60,
    # Both the stage's `bsub -n` and its numpy ThreadPoolExecutor. Measured at 2.27
    # cores of actual CPU (477 s over a 210 s wall) and 5.8 GB peak RSS, so this is a
    # compute figure now, not the memory request it used to be at 48 -- Janelia hands
    # out 15 GB per slot, and read concurrency comes from `_concurrency`, not from here.
    "n_cores_stats": 3,
    "n_cores_correction": 20,
    "n_cores_int_stats": 1,
    "n_cores_int_aggregate": 5,
    "n_cores_int_correct": 20,
    "chunks_per_job": 64,
    "max_concurrent_cores": 2000,
    "z_batch": 1,
    # ─── the per-tile gain solve (aggregate) ──────────────────────────────────
    # Listed here so `load_config()` shows every knob the stage reads; `aggregate.py`
    # still owns what they mean. The two `min_overlap_*` values are the gate that
    # decides an overlap is trustworthy -- lower them for a sparse specimen whose
    # overlaps hold real but scant tissue, and see the shortfalls the stage prints.
    "gain_grouping": "camera",          # "camera" (one gain per camera) | "tile"
    "gain_estimator": "intersection",   # "intersection" (matched voxels) | "independent"
    "gain_lambda": None,                # None -> 1e-6 for camera, 0.1 for tile
    # What the gain solve gates its overlap medians at. Independent of
    # `tile_threshold` on purpose: the two answer different questions, so
    # `tile_threshold = 0` (rescale every pixel) can pair with `gain_floor = "li"`
    # (measure the gain on tissue only).
    #   "tile"   -- whatever `tile_threshold` selected (default)
    #   "otsu" / "li" -- that method's threshold, recorded per tile by the stats stage
    #   a number -- one floor for every pair
    "gain_floor": "tile",
    # THE background/foreground split, for every stage that needs one: the per-tile
    # stats, the emptiness stage's pooled threshold, the gain floor, the
    # empty/bimodal/uniform classification, and the apply stage's mask. A threshold set
    # too high is the usual cause of a seam that survives correction.
    #   "otsu"   -- Assumes two classes of comparable size, so it
    #               lands deep in the tail on a sparse specimen
    #   "li"     -- minimum cross-entropy; resists a heavy tail. See resolve_threshold.
    #   "pooled" -- the emptiness stage's dataset-wide value (which is itself otsu).
    #   a number -- one floor everywhere; also lets the emptiness stage skip its
    #               sampling pass entirely.
    "tile_threshold": "li",
    "min_overlap_foreground": 256,      # = tilestats.MIN_FOREGROUND
    "min_overlap_fraction": 0.001,      # = tilestats.MIN_FG_FRACTION

    # ─── tile classification (tilestats._classify) ────────────────────────────
    # Decides `empty` / `bimodal` / `uniform` per tile, which is not cosmetic: an
    # `empty` tile is dropped from the gain solve AND passed through uncorrected, and a
    # `uniform` tile is rescaled over EVERY pixel (thr = -inf, whole-tile mean/std)
    # rather than over a foreground mask.
    "min_tile_foreground": 256,         # absolute voxel floor (old stats files only)
    "min_tile_fraction": 0.001,         # foreground as a fraction of the tile -> `empty`
    "min_uniform_std": 10.0,            # whole-tile std below this -> `empty` (all noise)
    "min_background_area": 0.02,        # empty_area below this -> `uniform`, not `bimodal`
    "max_gain_scale": 8.0,              # clamp on the per-tile intensity rescale

    # ─── emptiness stage ──────────────────────────────────────────────────────
    "empty_occupancy_floor": 0.02,      # column occupancy below this counts as empty
    "otsu_sample_voxels": 32_000_000,   # pooled subsample size for the threshold
    "background_percentile": 5,         # percentile of empty-pixel means -> dark floor

    # ─── apply ────────────────────────────────────────────────────────────────
    # The kernel folds (raw-dark)/flat into raw*a + b, so flat == 0 gives inf + (-inf)
    # = NaN, which casts to an ARBITRARY uint16. Flooring makes it saturate instead.
    "flat_floor": 1e-3,

    # ─── OME-TIFF output (output_format = "tiff") ─────────────────────────────
    "tiff_compression": "zstd",         # or "deflate", "lzw", None for uncompressed
    "tiff_slab_planes": 64,             # z planes held in memory at once while streaming
}

BASIC_DEFAULTS = {
    "estimate_darkfield": True,
    "lambda": 0.0,
    "lambda_darkfield": 0.0,
    "autotune": True,
    "max_iterations": 500,
    "optimization_tol": 1e-6,
    "reweight_tol": 1e-3,
    "max_reweighting_iterations": 10,
    "epsilon": 0.1,
    "working_size": 0,
    "override_darkfield": False,
}

# Paths in the toml may be written with a literal `$HOME`. Together with the three keys
# `_load_toml_config` expands itself (the two `*_intensity_path`s and `results_root`) this
# is every path key, so `expand` below is the only place either fixup has to happen.
_PATH_KEYS = ("input_basic_path", "output_basic_path", "output_stem", "error_stem")


def _slashes(path):
    """A path with forward slashes, which Windows accepts everywhere and tensorstore wants.

    The kvstore paths this package hands tensorstore are built by `/`-joining a suffix
    onto a configured root, so on Windows a root of `C:\\data\\exp` yields the mixed
    `C:\\data\\exp/setup0/timepoint0/s0`. Python's own `open()` does not care; tensorstore's
    `file` driver has to parse the key. Normalising the ROOTS is the whole fix --
    everything derived from them is already `/`-joined.

    Not `Path.as_posix()`: that resolves nothing here and would mangle a UNC root, whose
    leading `\\\\server\\share` must stay a double separator. A plain replace keeps it as
    `//server/share`. No-op on Linux and macOS, where `\\` is a legal filename character.
    """
    return path.replace("\\", "/") if os.sep == "\\" else path


def config_path():
    return Path.cwd() / "LocalPreferences.toml"


def _read_tables():
    path = config_path()
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


# The `n_cores_*` key each stage is submitted with, so a thread pool can be sized from the
# reservation the stage would have had. One mapping, because otherwise every pool repeats
# the key and they drift: `correct` sized itself from 8 off-cluster for a stage that asks
# LSF for 20.
CORES_KEY = {
    "stats": "n_cores_stats",
    "basic": "n_cores_correction",        # the BaSiC-only correction script
    "intensity": "n_cores_int_correct",   # the intensity pipeline's apply stage
    "both": "n_cores_int_correct",
    "int-stats": "n_cores_int_stats",
    "aggregate": "n_cores_int_aggregate",
    # spotfix does the same kind of work as the apply stage -- read a shard, one numpy
    # multiply, write it -- so it sizes from the same reservation rather than inventing a
    # key nobody sets. It only ever runs locally, where `slots()` clamps to the machine.
    "spotfix": "n_cores_int_correct",
}


def stage_cores(cfg, stage):
    """The `bsub -n` this stage is submitted with, as a thread-pool size.

    Pass it to `stores.slots()`, which prefers LSF's actual reservation and falls back to
    this -- so an off-cluster run gets the pool the cluster would have given it rather
    than an unrelated literal.

    Reads DEFAULTS when the key is absent rather than raising: a partial config (a test
    fixture, a hand-built dict) should size a pool, not kill the stage.
    """
    key = CORES_KEY[stage]
    return int(cfg.get(key) or DEFAULTS[key])


def load_config(require_intensity_io=False):
    """The merged configuration for this experiment.

    Defaults to `require_intensity_io=False`, the opposite of `_load_toml_config`, because
    a BaSiC-only experiment has no reason to have configured the intensity pipeline's I/O.
    """
    cfg = _load_toml_config(require_intensity_io=require_intensity_io)
    for key, default in DEFAULTS.items():
        cfg.setdefault(key, default)
    for key in _PATH_KEYS:
        cfg[key] = _slashes(str(cfg[key]).replace("$HOME", os.path.expanduser("~")))
    if cfg["basic_stats_level"] < 0:
        raise ValueError(f"basic_stats_level must be >= 0, got {cfg['basic_stats_level']}")
    return cfg


def basic_params(cfg=None):
    """The `[spotlight.basic]` table, with defaults filled in.

    A BaSiC key written one level up -- in `[spotlight]` rather than `[spotlight.basic]`
    -- is honoured with a warning rather than ignored. Silently ignoring it is the worse
    failure: `override_darkfield = true` in the wrong table reads back as `false`, and the
    run merely looks like it chose not to override.
    """
    cfg = load_config() if cfg is None else cfg
    nested = dict(cfg.get("basic", {}))
    params = dict(BASIC_DEFAULTS)
    stray = sorted(k for k in BASIC_DEFAULTS if k in cfg and k not in nested)
    if stray:
        print(f"warning: BaSiC setting(s) {stray} found in [{TABLE}] rather than "
              f"[{TABLE}.basic] -- using them anyway. Move them, or set them with "
              "set_basic_config().")
        params.update({k: cfg[k] for k in stray})
    params.update(nested)
    return params


def camera_setups(cfg):
    """Setups grouped by camera, 0-based camera index."""
    return camera_groups(cfg)


def num_cameras(cfg):
    return len(camera_setups(cfg))


def basic_view(cfg):
    """`cfg` with the intensity pipeline's I/O keys pointed at the BaSiC paths.

    Every store helper (`_input_location`, `open_output_array`, `source_pyramid_shapes`,
    ...) reads `input_intensity_path` / `output_intensity_path`. Rebinding those two keys
    is the whole adapter; no second format abstraction gets written on this side.
    """
    return {**cfg,
            "input_intensity_path": cfg["input_basic_path"],
            "output_intensity_path": cfg["output_basic_path"]}


# ─── writers ──────────────────────────────────────────────────────────────────


def _write_tables(tables):
    with open(config_path(), "wb") as f:
        tomli_w.dump(tables, f)


def set_config(**kwargs):
    """Merge `kwargs` into the `[spotlight]` table of ./LocalPreferences.toml.

    A legacy `[BigFlatFieldIlluminator]` table is migrated on first write rather than left
    beside the new one: two tables holding the same keys is the state where an edit lands
    in the one nothing reads.
    """
    tables = _read_tables()
    section = tables.pop(LEGACY_TABLE, {}) | tables.get(TABLE, {})
    section.update(kwargs)
    tables[TABLE] = section
    _write_tables(tables)


def set_basic_config(**kwargs):
    """Merge `kwargs` into the `[spotlight.basic]` table."""
    tables = _read_tables()
    section = tables.pop(LEGACY_TABLE, {}) | tables.get(TABLE, {})
    section["basic"] = dict(section.get("basic", {})) | kwargs
    tables[TABLE] = section
    _write_tables(tables)


# ─── the shared table, grouping, and result paths ─────────────────────────────
#
# `load_config` and the grouping/path helpers, plus `basic_field_paths` and `camera_of`.
# Those last two are here rather than in `fields.py` because `load_config` calls
# `basic_field_paths` to auto-detect `apply_basic`: with them in `fields.py` the two
# modules would import each other. They are also the better fit -- they answer "where do
# results live" and "which camera owns this setup", the same question as `stats_path` and
# `target_path`, not "how do I read a field".


def _load_toml_config(require_intensity_io=True):
    """Read the `[spotlight]` table from LocalPreferences.toml in the current working
    directory, so an experiment is selected by `cd`-ing into it, wherever this package
    sits on disk.

    Falls back to the pre-rename `[BigFlatFieldIlluminator]` table so directories written
    by the Julia package keep working; `set_config` only ever writes `[spotlight]`.

    `require_intensity_io=False` tolerates a toml with no `input_intensity_path` /
    `output_intensity_path`, for stages that read neither. The `emptiness` stage is the
    case that matters: it is driven from the BaSiC side (`create_quartile_histograms`
    invokes it), and a BaSiC-only experiment has no reason to have configured the
    per-setup intensity pipeline's I/O at all. Demanding those keys there turned a working
    BaSiC run into a KeyError.
    """
    path = config_path()
    try:
        with open(path, "rb") as f:
            _tables = tomllib.load(f)
    except tomllib.TOMLDecodeError as err:
        # tomllib names neither the file nor the reason a first-line failure is usually
        # spurious, and the file it reads is the CWD's -- so the traceback alone cannot tell
        # you whether the toml is broken or you are simply in the wrong directory.
        raise ValueError(
            f"{path} is not valid TOML: {err}. That path is the current working "
            f"directory's -- each experiment's toml is picked up by running from that "
            f"experiment's directory -- so check this is the directory you meant. A "
            f"complaint at line 1 column 1 on a file that looks correct is usually a "
            f"UTF-8 byte-order mark an editor added ahead of the first table header."
        ) from err
    cfg = _tables.get("spotlight") or _tables["BigFlatFieldIlluminator"]

    def expand(s):
        return _slashes(s.replace("$HOME", os.path.expanduser("~")))

    for key in ("input_intensity_path", "output_intensity_path"):
        if key in cfg:
            cfg[key] = expand(cfg[key])
        elif require_intensity_io:
            raise KeyError(f"{key} is required in LocalPreferences.toml for this stage")
        else:
            cfg[key] = ""
    cfg["results_root"] = expand(cfg["results_root"])
    # `format` was the pre-rename name for `input_format`. Rejected rather than ignored:
    # silently defaulting to zarr2 would read an n5 store as the wrong driver.
    if "format" in cfg:
        raise ValueError("`format` is now `input_format` (and `output_format` for the "
                         "output); rename it in LocalPreferences.toml")
    cfg.setdefault("input_format", "zarr2")
    # `output_format` defaults to the input's: same format in and out unless asked.
    cfg.setdefault("output_format", cfg["input_format"])
    cfg.setdefault("stats_scale", 2)
    if cfg["input_format"] == "tiff":
        raise ValueError(
            "input_format = 'tiff', but tiff cannot be read; set output_format = 'tiff' "
            "and leave input_format as the input's")
    # Separate tuples: `tiff` can be written but not read (see formats.OUTPUT_FORMATS).
    for key, allowed in (("input_format", FORMATS), ("output_format", OUTPUT_FORMATS)):
        if cfg[key] not in allowed:
            raise ValueError(f"{key}={cfg[key]!r} must be one of {allowed}")

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
    """Setups grouped by camera: `setup_ids` as-is if given (one group per camera, e.g.
    `[[171,...,194], [201,...,204]]`), else contiguous `setups_per_camera`-sized chunks of
    `0..last_setup`.

    Mirrors `camera_setups()` in src/BigFlatFieldIlluminator.jl, so the two pipelines
    agree on which setups belong to which camera.
    """
    ids = cfg.get("setup_ids", [])
    if ids:
        return [list(group) for group in ids]
    per_cam = cfg.get("setups_per_camera", 1)
    last = cfg["last_setup"]
    return [list(range(start, min(start + per_cam, last + 1)))
            for start in range(0, last + 1, per_cam)]


def basic_field_paths(cfg, camera):
    """(flat, dark) TIFF paths for a 0-based camera index.

    `run_basic()` writes 1-BASED `camera{N}` directories (Julia's
    `camera_setups(config)[camera]` is indexed from 1) while `camera_groups` here is
    0-based -- hence the +1.
    """
    d = Path(cfg["results_root"]) / f"camera{camera + 1}"
    return d / "Flat-field.tif", d / "Dark-field.tif"


def camera_of(cfg, setup):
    """0-based camera index owning `setup`, from `camera_groups`."""
    for cam, group in enumerate(camera_groups(cfg)):
        if setup in group:
            return cam
    raise RuntimeError(f"setup {setup} belongs to no camera group; check setup_ids / "
                       f"setups_per_camera / last_setup")


def stats_path(cfg, setup):
    return Path(cfg["results_root"]) / "intensity_stats" / f"setup{setup}.json"


def target_path(cfg):
    return Path(cfg["results_root"]) / "intensity_target.json"


def empty_fraction_path(cfg, camera):
    """The per-frame-pixel empty-fraction map for a 0-based camera index.

    Inside `camera{N}/`, beside that camera's fields and statistic arrays, rather than
    loose in `results_root`: everything else per-camera already lives there, and a flat
    directory of `basic_empty_fraction_camera*.tif` does not scale past a few cameras.
    """
    return Path(cfg["results_root"]) / f"camera{camera + 1}" / "empty_fraction.tif"
