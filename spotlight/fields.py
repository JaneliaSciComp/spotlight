"""The BaSiC flat/dark fields: reading them, orienting them, matching them to a level.

`run_basic()` writes one pair of float32 TIFFs per camera at the level-0 frame. Every
stage that corrects voxels needs them at whatever level IT reads, in canonical (Y, X), so
that resizing and that transpose live here rather than in each caller.

`basic_field_paths` and `camera_of` are NOT here but in `config.py`, because `load_config`
calls the first to auto-detect `apply_basic` -- having them here would make the two
modules import each other.
"""

import numpy as np

from .config import basic_field_paths, camera_of
from .formats import canonical_plane
from .stores import source_pyramid_shapes


# ─── joint BaSiC flat-field / dark-field correction ─────────────────────────────
#
# Applies `run_basic()`'s per-camera fields to the voxels every stage reads, so
# the flat/dark correction and the per-tile intensity correction happen in one
# read/write of the data instead of two (see the module docstring). Off unless
# `apply_basic` (auto-detected in `load_config`).




def _read_field_tiff(path):
    """One BaSiC field as a 2-D float32 array.

    `save_basic_field` writes a single-page Gray{Float32} TIFF, so any of these readers
    gives the same (rows, cols) array Julia's `load` returns.
    """
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


def _oriented_field(field, full_yx, path):
    """A field in canonical (Y, X), which is how `save_basic_field` writes it.

    Stated rather than inferred from the shape, so a square tile -- where both
    orientations "fit" -- still resolves. The transpose is accepted with a warning only
    when the shape rules canonical out: fields from the Julia package, or from a run
    before these planes were unified, are (X, Y) on an n5 dataset. Re-run `run_basic()` to
    silence it.
    """
    return canonical_plane(field, full_yx, "yx", what=str(path))


def _block_mean_2d(a, fy, fx, out_yx):
    """Mean-downsample a 2-D field by (fy, fx), cropped/edge-padded to `out_yx`.

    `np.add.reduceat` rather than a reshape-mean, so a ragged final block -- when the
    plane is not a multiple of the factor -- is averaged over its real extent instead of
    forcing an exact division. The crop/pad then absorbs the off-by-one between our
    ceil-division block count and whatever rounding the on-disk pyramid level used. Edge
    padding, since a flat field's border value is the best estimate for a border row that
    exists on one side only.
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

        The (Y, X) fields broadcast over the leading Z axis. `nonneg_offset = 0` plus the
        clamp at 0 matches `apply_correction_chunked`'s call in
        src/BigFlatFieldIlluminator.jl, so a joint run reproduces the two-pass result --
        up to the intermediate uint16 rounding the joint route skips.
        """
        c = np.asarray(canon).astype("float32")
        c -= self.dark
        c /= self.flat
        np.clip(c, 0, None, out=c)
        return c


_BASIC_CACHE = {}


def basic_model(cfg, setup, level_yx):
    """`BasicModel` for `setup` at a level whose in-plane shape is `level_yx`, or None
    when `apply_basic` is off.

    The (fy, fx) downsample factor is derived from the level's own shape against the
    full-res shape, so this is right both for a prebuilt pyramid level and for
    `open_downsampled`'s on-the-fly downsample driver, whose in-plane factor is
    2**stats_scale but which leaves Z alone. Cached per (camera, level shape): the fields
    are per-camera, so every shard of an array job shares one.
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
    def prep(p):
        return _block_mean_2d(_oriented_field(_read_field_tiff(p), full_yx, p),
                              fy, fx, level_yx)

    model = BasicModel(prep(flat_p), prep(dark_p))
    _BASIC_CACHE[key] = model
    return model


def _check_basic_mode(cfg, recorded, what):
    """Refuse to combine stats/target files with voxels corrected differently.

    Per-tile stats (threshold, foreground mean/std) and the solved gains are only
    meaningful for the pixel values they were measured on, so a target built from raw
    stats must not drive a joint `apply`, or the reverse. Files written before
    `apply_basic` existed carry no flag; they were raw-derived, so a missing flag reads as
    False.
    """
    if bool(recorded) != cfg["apply_basic"]:
        raise RuntimeError(
            f"{what} was written with apply_basic={bool(recorded)} but this run has "
            f"apply_basic={cfg['apply_basic']}; re-run the earlier stage(s) so the "
            f"stats describe the same voxel values that will be written")
