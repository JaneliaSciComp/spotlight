"""The per-camera BaSiC input stack.

Normally the 21 quantile planes the stats pass wrote, assembled into a multi-page uint16
TIFF at `{qstacks_dir}/camera{N}.tiff`. For cameras too shallow in Z for quantiles, the
raw slices instead -- see `raw_stack_mode`.
"""

import json
from pathlib import Path

import numpy as np
import tifffile

from .fields import basic_model
from . import config as _config
from . import stores
from .orderstats import LEVELS, N_QUARTILES, to_uint16
from .quantiles import BACKGROUND_PIXEL_STRIDE, background_quantile_dir  # noqa: F401

__all__ = [
    "raw_stack_mode", "qstack_frame_size", "in_plane_order", "save_qstack",
    "save_qstack_camera", "load_qstack", "unmix_empty_fraction",
    "background_quantile_profile", "MAX_UNMIX_PHI",
]

# phi at or above this is capped: 1/(1 - phi) is 10x at 0.9 and unbounded at 1.
MAX_UNMIX_PHI = np.float32(0.9)

# Re-exported: it moved to `formats` so the writers of every plane -- the BaSiC fields and
# the empty-fraction map -- can reach the one rule without importing this module.
from .formats import in_plane_order, in_plane_swap  # noqa: E402


def qstack_frame_size(cfg, scale):
    """In-plane size of a quantile stack built at pyramid level `scale`, in its own order.

    `source_size_xyz` reports (X, Y, Z) for every format, but the qstack does not: the
    stats arrays are written (X, Y) and then transposed for everything but n5.
    """
    x, y, _ = stores.source_size_xyz(cfg, scale=scale)
    return (y, x) if in_plane_order(cfg) == "yx" else (x, y)


def raw_stack_mode(cfg, camera, scale=0):
    """Whether this camera is too shallow in Z for per-pixel quantiles.

    Per-pixel quantiles are built from a pixel's Z-column WITHIN ONE SETUP -- merging
    `OrderStats` across setups averages their sorted buffers, it does not pool samples --
    so a dataset with fewer than 21 Z-slices per setup has no distribution to summarise.
    For those (e.g. single-plane 2-D acquisitions) the quantile summary is skipped
    entirely and BaSiC is handed every sample plane instead. No stats pass runs at all.
    """
    setups = _config.camera_setups(cfg)[camera]
    return stores.camera_source_size_xyz(cfg, setups, scale=scale)[2] < N_QUARTILES


# ─── reading ──────────────────────────────────────────────────────────────────


def read_quantile_stack(cfg, camera):
    """The 21 quantile planes as one (X, Y, 21) uint16 array, in the stats arrays' order."""
    lvl = cfg["basic_stats_level"]
    ctx = stores.context()
    planes = []
    for q in LEVELS:
        path = f'{cfg["results_root"]}/camera{camera + 1}/q{q:03d}/s{lvl}'
        arr = stores._open({"driver": "n5",
                            "kvstore": {"driver": "file", "path": path}}, ctx)
        planes.append(np.asarray(arr.read().result()))
    return np.stack(planes, axis=2)


def read_raw_stack(cfg, camera, scale=0):
    """Every one of a camera's slices as one (X, Y, Z * nsetups) array.

    The `raw_stack_mode` substitute for the quantile stack: with too few Z-slices to
    summarise, BaSiC is fed the data itself rather than statistics derived from it. The
    whole camera is read at once with no X/Y tiling -- affordable precisely because Z is
    under 21, which is what put us on this path.
    """
    setups = _config.camera_setups(cfg)[camera]
    x, y, z = stores.camera_source_size_xyz(cfg, setups, scale=scale)
    ctx = stores.context()
    planes = []
    for setup in setups:
        src = stores.open_source(cfg, setup, scale=scale, ctx=ctx)   # (Z, Y, X)
        block = np.asarray(src[:z, :y, :x].read().result())
        planes.append(np.ascontiguousarray(block.T))                 # -> (X, Y, Z)
    return np.concatenate(planes, axis=2)


def load_qstack(cfg, camera):
    """One camera's saved qstack as an (H, W, N) float32 array in RAW COUNTS.

    Counts, not a normalised [0, 1] range: the flatfield is a mean-1 multiplier and so
    scale-free either way, but the darkfield is an additive offset that has to come out
    in the same units as the voxels the apply stage subtracts it from.
    """
    path = Path(cfg["qstacks_dir"]) / f"camera{camera + 1}.tiff"
    if not path.is_file():
        raise FileNotFoundError(f"Qstack not found: {path}\nRun save_qstack() first.")
    pages = np.asarray(tifffile.imread(str(path)))
    if pages.ndim != 3:
        raise ValueError(f"Expected a 3-D qstack, got shape {pages.shape}")
    return np.ascontiguousarray(np.moveaxis(pages, 0, 2), dtype=np.float32)


# ─── un-mixing the partly-empty-tile bias ─────────────────────────────────────


def unmix_empty_fraction(stack, phi, background):
    """Remove the partly-empty-tile bias from a (Y, X, N) quantile stack, in place.

    The stats pass merges each setup's `OrderStats` by AVERAGING their sorted buffers
    rather than pooling samples, so a tile that is empty at a frame position contributes
    the background level at EVERY quantile. A frame pixel empty in a fraction `phi` of
    the tiles therefore reads

        q_observed[k] = phi * background[k] + (1 - phi) * q_true[k]

    which inverts exactly. The bias is multiplicative in appearance -- the deficit is
    `phi * (q_true - background)`, so it grows with quantile -- which is why BaSiC absorbs
    it into the flat field rather than the darkfield, and why neither dropping low
    quantiles nor supplying a ramped darkfield fixes it.

    `background` is PER QUANTILE, and that is not a refinement: a scalar is wrong. The
    same averaging that creates the bias means an empty column's contribution is an
    average of per-block ORDER STATISTICS, so it rises with quantile index (measured 183,
    201, 221 counts at q000/q050/q100 on one dataset -- a third of that dataset's entire
    signal range). Since `1/(1 - phi)` multiplies the error by 3-10x where phi is large, a
    scalar there does more harm than no correction at all.

    `1/(1 - phi)` diverges where a pixel is empty in nearly every tile, so `phi` is capped
    at `MAX_UNMIX_PHI`. Those pixels have almost no real signal to recover anyway; the cap
    leaves them merely under-corrected instead of exploded.
    """
    y, x, n = stack.shape
    if phi.shape != (y, x):
        raise ValueError(f"empty-fraction map is {phi.shape} but the qstack frame is {(y, x)}")
    bg = np.full(n, np.float32(background), dtype=np.float32) \
        if np.isscalar(background) else np.asarray(background, dtype=np.float32)
    if bg.size != n:
        raise ValueError(f"background profile has {bg.size} levels but the stack has {n} "
                         "planes; they must correspond plane-for-plane")
    p = np.minimum(phi, MAX_UNMIX_PHI).astype(np.float32)[:, :, None]
    np.clip((stack - p * bg) / (np.float32(1.0) - p), 0.0, 65535.0, out=stack)
    return stack


def empty_fraction_map(cfg, camera, frame_size):
    """The emptiness stage's empty-fraction map for this camera, resized to `frame_size`.

    Written at a coarse pyramid level, so it is upsampled here -- fine because it varies
    on the scale of tile overlap, not per pixel. Transposed if it arrives (X, Y): the map
    is measured on canonical (Z, Y, X) volumes while a non-zarr qstack stays (X, Y, N).
    """
    from .basic import imresize

    path = _config.empty_fraction_path(cfg, camera)
    if not path.is_file():
        # Pre-move location, so results directories written before the map moved into
        # camera{N}/ keep working rather than forcing a full re-measure.
        legacy = Path(cfg["results_root"]) / f"basic_empty_fraction_camera{camera + 1}.tif"
        if legacy.is_file():
            print(f"note: reading the empty-fraction map from its old location {legacy}; "
                  f"re-run the emptiness stage to move it to {path}")
            path = legacy
    if not path.is_file():
        print(f"warning: no empty-fraction map at {path}; cannot un-mix")
        return None
    try:
        phi = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    except (OSError, ValueError) as err:
        print(f"warning: could not read the empty-fraction map {path} ({err}); cannot un-mix")
        return None
    # On disk the map is canonical (Y, X), like every plane beside a camera; the qstack it
    # un-mixes keeps the source's (X, Y) for n5. Swap from the config rather than from the
    # shape, so this is right for a square frame too, and BEFORE resizing, or the resize
    # maps one aspect onto the other.
    phi = in_plane_swap(phi, in_plane_order(cfg))
    # A map from the Julia package, or from a run before these planes were unified, may be
    # (X, Y) already -- in which case the swap above put it the wrong way round. Aspect is
    # the only evidence available (phi is at a coarse level, so its shape cannot be matched
    # against the frame) and it settles every frame that is not square.
    if (phi.shape[0] > phi.shape[1]) != (frame_size[0] > frame_size[1]):
        print(f"note: {path} is not canonical (Y, X); transposing it. Re-run the emptiness "
              f"stage to rewrite it in the order every other plane uses.")
        phi = np.ascontiguousarray(phi.T)
    if phi.shape == tuple(frame_size):
        return phi
    return imresize(phi, tuple(frame_size))


def background_quantile_profile(cfg, camera, n):
    """The per-quantile background profile, summed over every stats job's partial.

    None when the stats pass did not measure one, or when no job found an empty column.
    """
    d = background_quantile_dir(cfg, camera)
    if not d.is_dir():
        return None
    total = np.zeros(n, dtype=np.float64)
    count = 0
    # `job*.json`, not `*.json`: `create_quartile_histograms` keeps its fingerprint
    # stamp in this directory, and it is not a partial.
    for f in sorted(d.glob("job*.json")):
        try:
            rec = json.loads(f.read_text())
            s = np.asarray(rec["sum"], dtype=np.float64)
        except (OSError, ValueError, KeyError) as err:
            print(f"warning: unreadable background-quantile partial {f} ({err}); ignoring it")
            continue
        if s.size != n:
            print(f"warning: ignoring background-quantile partial {f} with the wrong "
                  f"length (expected {n}, got {s.size})")
            continue
        total += s
        count += int(rec["count"])
    if count == 0:
        return None
    return (total / count).astype(np.float32)


# ─── writing ──────────────────────────────────────────────────────────────────


def save_qstack_camera(cfg, camera, lvl=None, raw_mode=None):
    """Write one camera's BaSiC input stack.

    `raw_mode` is a parameter so `save_qstack` can pass what it already computed:
    `raw_stack_mode` reads source metadata for every setup in the camera, which on a
    1600-tile mosaic is 1600 metadata reads for one boolean.
    """
    lvl = cfg["basic_stats_level"] if lvl is None else lvl
    if raw_mode is None:
        raw_mode = raw_stack_mode(cfg, camera, scale=lvl)
    stack = (read_raw_stack(cfg, camera, scale=lvl) if raw_mode
             else read_quantile_stack(cfg, camera))
    if in_plane_order(cfg) == "yx":
        stack = np.ascontiguousarray(stack.transpose(1, 0, 2))

    if cfg["basic_unmix_empty"]:
        counts = stack.astype(np.float32)
        # Every path here reports what it did. Falling through silently when the map is
        # unreadable is indistinguishable from "un-mixing is off" in the output -- the
        # qstack just comes out unchanged.
        if raw_mode:
            # The planes are Z-slices, not order statistics, so a single background level
            # is the right thing -- and the stats pass that would have measured a profile
            # never ran for this camera.
            from .basic import measured_background_level
            bg = measured_background_level(cfg, camera)
            if bg is None:
                raise RuntimeError(
                    f"basic_unmix_empty is on but no background level was measured for "
                    f"camera {camera + 1}. Run the emptiness stage so it merges "
                    "background_level into each tile's intensity_stats JSON, or set "
                    "basic_unmix_empty = false.")
        else:
            bg = background_quantile_profile(cfg, camera, counts.shape[2])
            if bg is None:
                raise RuntimeError(
                    f"basic_unmix_empty is on but no per-quantile background profile was "
                    f"measured for camera {camera + 1}. It is measured by the quantile "
                    "stats pass, which needs the emptiness stage's threshold to have been "
                    "written first -- so rerun create_quartile_histograms() and resubmit "
                    f"the stats jobs, or set basic_unmix_empty = false. Expected partials "
                    f"in {background_quantile_dir(cfg, camera)}.")
        phi = empty_fraction_map(cfg, camera, counts.shape[:2])
        if phi is None:
            raise RuntimeError(
                f"basic_unmix_empty is on but the empty-fraction map for camera "
                f"{camera + 1} could not be read from {cfg['results_root']}; rerun the "
                "emptiness stage, or set basic_unmix_empty = false.")
        band = max(1, counts.shape[0] // 8)
        before = float(counts[:band].mean())
        unmix_empty_fraction(counts, phi, bg)
        after = float(counts[:band].mean())
        stack = to_uint16(counts)
        # Report the profile's ENDS, not a summary: a background that barely rises across
        # the quantiles means the stack's blocks were shallow, and one that rises steeply
        # is where a scalar would have done the most damage.
        span = (bg, bg) if np.isscalar(bg) else (float(bg[0]), float(bg[-1]))
        print(f"un-mixed camera {camera + 1}: background {span[0]:.4g}..{span[1]:.4g}, "
              f"phi_max {float(phi.max()):.4g} (capped at {float(MAX_UNMIX_PHI)}), "
              f"top band {before:.6g} -> {after:.6g}")
    else:
        print(f"basic_unmix_empty is off: writing camera {camera + 1}'s stack unchanged")

    out_dir = Path(cfg["qstacks_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"camera{camera + 1}.tiff"
    # One page per plane, matching what TiffImages wrote from a (H, W, N) array.
    tifffile.imwrite(str(path), np.ascontiguousarray(np.moveaxis(stack, 2, 0)))
    print(f"wrote {path} {stack.shape}")


def save_qstack(cfg=None):
    cfg = _config.load_config() if cfg is None else cfg
    lvl = cfg["basic_stats_level"]
    raw = [raw_stack_mode(cfg, c, scale=lvl) for c in range(_config.num_cameras(cfg))]
    # Only the raw_stack_mode cameras need the emptiness stage run from here: they have no
    # stats pass at all, so this is their only chance at a map and background level. On the
    # normal path create_quartile_histograms already ran it -- its stats jobs need the
    # threshold -- and rerunning would rescan every tile to re-derive the same numbers.
    if any(raw):
        from .scripts import measure_emptiness
        measure_emptiness(cfg)
    for camera in range(_config.num_cameras(cfg)):
        save_qstack_camera(cfg, camera, lvl=lvl, raw_mode=raw[camera])
