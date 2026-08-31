"""`spotfix` -- repair a LOCAL dimming defect in one already-corrected tile, in place.

A tile sometimes goes dark over part of its volume while the tiles that overlap it hold
real signal there. Flat/dark and the per-tile gain cannot fix that: both are corrections
this tile applies to itself, and the evidence that something is missing lives in its
NEIGHBOURS. So the expectation here comes from the overlapping tiles resampled into this
tile's grid, never from a model of the tile itself. That one choice is what makes the
stage safe on dark anatomy -- a genuinely dark structure is dark in the neighbours too, so
nothing is demanded of it.

SCOPE, and it is narrow: local dimming only. The input must already be flat-field and
intensity corrected, because a tile that is uniformly a few percent dim is a per-tile GAIN
error, not a defect, and this stage would smear it into a large spatially varying
correction. `_precondition` measures that and refuses rather than guessing.

Runs locally, one tile at a time:

    python -m spotlight run spotfix 126 158

Writes into the dataset it reads (`output_intensity_path`), so the previous version of the
tile is RENAMED aside first -- never deleted, never overwritten in place. A rename is
atomic on one filesystem, costs no copy, and leaves the old pyramid intact if the run dies
halfway.

The algorithm, and where each number came from, is in CLAUDE.md under "spotfix".
"""

import asyncio
import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import tensorstore as ts

from . import config as _config
from . import stores
from .formats import _in_order, _output_path, _SPEC, canonical_shape, canonical_view
from .progress import Progress

__all__ = ["fix_tile", "neighbours", "gain_field", "DEFAULTS"]

# Every length is in MICRONS and every threshold is dimensionless or in units of the
# background noise, so that they mean the same thing on a dataset sampled differently.
# Voxel counts and ratios-of-background do not transfer: a 4:1 z:lateral experiment and a
# 6.4:1 one need different bin factors for the same physical cell.
DEFAULTS = {
    "spotfix_level": 4,             # analysis level; the gain is estimated here
    "spotfix_cell_um": 20.1,        # lateral size of one gain cell
    "spotfix_smooth_z_um": 22.6,    # despeckle footprint, z
    "spotfix_smooth_lat_um": 60.3,  # despeckle footprint, lateral
    "spotfix_contrast_um": 120.6,   # "dark against bright surroundings?" window
    "spotfix_presence_um": 261.25,  # "do the neighbours show specimen here?" window
    "spotfix_edge_step": 0.0526315789,  # gain discontinuity allowed at the mask edge
    "spotfix_loc_t": 0.5,           # deadband on neighbour presence
    "spotfix_floor_t": 0.25,        # deadband on the tile's own signal
    "spotfix_floor_sigma": 4.0,     # signal-floor ramp top, in bg_std above bg
}


def params(cfg):
    """Config with the spotfix defaults filled in (a toml may override any of them)."""
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}


def edge_r(cfg):
    """The mask threshold. `spotfix_edge_step` is the gain discontinuity at the edge of
    the correction: the mask ends where r == edge_r, so the gain just inside is 1/edge_r
    and just outside is exactly 1. Detection sensitivity and edge visibility are therefore
    the SAME number, which is why tightening it produces a visible band rather than a
    cleaner detection."""
    return 1.0 / (1.0 + float(params(cfg)["spotfix_edge_step"]))


# ─── geometry: which tiles overlap this one, and how they line up ────────────────


def _xml(cfg):
    return ET.parse(cfg["dataset_xml"]).getroot()


def _sizes_and_transforms(root):
    """setup -> (size_xyz, composed 4x4 pixel->world affine).

    Same convention `aggregate` uses: <ViewTransform>s are listed outermost-first and
    compose as M0 @ M1 @ ... so the last (calibration) applies first to a pixel index.
    """
    sizes, xf = {}, {}
    for vs in root.findall(".//ViewSetups/ViewSetup"):
        sizes[int(vs.findtext("id"))] = tuple(int(v) for v in vs.findtext("size").split())
    for vr in root.findall(".//ViewRegistrations/ViewRegistration"):
        if vr.get("timepoint") != "0":
            continue
        total = np.eye(4)
        for vt in vr.findall("ViewTransform"):
            if vt.get("type") != "affine":
                continue
            m = np.eye(4)
            m[:3, :] = np.array([float(v) for v in vt.findtext("affine").split()]).reshape(3, 4)
            total = total @ m
        xf[int(vr.get("setup"))] = total
    return sizes, xf


def neighbours(cfg, setup, root=None):
    """Setups whose world-space bounding box intersects `setup`'s, from the GEOMETRY.

    Not a hand-written ring and not a camera-adjacency assumption: an edge tile simply has
    fewer, and the stage has to cope with that (tile 126 of the mouse experiment has no
    +x neighbour at all).
    """
    root = _xml(cfg) if root is None else root
    sizes, xf = _sizes_and_transforms(root)
    if setup not in sizes or setup not in xf:
        raise KeyError(f"setup {setup} has no ViewSetup/ViewRegistration in "
                       f"{cfg['dataset_xml']}")

    def bbox(s):
        sx, sy, sz = sizes[s]
        corners = np.array([[x, y, z, 1.0]
                            for x in (0, sx) for y in (0, sy) for z in (0, sz)])
        w = (xf[s] @ corners.T).T[:, :3]
        return w.min(axis=0), w.max(axis=0)

    lo0, hi0 = bbox(setup)
    out = []
    for s in sizes:
        if s == setup or s not in xf:
            continue
        lo, hi = bbox(s)
        if (lo < hi0).all() and (hi > lo0).all():
            out.append(s)
    return sorted(out)


def _origin_world(cfg, setup, root=None):
    root = _xml(cfg) if root is None else root
    _, xf = _sizes_and_transforms(root)
    return xf[setup][:3, 3].copy()


def voxel_world(cfg, setup, level, root=None):
    """World units spanned by ONE voxel at `level`, as (x, y, z).

    Derived, not assumed. On a 4:1 z:lateral experiment the z downsampling at level 4
    exactly cancels the calibration and all three axes come out 16; on a 6.4:1 one the z
    axis is 25.5 and a single scalar would misplace every neighbour in z.
    """
    root = _xml(cfg) if root is None else root
    _, xf = _sizes_and_transforms(root)
    cal = np.abs(np.diag(xf[setup][:3, :3]))     # calibration folded into the chain
    # The OUTPUT store's pyramid, not the input's: this stage reads and writes the
    # corrected dataset, and the two stores need not have the same number of levels.
    shapes = _levels_in(tile_dir(cfg, setup), cfg["output_format"])
    if not shapes:
        raise FileNotFoundError(f"no pyramid under {tile_dir(cfg, setup)}")
    z0, y0, x0 = shapes[0]
    zl, yl, xl = shapes[min(level, len(shapes) - 1)]
    fz, fy, fx = z0 / zl, y0 / yl, x0 / xl
    return np.array([fx * cal[0], fy * cal[1], fz * cal[2]])


# ─── the neighbour reference ─────────────────────────────────────────────────────


def _read_level(cfg, setup, level, ctx):
    """One setup's array at `level` from the store this stage reads AND writes.

    That store is `output_intensity_path` -- the corrected dataset -- because spotfix runs
    after correction and repairs its output. `_output_path` is the right resolver for it:
    the layout is always this codebase's own.
    """
    fmt = cfg["output_format"]
    spec = {"driver": _SPEC[fmt]["driver"],
            "kvstore": {"driver": "file",
                        "path": _output_path(fmt, cfg["output_intensity_path"], setup, level)}}
    arr = canonical_view(stores._open(spec, ctx, open=True, read=True), _SPEC[fmt]["order"])
    return np.asarray(arr[...].read().result(), dtype=np.float32)


def _place(vals, shift_zyx, shape):
    """`vals` shifted into a `shape`-sized grid aligned to the reference tile.

    The empty-overlap guard is the point: an earlier version let a negative stop wrap and
    silently dropped every non-overlapping tile, which looks like a coverage result rather
    than a bug.
    """
    out = np.zeros(shape, vals.dtype)
    sl = []
    for k in range(3):
        s = int(round(shift_zyx[k]))
        lo, hi = max(0, s), min(shape[k], vals.shape[k] + s)
        if hi <= lo:
            return out, 0
        sl.append((slice(lo, hi), slice(lo - s, hi - s)))
    r = tuple(a for a, _ in sl); o = tuple(b for _, b in sl)
    out[r] = vals[o]
    return out, int(np.prod([a.stop - a.start for a in r]))


def neighbour_reference(cfg, setup, nbrs, level, ctx=None):
    """(tile, neighbour_mean, covered) on the tile's own `level` grid."""
    root = _xml(cfg)
    a = _read_level(cfg, setup, level, ctx)
    vw = voxel_world(cfg, setup, level, root)          # (x, y, z) world units per voxel
    p0 = _origin_world(cfg, setup, root)
    tot = np.zeros(a.shape, np.float32)
    cnt = np.zeros(a.shape, np.uint8)
    for n in nbrs:
        d = (_origin_world(cfg, n, root) - p0) / vw     # (dx, dy, dz) in voxels
        v, ok = _place(_read_level(cfg, n, level, ctx), (d[2], d[1], d[0]), a.shape)
        if not ok:
            continue
        m = v > 0                                      # exactly 0 == not covered by this one
        tot[m] += v[m]
        cnt[m] += 1
    nb = np.zeros(a.shape, np.float32)
    np.divide(tot, np.maximum(cnt, 1), out=nb)
    return a, nb, cnt > 0


def _coarsen(a, nb, cov, ybin, xbin):
    """Per-cell statistics on the gain grid: one cell per z voxel, `ybin`x`xbin` laterally.

    z is deliberately NOT binned -- the defect changes fast along z and slowly laterally,
    and a coarser z grid puts healthy and defective planes in one cell, which
    over-brightens the healthy ones and under-corrects the dark ones.

    `obs` is the median over ALL of a cell's voxels. Taking it over only the voxels above
    background reports the few survivors in a mostly-dead cell and understates the defect
    several-fold.
    """
    nz, ny, nx = a.shape
    ny -= ny % ybin
    nx -= nx % xbin
    r = lambda A: A[:, :ny, :nx].reshape(nz, ny // ybin, ybin, nx // xbin, xbin)
    obs = np.median(r(a), axis=(2, 4))
    nbf = np.where(r(cov), r(nb), np.nan)
    with np.errstate(invalid="ignore"):
        nbo = np.nanmedian(nbf, axis=(2, 4))
    nbo = np.nan_to_num(nbo)
    covf = r(cov).mean(axis=(2, 4))
    nbfg = (np.where(r(cov), r(nb), 0.0) > 0).mean(axis=(2, 4))
    return obs.astype(np.float32), nbo.astype(np.float32), covf, nbfg


def _signal(obs, bg, bg_std, sigma):
    """Does this cell have its own signal above background? Ramps bg -> bg + sigma*bg_std.

    In units of the background NOISE, not of the background LEVEL: `3*bg` was 4.0 sd on
    one experiment and 3.1 on another, i.e. the same constant meaning two different tests.
    """
    top = bg + sigma * max(bg_std, 1e-6)
    return np.clip((obs - bg) / max(top - bg, 1e-6), 0.0, 1.0)


def _presence(nbfg, covf, k, min_cov=0.05, lo=0.03, hi=0.12):
    """Do the neighbours still show specimen HERE, laterally local?

    The expectation falls back to a per-z-slice level where nothing covers a cell, which is
    a survivorship statistic -- a cell can inherit the brightness of tissue millimetres
    away. This is the local veto on that, and it is not redundant with a spatially local
    expectation: measured, replacing the slice fallback with a nearest-evidence fill does
    NOT remove the need for it. The expectation answers "what level should be here"; this
    answers "is there specimen here at all".
    """
    from scipy.ndimage import uniform_filter
    ev = (covf >= min_cov).astype(np.float32)
    num = uniform_filter(np.where(covf >= min_cov, nbfg, 0.0).astype(np.float32),
                         size=(1, k, k), mode="nearest")
    den = uniform_filter(ev, size=(1, k, k), mode="nearest")
    frac = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    return np.clip((frac - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def expectation(obs, nbo, covf, nbfg, bg, signal, min_cov=0.05, min_fg=0.10,
                collapse=0.5, floor_t=0.25, mult=3.0):
    """What should this cell read? One rule, three cases:

      1. a neighbour covers it        -> the neighbour's own value (a measurement; wins)
      2. no neighbour, slice holds    -> that z slice's neighbour level
      3. no neighbour, slice COLLAPSED-> this column's own healthy-depth plateau, but only
                                         where the tile still shows signal

    Case 3 exists because the slice level falls below what a column's own plateau says when
    the specimen ended inside the covered strips but persists here; requiring signal stops
    a column sitting at background from demanding a 25x lift on noise.
    """
    have = covf >= min_cov
    level = np.where(have, nbo, 0.0).astype(np.float32)
    for i in range(level.shape[0]):
        if have[i].any():
            level[i][~have[i]] = np.median(nbo[i][have[i]])
    with np.errstate(invalid="ignore"):
        plateau = np.nanmedian(np.where(obs > mult * bg, obs, np.nan), axis=0)
    plateau = np.nan_to_num(plateau)[None, :, :]
    ev = have & (nbfg >= min_fg)
    use = (~ev) & (level < collapse * plateau) & (signal >= floor_t)
    return np.where(use, np.broadcast_to(plateau, obs.shape), level).astype(np.float32)


def _precondition(r, ev, valid, cfg):
    """The tile's level against its neighbours where healthy, and whether it is usable.

    A uniform offset is a per-tile gain, not a local defect. If it approaches the mask
    threshold's distance from 1.0 then the offset ALONE drops half the tile below the
    threshold and detection can no longer separate local from global: measured, a uniform
    0.93x takes a tile from 12% to 29% of its grid masked. Half the feather width is the
    usable bound.
    """
    h = ev & valid & (r > 0.8) & (r < 1.2)
    level = float(np.median(r[h])) if h.any() else 1.0
    tol = 0.5 * (1.0 - edge_r(cfg))
    return level, abs(level - 1.0) <= tol, tol


def gain_field(cfg, setup, nbrs=None, ctx=None, verbose=True):
    """The per-cell gain for one tile, plus the grid it lives on and a report.

    Returns `(gain, level, report)`: `gain` on the (z, y/ybin, x/xbin) analysis grid,
    `level` the pyramid level it was estimated at, and `report` a dict worth printing and
    keeping.
    """
    from scipy.ndimage import median_filter
    p = params(cfg)
    n_levels = len(_levels_in(tile_dir(cfg, setup), cfg["output_format"]))
    if not n_levels:
        raise FileNotFoundError(f"no pyramid under {tile_dir(cfg, setup)}")
    # Clamp to what this dataset actually has: the default analysis level is 4 because that
    # is where a voxel is a couple of microns on the experiments this was built for, but a
    # shallower pyramid must fall back to its deepest level rather than fail to open one.
    lvl = min(int(p["spotfix_level"]), n_levels - 1)
    root = _xml(cfg)
    nbrs = neighbours(cfg, setup, root) if nbrs is None else nbrs
    if not nbrs:
        raise RuntimeError(f"setup {setup} has no overlapping neighbours; spotfix has no "
                           f"independent measurement to correct against")

    stats = json.load(open(_config.target_path(cfg)))["setups"][str(setup)]
    bg, bg_std = float(stats["bg_mean"]), float(stats["bg_std"])

    a, nb, cov = neighbour_reference(cfg, setup, nbrs, lvl, ctx)
    vw = voxel_world(cfg, setup, lvl, root)
    cal = np.abs(np.diag(_sizes_and_transforms(root)[1][setup][:3, :3]))
    # voxel size in MICRONS: world units are level-0 lateral voxels, so one world unit is
    # the lateral pitch. Everything spatial below is a length divided by these.
    um_per_world = float(cfg.get("spotfix_um_per_world") or _lateral_um(cfg, root))
    vox_um = (vw[2] * um_per_world, vw[1] * um_per_world, vw[0] * um_per_world)  # z, y, x
    ybin = max(1, int(round(p["spotfix_cell_um"] / vox_um[1])))
    xbin = max(1, int(round(p["spotfix_cell_um"] / vox_um[2])))

    obs, nbo, covf, nbfg = _coarsen(a, nb, cov, ybin, xbin)
    del a, nb, cov
    signal = _signal(obs, bg, bg_std, float(p["spotfix_floor_sigma"]))
    exp = expectation(obs, nbo, covf, nbfg, bg, signal,
                      floor_t=float(p["spotfix_floor_t"]))
    valid = exp > 0
    ev = (covf >= 0.05) & (nbfg >= 0.10)
    r = np.where(valid, obs / np.maximum(exp, 1e-9), 1.0)

    lev, ok, tol = _precondition(r, ev, valid, cfg)
    kp = max(1, int(round(p["spotfix_presence_um"] / (ybin * vox_um[1]))))
    loc = _presence(nbfg, covf, kp)
    lt, ft = float(p["spotfix_loc_t"]), float(p["spotfix_floor_t"])
    gate = np.maximum(np.where(loc >= lt, loc, 0.0), np.where(signal >= ft, signal, 0.0))

    er = edge_r(cfg)
    mask = valid & (r < er)
    allow = np.clip(mask.astype(np.float32) * gate, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        need = np.where(obs > 1e-6, exp / np.maximum(obs, 1e-6), np.inf)
    # A non-finite need falls back to 1.0 (no correction) -- the safe direction, and what
    # keeps a tile's unimaged z padding (obs == 0) from being amplified out of nothing.
    capped = np.where(np.isfinite(need), np.maximum(need, 1.0), 1.0)
    g = np.maximum(1.0 + allow * (capped - 1.0), 1.0).astype(np.float32)

    cell_um = (vox_um[0], ybin * vox_um[1], xbin * vox_um[2])
    fp = (max(1, int(round(p["spotfix_smooth_z_um"] / cell_um[0]))),
          max(1, int(round(p["spotfix_smooth_lat_um"] / cell_um[1]))),
          max(1, int(round(p["spotfix_smooth_lat_um"] / cell_um[2]))))
    g = np.maximum(median_filter(g, size=fp, mode="nearest"), 1.0)

    report = {
        "setup": setup, "level": lvl, "neighbours": nbrs,
        "bg_mean": bg, "bg_std": bg_std,
        "voxel_um_zyx": [round(v, 4) for v in vox_um],
        "cell_um_zyx": [round(v, 3) for v in cell_um],
        "bins_yx": [ybin, xbin], "grid": list(g.shape),
        "edge_r": round(er, 6), "edge_step_pct": round(100 * (1 / er - 1), 2),
        "despeckle_cells": list(fp),
        "healthy_level": round(lev, 4), "precondition_ok": bool(ok),
        "precondition_tol": round(tol, 4),
        "cells_masked_pct": round(100 * float(mask.mean()), 3),
        "cells_gained_pct": round(100 * float((g > 1).mean()), 3),
        "gain_max": round(float(g.max()), 3),
        "gain_median_over_gained": (round(float(np.median(g[g > 1])), 4)
                                    if (g > 1).any() else 1.0),
    }
    if verbose:
        print(f"spotfix: setup {setup} level {lvl}, {len(nbrs)} neighbour(s) {nbrs}")
        print(f"spotfix: voxel {tuple(round(v, 3) for v in vox_um)} um -> cell "
              f"{tuple(round(v, 1) for v in cell_um)} um ({ybin}x{xbin} lateral bins), "
              f"grid {g.shape}")
        print(f"spotfix: mask r < {er:.4f} (edge step {100 * (1 / er - 1):.1f}%) -> "
              f"{report['cells_masked_pct']:.2f}% of cells; gain > 1 on "
              f"{report['cells_gained_pct']:.2f}%, median "
              f"{report['gain_median_over_gained']:.3f}x, max {report['gain_max']:.2f}x")
        print(f"spotfix: despeckle {p['spotfix_smooth_z_um']:.1f} um z / "
              f"{p['spotfix_smooth_lat_um']:.1f} um lateral = {fp} cells")
        print(f"spotfix: healthy level {lev:.4f} vs its neighbours "
              f"(tolerance +-{tol:.4f}) -- precondition {'OK' if ok else 'FAILED'}")
    return g, lvl, report


def _lateral_um(cfg, root=None):
    """Microns per world unit, from <voxelSize>. World units are level-0 LATERAL voxels
    (the calibration folds the z anisotropy in), so this is the lateral pitch."""
    root = _xml(cfg) if root is None else root
    vs = root.find(".//voxelSize")
    if vs is None:
        return 1.0
    vx, _vy, _vz = (float(t) for t in vs.findtext("size").split())
    return vx


# ─── backup, then write into the dataset ─────────────────────────────────────────


def tile_dir(cfg, setup):
    """The directory holding one setup's whole pyramid in the store spotfix writes."""
    root = cfg["output_intensity_path"]
    if cfg["output_format"] == "n5":
        return f"{root}/setup{setup}"
    return f"{root}/s{setup}-t0.zarr"


def backup_tile(cfg, setup):
    """RENAME the tile aside and return the backup path.

    A rename, not a copy and not a delete: atomic on one filesystem, costs nothing for a
    70 GB tile, and if the run dies halfway the previous pyramid is still whole. Deleting
    is also the operation that parks indefinitely on an smbfs mount, so this stage never
    does it -- clearing old backups is left to whoever can see how many they want to keep.
    """
    src = tile_dir(cfg, setup)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"no tile to fix at {src}")
    n = 0
    while True:
        dst = f"{src}.prespotfix" + (f".{n}" if n else "")
        if not os.path.exists(dst):
            break
        n += 1
    os.rename(src, dst)
    print(f"spotfix: setup {setup} previous version kept at {dst}")
    return dst


def _levels_in(path, fmt):
    """Canonical (Z, Y, X) shape of each pyramid level under a tile directory."""
    spec = _SPEC[fmt]
    shapes, level = [], 0
    while True:
        p = (f"{path}/timepoint0/s{level}" if fmt == "n5" else f"{path}/{level}")
        if not os.path.exists(f"{p}/{spec['meta']}"):
            break
        arr = ts.open({"driver": spec["driver"],
                       "kvstore": {"driver": "file", "path": p}},
                      open=True, read=True).result()
        shapes.append(canonical_shape(arr.domain.shape, spec["order"]))
        level += 1
    return shapes


def _sampler(g, out_zyx):
    """Trilinear sampling of the coarse gain onto a full-resolution region, one z plane at
    a time so nothing shard-sized is ever materialised as float.

    Endpoint-aligned, matching `scipy.ndimage.zoom(grid_mode=False)`: output index j maps
    to coarse index j*(n_coarse-1)/(n_out-1). A cell-width mapping is off by half a cell
    and puts the gain on the wrong cell at every boundary.
    """
    idx, wgt = [], []
    for k in range(3):
        n_c, n_o = g.shape[k], out_zyx[k]
        c = (np.arange(n_o) * ((n_c - 1) / (n_o - 1))) if n_o > 1 else np.zeros(1)
        i0 = np.clip(np.floor(c).astype(np.int64), 0, max(n_c - 1, 0))
        i1 = np.clip(i0 + 1, 0, max(n_c - 1, 0))
        idx.append((i0, i1))
        wgt.append((c - i0).astype(np.float32))

    def plane(z, y0, y1, x0, x1):
        (iz0, iz1), (iy0, iy1), (ix0, ix1) = idx
        tz, ty, tx = wgt
        p = (1.0 - tz[z]) * g[iz0[z]] + tz[z] * g[iz1[z]]            # (ny_c, nx_c)
        ys, xs = slice(y0, y1), slice(x0, x1)
        p = ((1.0 - ty[ys])[:, None] * p[iy0[ys]] + ty[ys][:, None] * p[iy1[ys]])
        p = ((1.0 - tx[xs])[None, :] * p[:, ix0[xs]] + tx[xs][None, :] * p[:, ix1[xs]])
        return p

    return plane


def _apply_shard(canon, plane, z0, y0, y1, x0, x1, hi):
    """Multiply one shard by the gain, in place, plane by plane.

    `np.rint`, not truncation: casting a gain that interpolates to 0.99999997 downwards
    takes a whole tile down by one gray level everywhere it is applied.
    """
    for k in range(canon.shape[0]):
        g = plane(z0 + k, y0, y1, x0, x1)
        if g.max() <= 1.0 + 1e-7:
            continue
        v = canon[k].astype(np.float32) * g
        canon[k] = np.clip(np.rint(v), 0, hi).astype(canon.dtype)
    return canon


async def _write(cfg, setup, g, backup, report):
    """Read the backed-up tile, apply the gain, write the fixed pyramid in its place."""
    fmt = cfg["output_format"]
    spec = _SPEC[fmt]
    order = spec["order"]
    ctx = stores.context(cfg)

    shapes = _levels_in(backup, fmt)
    if not shapes:
        raise RuntimeError(f"no pyramid levels found under {backup}")
    zyx = shapes[0]
    factors = [(max(round(shapes[0][0] / s[0]), 1), max(round(shapes[0][1] / s[1]), 1),
                max(round(shapes[0][2] / s[2]), 1)) for s in shapes]

    src_path = (f"{backup}/timepoint0/s0" if fmt == "n5" else f"{backup}/0")
    src = canonical_view(stores._open(
        {"driver": spec["driver"], "kvstore": {"driver": "file", "path": src_path}},
        ctx, open=True, read=True), order)
    dtype_name = src.dtype.name
    hi = np.iinfo(np.dtype(dtype_name)).max

    out, shard, out_path = stores.open_output_array(
        cfg, setup, 0, _in_order(zyx, order), dtype_name, ctx)
    out_c = canonical_view(out, order)
    sz, sy, sx = canonical_shape(shard, order)
    Z, Y, X = zyx
    origins = [(oz, oy, ox) for oz in range(0, Z, sz)
               for oy in range(0, Y, sy) for ox in range(0, X, sx)]
    plane = _sampler(g, zyx)

    # Same shape as `correct`: bounded shards in flight, the numpy in a thread pool so it
    # overlaps across cores instead of blocking the event loop, both sized from the
    # reservation via `stores.slots`.
    itemsize = np.dtype(dtype_name).itemsize
    limit = max(1, min(len(origins),
                       stores.memory_budget() // max(sz * sy * sx * itemsize, 1)))
    sem = asyncio.Semaphore(limit)
    n_cores = stores.slots(_config.stage_cores(cfg, "spotfix"))
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=n_cores)
    print(f"spotfix: setup {setup} writing {zyx} -> {out_path}")
    print(f"spotfix: {len(origins)} shard(s) of {(sz, sy, sx)}, {limit} in flight, "
          f"{n_cores} kernel thread(s)")

    async def one(oz, oy, ox):
        z1 = min(oz + sz, Z); y1 = min(oy + sy, Y); x1 = min(ox + sx, X)
        async with sem:
            canon = await src[oz:z1, oy:y1, ox:x1].read(order="C")
            canon = await asyncio.get_running_loop().run_in_executor(
                pool, _apply_shard, canon, plane, oz, oy, y1, ox, x1, hi)
            txn = ts.Transaction(atomic=False)
            await out_c[oz:z1, oy:y1, ox:x1].with_transaction(txn).write(canon)
            await txn.commit_async()
        bar.advance()

    bar = Progress(len(origins), f"spotfix setup {setup}")
    try:
        await asyncio.gather(*(one(*o) for o in origins))
    finally:
        bar.close()
        pool.shutdown(wait=True)

    # The pyramid, mean-downsampled from the FIXED level 0 with the factors the tile
    # already had, so the levels stay consistent with what the dataset advertises.
    level0 = ts.open({"driver": spec["driver"],
                      "kvstore": {"driver": "file", "path": out_path}},
                     context=ctx, open=True, read=True).result()
    for level in range(1, len(factors)):
        ds = ts.open({"driver": "downsample", "base": level0.spec(),
                      "downsample_factors": _in_order(factors[level], order),
                      "downsample_method": "mean"},
                     context=ctx, open=True, read=True).result()
        lvl, _, _ = stores.open_output_array(
            cfg, setup, level, list(ds.domain.shape), dtype_name, ctx)
        ltxn = ts.Transaction(atomic=False)
        print(f"spotfix: setup {setup} level {level} {tuple(ds.domain.shape)}")
        await lvl.with_transaction(ltxn).write(ds)
        await ltxn.commit_async()
    stores.write_group_metadata(cfg, setup, factors)
    report["levels"] = len(factors)
    report["backup"] = backup
    report["out_path"] = out_path


def fix_tile(cfg, setup, dry_run=False, force=False):
    """Fix one tile in place, keeping the previous version alongside.

    Order matters: the gain is measured from the CURRENT tile, then that tile is renamed
    aside, then the fixed version is written to the original path. Nothing is deleted and
    the source is never the file being written.
    """
    t0 = time.perf_counter()
    g, lvl, report = gain_field(cfg, setup)
    if not report["precondition_ok"] and not force:
        raise RuntimeError(
            f"spotfix: setup {setup} sits at {report['healthy_level']:.4f} of its "
            f"neighbours where healthy, outside +-{report['precondition_tol']:.4f}. That is "
            f"a per-tile GAIN error, not a local defect; correcting it here would smear a "
            f"uniform offset into a large spatially varying correction. Re-run the "
            f"intensity correction first, or pass force=True if you know better.")
    if report["cells_gained_pct"] == 0.0:
        print(f"spotfix: setup {setup} nothing to fix (no cell wants a gain above 1); "
              f"leaving it untouched")
        report["skipped"] = True
        return report
    if dry_run:
        print(f"spotfix: setup {setup} dry run, nothing written")
        report["skipped"] = True
        return report
    backup = backup_tile(cfg, setup)
    try:
        asyncio.run(_write(cfg, setup, g, backup, report))
    except BaseException:
        # Put the original back if the write never got going, so a failure does not leave
        # the dataset without this tile.
        dst = tile_dir(cfg, setup)
        if not os.path.exists(dst):
            os.rename(backup, dst)
            print(f"spotfix: setup {setup} failed before writing; restored {dst}")
        raise
    report["seconds"] = round(time.perf_counter() - t0, 1)
    path = os.path.join(str(cfg["results_root"]), f"spotfix_setup{setup}.json")
    stores._atomic_write_json(path, report)
    print(f"spotfix: setup {setup} done in {report['seconds']:.1f}s; report at {path}")
    return report
