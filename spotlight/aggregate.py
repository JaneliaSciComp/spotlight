"""Stage: solve one gain per tile (or per camera) from tile overlaps, and write the target.

A single job, between `int-stats` and `correct`. Two overlapping tiles image the SAME
physical tissue in their shared region, so a mismatch there is sensor gain, not content --
which is what lets this isolate gain from the specimen. The overlaps are discovered from
the BigStitcher `dataset.xml` geometry, so no camera adjacency is assumed.

The dataset.xml parsing and overlap geometry live here rather than in a `geometry.py`
because every one of those functions has exactly one caller: `cmd_aggregate`. If this file
grows past ~500 lines, that is the seam to cut on.

What it does, in order:

1. Read every setup's per-tile stats (threshold, foreground mean/std) and classify each
   tile with `_classify`.
2. Discover every physically-overlapping tile pair from the SpimData2 `dataset.xml`
   (config key `dataset_xml`), within-camera and cross-camera (`_all_overlap_pairs`).
3. Per pair, one robust log-gain constraint from the two tiles' foreground medians in the
   shared region (`_pair_gain_constraint`, tuned by `gain_estimator`).
4. Solve a gain `g_s` by regularized global least squares (`_solve_tile_gains`); the
   correction is `raw/g_s`. `gain_grouping="camera"` (default) shares one gain across a
   camera's tiles, which robustly removes the sensor-level step between cameras -- many
   overlaps per camera pin it, so it needs almost no shrinkage. `"tile"` solves one per
   tile (opt-in; keep `gain_lambda` at ~0.1 to damp drift).
5. Write `tile_gains.json` (gains, grouping/estimator, per-camera summary) and
   `intensity_target.json`: each setup's stats plus `corrected_mean`/`corrected_std`,
   which encode `g_s` so the correction stage computes `raw/g_s` without knowing about
   gains at all. `target_mean`/`target_std` are the median over tiles of the
   gain-equalized mean/std.

Overlap-driven gains preserve real texture: neighbours whose shared region agrees keep
equal gains, so genuine content differences survive. Matching every tile to one mean would
erase them.

CLI: `python -m spotlight int-aggregate`.
"""

import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .config import (camera_groups, camera_of, stage_cores, stats_path,
                     target_path, tile_list)
from .fields import _check_basic_mode, basic_model
from .formats import _SPEC, _input_location, canonical_view
from .stores import _atomic_write_json, slots, source_pyramid_factors
from .tilestats import (
    MIN_FG_FRACTION, MIN_FOREGROUND, THRESHOLD_METHODS, _classify, limits,
    open_downsampled,
)


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

    Each <ViewRegistration> lists its <ViewTransform>s outermost-first (e.g. "Stitching
    Transform", then "Translation to Regular Grid", then "calibration" last), and the
    BDV/BigStitcher convention composes them as `M_total = M_0 @ M_1 @ ... @ M_last` -- so
    the LAST-listed transform (calibration: pixel index -> physical units) applies FIRST
    to a raw pixel coordinate. Points are (x, y, z, 1) column vectors, matching the
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
    """(pairs, sizes, transforms): every `(setup_a, setup_b, world_bbox)` whose two
    world-space bounding boxes intersect, within-camera and cross-camera alike.

    Within-camera overlaps are what let the per-tile solve tell gain from content: a tile
    dim from its own gain reads low against its same-camera neighbours on shared tissue,
    one dim from content agrees with them. Overlap comes purely from the geometry, so no
    camera adjacency is assumed.

    The pairwise test is O(N^2) but vectorized per row -- a few seconds for a few thousand
    tiles, and only real overlaps are materialized.
    """
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
    """The setup's downsampled (z, y, x) ranges for a world-space bbox.

    Takes a precomputed (fz, fy, fx) `factor` (level `stats_scale`) and downsampled
    `shape` rather than re-opening the array: across this many pairs, one pyramid re-read
    per pair would dominate the solve.
    """
    px_min, px_max = _world_bbox_to_pixels(setup, world_bbox, sizes, transforms)
    fz, fy, fx = factor
    x_range = (int(px_min[0] // fx), int(np.ceil(px_max[0] / fx)))
    y_range = (int(px_min[1] // fy), int(np.ceil(px_max[1] / fy)))
    z_range = (int(px_min[2] // fz), int(np.ceil(px_max[2] / fz)))
    Z, Y, X = shape
    return ((max(0, z_range[0]), min(Z, z_range[1])),
            (max(0, y_range[0]), min(Y, y_range[1])),
            (max(0, x_range[0]), min(X, x_range[1])))


GAIN_FLOOR_MODES = ("tile", *THRESHOLD_METHODS)


def _floor_setting(cfg):
    """`gain_floor` validated: a float, or one of GAIN_FLOOR_MODES.

    "tile" follows whatever `tile_threshold` selected. "otsu"/"li" name a METHOD directly,
    which lets the gain solve gate on tissue while `tile_threshold = 0` puts the apply
    stage in its all-pixels `uniform` mode. The two answer different questions and need
    not agree.
    """
    mode = cfg.get("gain_floor") or "tile"
    if not isinstance(mode, str):
        return float(mode)
    if mode not in GAIN_FLOOR_MODES:
        raise ValueError(f"gain_floor={mode!r} must be one of {GAIN_FLOOR_MODES} "
                         f"or a number")
    return mode


def _tile_floor(st, setting, setup):
    """The floor for one tile under `setting`. A number applies to every tile, so the
    pair's `max(...)` reduces to it and no separate override plumbing is needed."""
    if not isinstance(setting, str):
        return float(setting)
    if setting == "tile":
        return float(st["threshold"])
    got = (st.get("thresholds") or {}).get(setting)
    if got is None:
        raise RuntimeError(
            f"gain_floor={setting!r} needs the per-method thresholds, which setup "
            f"{setup}'s stats file predates; re-run the stats stage "
            f"(`python -m spotlight int-stats`) or set gain_floor to 'tile' or a number")
    return float(got)


def _floor_label(setting, cache, setups):
    """What the gate is actually gating at, for the header line.

    Reports the SOURCE the stats stage recorded, not the `gain_floor` setting: the default
    setting only says "use the per-tile threshold", not what produced it. Printing the
    setting made a run with `tile_threshold = "li"` announce `floor=otsu`.
    """
    thrs = [cache[s]["thr"] for s in setups if s in cache]
    if not thrs:
        return str(setting)
    span = (f"{thrs[0]:.0f}" if len(set(thrs)) == 1
            else f"{min(thrs):.0f}-{max(thrs):.0f}")
    if not isinstance(setting, str):
        return f"{span} (gain_floor)"
    return f"per-tile {span} (gain_floor={setting})"


def _pair_gain_constraint(setup_a, setup_b, world_bbox, sizes, transforms, cache, order,
                          estimator="intersection", min_fg=None, min_frac=None,
                          reject=None, accept=None):
    """One log-gain constraint for an overlapping tile pair.

    Returns `(a, b, log(med_a) - log(med_b), weight)` -- the target for `log g_a - log
    g_b`, so `raw/g` equalizes the two on shared tissue -- or None if the overlap holds
    too little foreground to trust. `cache[s]` carries the tile's open array, downsample
    factor, shape and threshold.

    `estimator` selects how the two medians are taken over the shared region:
      * "intersection" (default): median over the voxels BOTH tiles call foreground. Matched
        voxels, so the ratio is the true gain even where the two thresholds differ -- but a
        few voxels of misregistration bias it, which is why the other exists.
      * "independent": each tile's own foreground distribution above the common floor.
        Robust to registration error, at the cost of a possible population bias. On this
        data, switching to intersection recovered the true camera step (9/10 ratio 1.51 ->
        1.66) while the common floor had already removed the threshold-mismatch bias --
        hence the default.

    Both tiles are gated at a COMMON floor `max(thr_a, thr_b)`, never each at its own
    threshold. Per-tile thresholds routinely differ by hundreds of gray levels, and gating
    each at its own compares medians over DIFFERENT intensity populations of the SAME
    tissue: the lower-threshold tile sweeps in a band of dimmer voxels its neighbour
    excludes, dragging its median down and manufacturing a gain difference (observed: a
    tile reading ~identically to its neighbours got 0.92 purely from a ~200-level
    threshold gap). A shared floor makes both medians span one population.
    """
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
    min_fg = MIN_FOREGROUND if min_fg is None else min_fg
    min_frac = MIN_FG_FRACTION if min_frac is None else min_frac
    # `gain_floor` (a number) replaces the per-tile Otsu split for the gain comparison
    # only; None means use it. Either way the floor is COMMON to both tiles, which is
    # what stops a threshold mismatch manufacturing a gain (see the docstring above).
    # Already resolved per tile by `_tile_floor`; a numeric setting made both equal, so
    # this max is the common floor in every case.
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
    why = None
    if fg_a.size < min_fg or fg_b.size < min_fg:
        why = (f"foreground {min(fg_a.size, fg_b.size)} < min_overlap_foreground {min_fg}")
    elif fg_a.size < min_frac * va.size or fg_b.size < min_frac * vb.size:
        frac = min(fg_a.size / max(va.size, 1), fg_b.size / max(vb.size, 1))
        why = (f"foreground {frac:.2%} of the overlap < min_overlap_fraction {min_frac:.2%}")
    if why:
        # Report the SHORTFALL, not just the rejection: "0/2 usable" cannot tell you
        # whether the overlap missed by a factor of two or by three orders of magnitude,
        # and that is the difference between lowering a threshold and not having data.
        if reject is not None:
            reject.append(f"  ({setup_a},{setup_b}) rejected: {why} "
                          f"[overlap {va.size} voxels, thr {thr:.0f}, "
                          f"estimator {estimator}]")
        return None
    med_a, med_b = float(np.median(fg_a)), float(np.median(fg_b))
    # FLOOR SENSITIVITY. A true multiplicative gain is the same ratio wherever the floor
    # is put; a ratio that moves with the floor is measuring the two tiles' intensity
    # DISTRIBUTIONS differing in shape (different tissue, or misregistration), not their
    # gain. Cheap to check -- re-take both medians an octave up -- and worth checking,
    # because the gate cannot distinguish the two and the drift is what tells you the
    # constraint is about content rather than sensor.
    if accept is not None:
        accept.append(f"  ({setup_a},{setup_b}) ratio {med_a / med_b:.3f} from "
                      f"{fg_a.size} voxels above floor {thr:.0f} "
                      f"({fg_a.size / max(va.size, 1):.2%} of a {va.size}-voxel "
                      f"overlap){_drift(fg_a, fg_b, estimator)}")
    return (setup_a, setup_b, float(np.log(med_a) - np.log(med_b)),
            float(min(fg_a.size, fg_b.size)))


def _drift(fg_a, fg_b, estimator):
    """Is the pair's ratio the same at low and high intensity? Empty string if so.

    A true multiplicative gain gives ONE ratio at every intensity. A ratio that moves with
    intensity means the two distributions differ in shape -- different tissue in the
    overlap, or misregistration -- and the gate cannot tell that from a gain, so it has to
    be said out loud.

    Only for "intersection", where the gate's mask makes `fg_a[i]` and `fg_b[i]` the SAME
    voxel and the ratio is per-voxel. The obvious alternative -- re-take both medians
    above a higher absolute floor -- is wrong in a way that looks convincing: a common
    floor cuts more off the dimmer tile, so a PERFECT gain of 2 reports a 43% drift.
    "independent" has no pairing, so it gets no check rather than a misleading one.
    """
    if estimator != "intersection":
        # Say so, rather than returning "" -- a silent pass and "not checked" look
        # identical in the log, and here they mean opposite things.
        return " (stability not checked: 'independent' has no voxel pairing)"
    if fg_a.size < 128:
        return " (stability not checked: too few voxels)"
    a = fg_a.astype(np.float64)
    r = fg_b.astype(np.float64) / np.maximum(a, 1e-12)
    split = np.median(a)
    lo, hi = r[a <= split], r[a > split]
    if lo.size < 32 or hi.size < 32:
        return ""
    r_lo, r_hi = float(np.median(lo)), float(np.median(hi))
    if r_lo <= 0 or abs(r_hi / r_lo - 1) <= 0.15:
        return ""
    return (f" -- UNSTABLE: b/a is {r_lo:.2f} on the dim half of the overlap but "
            f"{r_hi:.2f} on the bright half, so this is a distribution difference more "
            f"than a gain; treat the solved gain as unreliable")


def _solve_tile_gains(constraints, setups, lam=0.1, group_of=None):
    """Multiplicative gains from all pairwise overlap constraints, by regularized global
    least squares in log space, solved as one sparse `scipy.sparse.linalg.lsqr`:

        min  Σ_(a,b) w_ab (log g_A - log g_B - d_ab)^2  +  lam Σ_g (log g_g)^2

    `d_ab = log(med_a) - log(med_b)`; `w_ab` is the overlap foreground size, normalized by
    its median so `lam` is scale-free; A/B are the GROUPS tiles a/b belong to.
    `group_of=None` gives one gain per tile; a setup->camera map gives one per CAMERA.
    Either way the return maps every setup to its gain.

    The regularization pulls log-gains toward 0. It fixes the otherwise-free global gauge
    (overlaps constrain only differences), damps groups with few or noisy constraints, and
    pins an unconstrained group to exactly 1.0. As lam -> 0 it is a pure gauge anchor --
    the minimum-norm sum-zero solution -- so per camera lam can be tiny (many constraints,
    so the data dominates: lam 1e-6 and a hard gauge agree to RMS 0). Per tile keep lam >~
    0.01 to damp the smooth drift null-mode.
    """
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











def cmd_aggregate(cfg):
    """The whole reduce step; the module docstring lists its five stages.

    Two things not stated there. The pair constraints run across a thread pool -- the
    tensorstore reads release the GIL, so this uses the cores the bsub job asked for.

    And how `g_s` reaches the unchanged `apply` stage: with equalized centre/spread M, S
    (median over usable tiles of own_mean/g, own_std/g) and target_mean=M, target_std=S,
    storing corrected_mean=g_s*M and corrected_std=g_s*S makes apply's
    `(raw-mean_i)*(target_std/std_i)+target_mean` compute exactly `raw/g_s` -- a pure
    per-tile gain, texture preserved.
    """
    setups_all = tile_list(cfg)
    _, order = _input_location(cfg, setups_all[0], cfg["stats_scale"])
    n_cores = slots(stage_cores(cfg, "aggregate"))
    # gain grouping: "camera" (default) shares one gain across a camera's tiles --
    # robust (many constraints per node) and matches the sensor-level step we correct;
    # "tile" is the old per-tile mode (opt in). Regularization is just `gain_lambda`;
    # the gauge is its lam->0 limit, so per-camera defaults to a tiny lam and per-tile
    # to 0.1 (needs damping). Estimator "intersection" (default) matches voxels.
    grouping = cfg.get("gain_grouping") or "camera"
    if grouping not in ("camera", "tile"):
        raise ValueError(f"gain_grouping={grouping!r} must be 'camera' or 'tile'")
    # `gain_floor`: "tile" (default) means the per-tile threshold the STATS stage
    # recorded -- so it already follows `tile_threshold`, and setting that to "li" makes
    # this floor Li-derived without touching anything here. A number overrides it for the
    # gain comparison only; prefer `tile_threshold` when the thresholds are themselves
    # wrong, since that also fixes the classification and the apply mask.
    #
    # "otsu" is the old spelling of "tile", accepted so existing tomls keep working. It
    # was always a misnomer -- the value came from the stats files, whatever produced
    # them -- and it printed `floor=otsu` on runs that were using Li, which is what
    # prompted the rename.
    floor_setting = _floor_setting(cfg)
    estimator = cfg.get("gain_estimator") or "intersection"
    if estimator not in ("intersection", "independent"):
        raise ValueError(f"gain_estimator={estimator!r} must be 'intersection' or 'independent'")
    # `or` not `.get(default)`: DEFAULTS now carries gain_lambda=None as "unset", and
    # the resolved value depends on `grouping`, so the fallback has to happen here.
    lam = float(cfg.get("gain_lambda") or (1e-6 if grouping == "camera" else 0.1))

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
            f"stage (`python -m spotlight emptiness`) -- `_classify` needs it to tell "
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
                   "thr": _tile_floor(stat_cache[s], floor_setting, s)}
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
    lim = limits(cfg)
    nonempty = {s for s in setups if _classify(stat_cache[s], lim) != "empty"}
    pairs = [(a, b, bb) for (a, b, bb) in pairs if a in nonempty and b in nonempty]
    print(dt.now(), f"aggregate: {len(setups)} tiles ({len(nonempty)} non-empty), "
                    f"{len(pairs)} overlapping pairs, {n_cores} threads, "
                    f"apply_basic={cfg['apply_basic']}, grouping={grouping}, "
                    f"estimator={estimator}, lambda={lam:g}, "
                    f"floor={_floor_label(floor_setting, cache, setups)}", flush=True)

    # robust log-gain constraint per pair (parallel)
    # Overridable because a sparse specimen can be genuinely informative while failing
    # a gate tuned for dense tissue. Lowering them trades a noisier gain for having one
    # at all -- see the guidance printed below when nothing survives.
    min_fg = float(cfg.get("min_overlap_foreground") or MIN_FOREGROUND)
    min_frac = float(cfg.get("min_overlap_fraction") or MIN_FG_FRACTION)
    reject, accept = [], []
    with ThreadPoolExecutor(max_workers=n_cores) as pool:
        raw = list(pool.map(
            lambda p: _pair_gain_constraint(p[0], p[1], p[2], sizes, transforms, cache,
                                            order, estimator, min_fg, min_frac, reject,
                                            accept),
            pairs))
    constraints = [c for c in raw if c is not None]
    print(dt.now(), f"aggregate: {len(constraints)}/{len(pairs)} usable overlap constraints",
          flush=True)
    for line in accept:
        print(line, flush=True)
    if reject:
        print(dt.now(), f"aggregate: {len(reject)} pair(s) rejected:", flush=True)
        for line in reject[:20]:
            print(line, flush=True)
        if len(reject) > 20:
            print(f"  ... and {len(reject) - 20} more", flush=True)
    if not constraints:
        print(dt.now(),
              "aggregate: NO usable constraints, so no gain can be solved. For sparse "
              "specimens, in order:\n"
              "  1. gain_estimator = \"independent\"  -- the default 'intersection' "
              "counts only voxels BOTH tiles call foreground, which a sparse overlap "
              "rarely has; 'independent' uses each tile's own foreground.\n"
              "  1b. tile_threshold = <number>  -- if the per-tile Otsu thresholds "
              "disagree wildly (they are printed by the stats stage), they are splitting "
              "different populations. One floor for every tile fixes the gain floor, the "
              "empty/bimodal classification AND the apply mask together.\n"
              "  2. lower min_overlap_foreground (default "
              f"{MIN_FOREGROUND}) and/or min_overlap_fraction (default "
              f"{MIN_FG_FRACTION}) to the shortfalls above.\n"
              "  3. raise gain_lambda -- a constraint from few voxels is noisy, and the "
              "regularization is what stops that noise becoming a gain.\n"
              "If the rejections above show single-digit foreground counts, the overlaps "
              "genuinely hold no shared signal and no setting will fix that.", flush=True)

    # per-camera groups every tile in a camera onto one gain; per-tile leaves each free
    groups = camera_groups(cfg)
    cam_of = {s: cam for cam, group in enumerate(groups) for s in group}
    group_of = cam_of if grouping == "camera" else None
    gains = _solve_tile_gains(constraints, setups, lam, group_of)

    # global equalized centre/spread from usable tiles' own stats / gain
    eq_means, eq_stds = [], []
    for s in setups:
        st = stat_cache[s]
        kind = _classify(st, lim)
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
        if _classify(st, lim) != "empty":
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
