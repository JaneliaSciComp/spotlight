"""On-disk layout: which driver, which path, which axis order.

A leaf module by necessity, not taste. `config.load_config` validates the configured
formats against `FORMATS`, and `stores.py` imports `config` -- so putting this table in
`stores.py` would close a cycle. Nothing here imports anything else in the package.

Internally every volume is canonical (Z, Y, X); this is the only place that knows how a
given format stores it. Input and output formats are independent (`input_format` /
`output_format`), so any input layout can be written to any output layout.

Two of the six are not this codebase's own convention:

* `zarr3_raw` (INPUT ONLY) reads TensorSwitch's actual on-disk OME-ZARR: a 3-D array
  declared with the non-standard axes order [x, y, z] (x FIRST, not last), nested under a
  "raw/" subgroup with "s"-prefixed scale dirs (s0, s1, ...) -- distinct from this
  package's own flat "0", "1", ... scale dirs.
* `zarr3_zyx` (typically the OUTPUT) writes the standard 3-D NGFF spatial order
  [z, y, x] instead of the default 5-D (t, c, z, y, x), so tools downstream of the
  correction that assume the standard order need no changes to read the result.
"""

import json
from datetime import datetime as dt

import numpy as np


FORMATS = ("n5", "zarr2", "zarr3_unsharded", "zarr3", "zarr3_zyx", "zarr3_raw")

# `tiff` is OUTPUT ONLY, and is not in `_SPEC`: every other format names a tensorstore
# driver, and tensorstore has no TIFF writer, so `correct.py` dispatches it to
# `tiffout.py` instead. Listing it as an input would promise a reader that does not
# exist -- hence two tuples rather than one. See tiffout.py for why TIFF cannot simply
# be another row in the table (single sequential file vs a directory of chunks).
OUTPUT_FORMATS = FORMATS + ("tiff",)

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


# ─── the 2-D planes written beside a camera ───────────────────────────────────
#
# `Flat-field.tif`, `Dark-field.tif` and `empty_fraction.tif` are all one frame-shaped
# plane, and every one of them is stored CANONICAL (Y, X) -- the row-major order Fiji and
# every other TIFF reader assumes, so they open upright and can be compared against the
# data without a mental transpose.
#
# `in_plane_order` is still real, but it describes the QSTACK, not these planes: BaSiC is
# fitted on a stack that keeps the source's (X, Y) for n5, so its fields come out in that
# order and are swapped to canonical on the way to disk. They used to be written in
# whatever order they arrived in, while the empty-fraction map was always canonical, which
# left the readers inferring orientation from aspect ratio -- an inference that cannot work
# on a square frame.


def in_plane_order(cfg):
    """The in-plane axis order every plane this package writes beside a camera is in.

    n5 keeps the source's (X, Y); every other format is transposed to (Y, X). Derived from
    `input_format` rather than guessed from a shape, so a square frame -- where both
    orientations fit -- still resolves.
    """
    return "xy" if cfg["input_format"] == "n5" else "yx"


def in_plane_swap(plane, order):
    """Convert a 2-D plane between canonical (Y, X) and the qstack's in-plane `order`.

    Its own inverse -- the conversion is one transpose or nothing -- so a single function
    serves both directions: `run_basic_camera` turning BaSiC's qstack-ordered field into the
    canonical plane it writes, and `empty_fraction_map` turning the canonical map on disk
    back into the order the qstack it un-mixes is in.
    """
    return np.ascontiguousarray(plane.T) if order == "xy" else plane


def canonical_plane(plane, expected_yx, order, what=""):
    """A stored plane as canonical (Y, X), given the (Y, X) size it should cover.

    The transpose is accepted with a warning only when the shape rules the expected order
    out -- what a plane written by a differently-formatted earlier run looks like. Anything
    that is neither is an error rather than a guess: silently un-mixing or dividing by a
    plane whose axes are swapped corrupts every voxel it touches.
    """
    Y, X = expected_yx
    expected = (Y, X) if order == "yx" else (X, Y)
    if tuple(plane.shape) == expected:
        return plane if order == "yx" else np.ascontiguousarray(plane.T)
    if tuple(plane.shape) == expected[::-1]:
        print(dt.now(), f"WARNING: {what} shape {tuple(plane.shape)} is the transpose of "
                        f"the expected {order} order {expected}; transposing", flush=True)
        return np.ascontiguousarray(plane.T) if order == "yx" else plane
    raise ValueError(f"{what}: plane shape {tuple(plane.shape)} matches neither the "
                     f"expected {order} plane {expected} nor its transpose")


def write_plane_tiff(path, plane):
    """Write one 2-D plane as a single-page float32 TIFF, so readers round-trip it.

    `tifffile` is imported here rather than at module scope: the emptiness stage treats a
    missing one as "no map this run" instead of a failure, and that only works if the
    import raises at call time.
    """
    import tifffile
    tifffile.imwrite(str(path), np.ascontiguousarray(plane, dtype=np.float32))
