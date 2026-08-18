"""Build small synthetic stores, so the I/O paths get exercised and not just the math.

Two deliberate choices:

* The frame is never square (X=64, Y=48). A transposed read then fails on SHAPE rather
  than passing quietly, which is the cheapest possible insurance against the single most
  likely porting bug.
* Voxel values encode their own coordinates, so an axis SWAP that happens to preserve the
  shape still fails on value.

The stores are written through `open_output_array` / `write_group_metadata` --
the same writers the pipeline uses -- so the layouts under test are the real ones rather
than a test-only approximation of them.
"""

from pathlib import Path

import numpy as np

from spotlight.formats import _SPEC, _in_order, canonical_view
from spotlight.stores import _context, open_output_array, write_group_metadata


X, Y = 64, 48
FORMATS = ("n5", "zarr2", "zarr3", "zarr3_unsharded")

# setup 3 is deeper, to exercise the "setups differ in Z" path that `camera_source_size`
# exists for.
DEPTHS = {0: 63, 1: 63, 2: 63, 3: 84}


def volume(setup, z, y=Y, x=X):
    """(Z, Y, X) uint16 with coordinate-encoding values, offset per setup."""
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    return ((1009 * xx + 101 * yy + 7 * zz + 37 * setup) % 4096).astype(np.uint16)


def write_store(root, fmt, setups=(0, 1, 2, 3), depths=None):
    """Write one store and return a config fragment that reads it back."""
    root = Path(root)
    depths = DEPTHS if depths is None else depths
    cfg = {
        "output_intensity_path": str(root),
        "output_format": fmt,
        # [X, Y, Z], as the toml stores it. Z is 32 rather than something smaller so the
        # block size stays above N_QUARTILES -- at 16 there is no distribution to
        # summarise and the pass correctly refuses to run.
        "chunk_size": [32, 32, 32],
        "shard_size": [64, 64, 64],
    }
    ctx = _context()
    for setup in setups:
        z = depths[setup]
        vol = volume(setup, z)
        order = _SPEC[fmt]["order"]
        shape = _in_order((z, Y, X), order)
        arr, _, _ = open_output_array(cfg, setup, 0, shape, "uint16", ctx)
        canonical_view(arr, order)[:, :, :].write(vol).result()
        write_group_metadata(cfg, setup, [(1, 1, 1)])
    return {
        "input_basic_path": str(root),
        "output_basic_path": str(root) + "_out",
        "input_format": fmt,
        "output_format": fmt,
        "chunk_size": [32, 32, 32],
        "shard_size": [64, 64, 64],
        "z_batch": 1,
        "stats_scale": 0,
        "basic_stats_level": 0,
        "last_setup": max(setups),
        "setups_per_camera": len(setups),
        "setup_ids": [],
        "qstacks_dir": str(root.parent / "qstacks"),
        "results_root": str(root.parent / "results"),
        "basic_unmix_empty": False,
        "n_cores_stats": 2,
        "apply_basic": True,
        "input_intensity_path": str(root),
        "output_intensity_path": str(root) + "_out",
    }
