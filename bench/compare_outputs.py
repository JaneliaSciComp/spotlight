"""Compare two stores voxel for voxel: `python compare_outputs.py <a> <b> <format> [setups]`.

Arguably the more valuable half of a benchmark run. Timing tells you the port is faster;
this tells you it is still right, on real data, at real scale -- and it costs one extra
job on a run that already read everything.

Exits non-zero on the first mismatch, naming the chunk: "they differ" without a location
is not actionable.
"""

import sys
from pathlib import Path

import numpy as np

from spotlight.formats import _SPEC, _input_location, canonical_view
from spotlight.stores import _context
from spotlight import stores


def _open(root, fmt, setup):
    cfg = {"input_intensity_path": root, "input_format": fmt}
    path, order = _input_location(cfg, setup, 0)
    import tensorstore as ts
    arr = ts.open({"driver": _SPEC[fmt]["driver"],
                   "kvstore": {"driver": "file", "path": path},
                   "context": _context()}).result()
    return canonical_view(arr, order)


def compare(root_a, root_b, fmt, setups, block=(64, 512, 512)):
    bad = 0
    for setup in setups:
        a, b = _open(root_a, fmt, setup), _open(root_b, fmt, setup)
        if a.shape != b.shape:
            print(f"setup {setup}: SHAPE {a.shape} != {b.shape}")
            bad += 1
            continue
        z, y, x = a.shape
        bz, by, bx = block
        for oz in range(0, z, bz):
            for oy in range(0, y, by):
                for ox in range(0, x, bx):
                    sl = (slice(oz, min(oz + bz, z)),
                          slice(oy, min(oy + by, y)),
                          slice(ox, min(ox + bx, x)))
                    va = np.asarray(a[sl].read().result())
                    vb = np.asarray(b[sl].read().result())
                    if not np.array_equal(va, vb):
                        d = np.abs(va.astype(np.int32) - vb.astype(np.int32))
                        print(f"setup {setup}: MISMATCH at z={oz} y={oy} x={ox}: "
                              f"{int((d != 0).sum())} voxels differ, max |delta| {int(d.max())}")
                        bad += 1
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            print(f"setup {setup}: identical")
    return bad


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    a, b, fmt = sys.argv[1:4]
    setups = [int(s) for s in sys.argv[4:]] or [0]
    sys.exit(1 if compare(a, b, fmt, setups) else 0)
