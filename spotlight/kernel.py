"""The per-shard correction kernel: pure numpy, no I/O, no asyncio.

One unit on purpose. `_BLOCK_VOXELS` sits under the measurements that chose it, and
`_blocks`, `_scratch`, `_win`, `ShardCorrection` and `_correct_shard` are the code those
measurements describe. A tuning constant separated from the benchmark that set it is worse
than no constant at all -- the next person re-tunes it blind.

Used by `correct.py` for all three correction modes. Composing flat/dark and per-tile
intensity into one affine pass is what lets the data be read once and written once.
"""

import os
import threading

import numpy as np


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
_BLOCK_VOXELS = int(os.getenv("SPOTLIGHT_BLOCK_KIB", "1024")) * 1024 // 4

_SCRATCH = threading.local()


def _scratch(shape):
    """Two per-thread float32 scratch buffers shaped like `shape`, reused across blocks
    and shards -- a pool thread corrects thousands of blocks, so reusing the pages beats
    re-faulting them every time. Kept flat and reshaped, so a short trailing block borrows
    the same allocation and still gets a contiguous view.
    """
    n = int(np.prod(shape))
    bufs = getattr(_SCRATCH, "bufs", None)
    if bufs is None or bufs[0].size < n:
        bufs = (np.empty(n, "float32"), np.empty(n, "float32"))
        _SCRATCH.bufs = bufs
    return bufs[0][:n].reshape(shape), bufs[1][:n].reshape(shape)


def _blocks(zyx):
    """Contiguous (z_start, z_stop, y_start, y_stop) blocks of ~`_BLOCK_VOXELS` covering a
    canonical (Z, Y, X) array. x is always taken whole, so every block is a contiguous
    span of the underlying buffer -- and the (Y, X) coefficient planes slice to it
    contiguously too.

    Whole Z-planes when a plane fits the budget, otherwise row-blocks of a single plane:
    for a 2560x4096 plane that is 64 rows at a time rather than one 40 MiB plane.
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
    """A coefficient plane restricted to a shard's (y, x) window. Scalars and None pass
    through.
    """
    if plane is None or np.isscalar(plane):
        return plane
    return plane[y_range[0]:y_range[1], x_range[0]:x_range[1]]


class ShardCorrection:
    """Everything one setup's shards need, precomputed once per apply job.

    `mode` is the intensity strategy from `_classify`: "bimodal" (rescale the foreground
    only), "uniform" (rescale every voxel), or "none" (an empty tile -- no rescale, but
    still flat/dark-corrected when joint).

    With a `BasicModel`, the flat/dark correction `(raw - dark) / flat` is rewritten as
    the per-pixel affine `raw * basic_a + basic_b` with `basic_a = 1/flat`, `basic_b =
    -dark/flat`, computed once here. For "uniform" the intensity rescale folds into the
    SAME two planes:

        max(raw*a + b, 0) * S + K  ==  max(raw*(a*S) + (b*S + K), K)

    (valid because S > 0), so that whole branch is one multiply-add and a scalar maximum.
    `S = scale_i`, `K = target_mean - mean_i*scale_i`.

    Folding trades exactness for speed at the last bit: `raw*(1/flat)` and `(raw -
    dark)/flat` differ by an ULP or so, flipping a tiny fraction of voxels by one gray
    level -- less than the intermediate-uint16 rounding the joint route already removes,
    so it is only accepted where joint correction is active. WITHOUT a basic model this
    class keeps the original operation order (`(c - mean_i)*scale_i + target_mean`)
    exactly, so an intensity-only run still produces bit-identical output to previous
    releases.
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

    Pure numpy -- no I/O, no asyncio. Run via ThreadPoolExecutor, not a process pool:
    numpy's C loops release the GIL for arrays this size, so threads get real multi-core
    overlap without paying to pickle a multi-hundred-MB shard across a process boundary.

    Every branch ends with the same tail -- clip to the uint16 range, round half-to-even
    (`np.rint`, matching Julia's `round(UInt16, x)`), cast back into `canon` -- so the
    data is quantized exactly ONCE no matter how many corrections were composed on the
    way.

    `canon` must be C-contiguous for `_blocks` to hand out contiguous work, which
    `canonical_view` guarantees for every caller here. Anything else is corrected via a
    C-ordered copy rather than silently running strided, which measured ~7x slower on an
    xyz-stored source.
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
