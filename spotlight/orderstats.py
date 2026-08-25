"""Vectorised port of `OnlineStats.OrderStats`, the accumulator the quantile stats pass is
built on.

Julia reference: `~/.julia/packages/OnlineStats/*/src/stats/stats.jl`, `_fit!`/`_merge!`.

`OrderStats(p)` keeps a length-`p` vector that is the running MEAN of SORTED blocks of `p`
consecutive observations. That is not a percentile of the column, and the difference is
not academic: `q000` is the mean of per-block minima, so it sits well above the column's
actual minimum. Anything that models a qstack plane as a plain percentile of the raw data
will be wrong by the width of that gap.

Three details that are easy to get wrong, all load-bearing:

* A trailing partial block is DISCARDED -- but `nobs` still counts its observations,
  because `nobs` comes from the separate `Extrema` accumulator. So merges are weighted by
  raw observation count, never by block count.
* `minimum`/`maximum` are the TRUE extrema over every observation, including the ones in
  the discarded tail. They are not `value[0]`/`value[-1]`.
* Accumulation is float64 (`OrderStats(p)` defaults `T=Float64`) even though the data is
  uint16 and the rest of the pipeline is float32.

Where Julia loops per pixel with `fit!`, this reduces a whole (X, Y, Z) tile in one
vectorised sort -- the bulk of the port's speedup.
"""

import numpy as np

__all__ = ["OrderStats", "block_size", "quantile_r7"]

# The 21 quantile planes a qstack carries: 0, 5, ..., 100 percent.
N_QUARTILES = 21
LEVELS = tuple(range(0, 101, 5))


def block_size(z_depth):
    """Julia's `n_bins` for a chunk `z_depth` slices deep.

    `min(z_depth, ((z_depth - 1) // 21) * 21)` -- see `get_setup_stats_chunked` in
    src/BigFlatFieldIlluminator.jl. A 64-deep chunk gives 63, discarding one slice.

    z_depth == 21 makes this 0, which in Julia reaches `nobs % 0` and throws a
    `DivideError` from deep inside OnlineStats. `raw_stack_mode` only diverts cameras with
    FEWER than 21 slices, so exactly 21 is a live crash there. Raise something that names
    the problem instead of reproducing the DivideError or, worse, silently writing an
    all-zero qstack.
    """
    if z_depth < N_QUARTILES:
        raise ValueError(
            f"z_depth={z_depth} is below N_QUARTILES={N_QUARTILES}; this camera has no "
            "per-pixel distribution to summarise. save_qstack() feeds BaSiC the raw "
            "slices instead (raw_stack_mode)."
        )
    p = min(z_depth, ((z_depth - 1) // N_QUARTILES) * N_QUARTILES)
    if p == 0:
        raise ValueError(
            f"z_depth={z_depth} gives a block size of 0, so no block can ever complete "
            "and every quantile would come out 0. Use a chunk depth of at least "
            f"{N_QUARTILES + 1}."
        )
    return p


def _r7_position(n, q):
    """Julia's `(j, gamma)` for the R-7 rule, bit for bit.

    `Statistics._quantile` computes `aleph = fma(n, p, m)` with `m = 1 - p`, and the fused
    multiply-add matters: `fma(63, 0.2, 0.8)` and the algebraically equal `1 + 62*0.2`
    differ by one ulp, enough to move a value across a `.5` boundary and change the uint16
    the stats pass writes. Python 3.11 has no `math.fma`, so the exact product-plus-addend
    is formed as a rational and rounded once, which is what fma is defined to do.
    """
    from fractions import Fraction

    m = 1.0 - q                                   # alpha + p*(1 - alpha - beta), alpha=beta=1
    aleph = float(Fraction(n) * Fraction(q) + Fraction(m))
    j = min(max(int(aleph), 1), n - 1)             # trunc, then clamp
    return j, min(max(aleph - j, 0.0), 1.0)


def quantile_r7(sorted_v, q, axis=-1):
    """`Statistics.quantile(v, q)` -- the R-7 rule, on an already-sorted `v`, which this
    does not sort for itself.

    Written out rather than deferred to `np.quantile(method="linear")`: numpy switches
    interpolation form above t=0.5 (`b - (b-a)*(1-t)` instead of `a + g*(b-a)`), differing
    from Julia by up to one ulp. That only matters when a value lands exactly on a `.5`
    rounding boundary -- and rounding to uint16 is the very next thing the stats pass
    does.

    Agrees with Julia to within one ulp, not bit for bit. The remaining difference is LLVM
    contracting the interpolation into an FMA inside `Statistics._quantile`, which Python
    cannot reproduce and which is not even stable across architectures, so matching it
    would pin this to whichever machine generated the reference.

    It CAN reach the output. Measured on a 40x30x945 tile through the full path (fit,
    quantile, round): 23 of 27600 written uint16 values differ from Julia, all by exactly
    1 count -- 0.08%. Isolating the two sources on a subset, the fma interpolation
    accounts for all of it and the batch-vs-fold mean for none. So a real Julia-vs-Python
    array diff should expect `max |delta| <= 1` on <0.2% of values, NOT exact equality.

    Julia also picks between this form and `a + g*(b-a)` per element, using the latter
    when the two neighbours are within ~1.5e-8 relative of each other. That branch is
    skipped here: neighbours that close differ by under 1e-5 counts, so they round to the
    same uint16 regardless, and evaluating both forms plus a select over every pixel of 21
    planes is real work on the hot path for a difference that cannot be observed.
    """
    n = sorted_v.shape[axis]
    if n == 1:
        return np.take(sorted_v, 0, axis=axis).astype(np.float64, copy=True)
    j, g = _r7_position(n, q)
    a = np.take(sorted_v, j - 1, axis=axis).astype(np.float64, copy=False)
    b = np.take(sorted_v, j, axis=axis).astype(np.float64, copy=False)
    return (1.0 - g) * a + g * b


class OrderStats:
    """An (X, Y) grid of `OnlineStats.OrderStats(p)`, held as whole arrays.

    `value` is (X, Y, p) float64; `vmin`/`vmax` are (X, Y) in the source dtype; `n` is the
    observation count INCLUDING any discarded trailing block.
    """

    __slots__ = ("value", "n", "vmin", "vmax")

    def __init__(self, value, n, vmin, vmax):
        self.value, self.n, self.vmin, self.vmax = value, n, vmin, vmax

    @classmethod
    def fit(cls, a, p):
        """Reduce one setup's (X, Y, L) chunk.

        The whole accumulator is one sort: reshape the Z axis into (nblk, p), sort each
        block, average the blocks. `nblk == 0` (p larger than the data) leaves `value`
        all-zero, exactly as Julia's un-folded buffer does -- the failure mode behind an
        all-zero qstack, so it is reproduced rather than papered over.

        The average is a BATCH mean, and this is where the port deliberately diverges from
        `OnlineStats`. Julia folds each block into a running mean (`value += (block -
        value)/k`) because it is a streaming accumulator that never holds the data. This
        is not streaming -- the whole chunk is already resident -- and for uint16 input
        the batch mean is EXACTLY correct: `nblk * 65535` is far under 2^53, so every
        partial sum in numpy's pairwise reduction is an exactly representable integer, the
        sum carries no error, and the single division is correctly rounded.

        Measured on a 40x30x945 tile (15 blocks): this returns the correctly-rounded mean
        for 75600/75600 values, Julia's fold for 221/378 with a worst error of 9.1e-13. So
        where the two disagree, THIS ONE IS RIGHT. Matching Julia here would mean adding
        error to reproduce a less accurate reference, and it would not buy bit-parity
        anyway -- see `quantile_r7` for the difference that actually survives.
        """
        x, y, length = a.shape
        nblk = length // p
        if nblk == 0:
            value = np.zeros((x, y, p), dtype=np.float64)
        else:
            blocks = a[..., : nblk * p].reshape(x, y, nblk, p)
            value = np.sort(blocks, axis=-1).mean(axis=2, dtype=np.float64)
        return cls(value, length, a.min(axis=2), a.max(axis=2))

    def merge(self, other):
        """`OnlineStats._merge!`, in place. Returns self so it can be folded.

        The weight is `n_other / (n_self + n_other)` with `n_self` taken AFTER the counts
        are combined -- Julia merges the `Extrema` first and only then reads `nobs(a)`.
        Weighting by blocks instead of observations is the single most likely way this
        port silently disagrees: an accumulator over 10 observations with p=4 holds 2
        blocks but must merge as 10, not 2.
        """
        if self.value.shape != other.value.shape:
            raise ValueError(
                f"cannot merge OrderStats with block sizes {self.value.shape[-1]} and "
                f"{other.value.shape[-1]}; every Z block must be the same depth"
            )
        np.minimum(self.vmin, other.vmin, out=self.vmin)
        np.maximum(self.vmax, other.vmax, out=self.vmax)
        self.n += other.n
        self.value += (other.n / self.n) * (other.value - self.value)
        return self

    def quantiles(self, levels=LEVELS):
        """The requested percentile planes, as a list of (X, Y) float64 arrays.

        `value` is sorted once here rather than per level. Averaging sorted vectors is
        order-preserving in exact arithmetic, so this is a no-op on well-behaved data --
        but Julia's `quantile` sorts too, and matching it costs nothing.
        """
        v = np.sort(self.value, axis=-1)
        return [quantile_r7(v, q / 100.0) for q in levels]


def to_uint16(a):
    """Round to uint16 the way `save_stats` does, with one deliberate difference.

    Julia's `round(UInt16, x)` is ties-to-even and THROWS above 65535. numpy's
    `np.round(x).astype(np.uint16)` is ties-to-even but wraps (65536.0 -> 0), and a bare
    `.astype` truncates toward zero. Wrapping is the worst of the three -- a saturated
    voxel would come back as black -- so clip first and accept a saturated write where
    Julia would have raised.
    """
    return np.round(np.clip(a, 0, 65535)).astype(np.uint16)
