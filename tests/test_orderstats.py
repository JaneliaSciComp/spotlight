"""Parity tests for the order-statistics accumulator.

These are the highest-value tests in the package: everything downstream -- the qstack,
the BaSiC fit, the correction -- is built on `OrderStats`, and its semantics are subtle
enough that a wrong port produces plausible numbers rather than an error.

Reference values come from running the actual Julia `OnlineStats` (see `gen_golden.jl`).
Most are asserted EXACTLY -- the semantics (block averaging, discarded tails, merge
weighting, extrema) are shared, and any difference there is a porting bug.

Two things are deliberately NOT bit-identical, and the last three tests pin both rather
than leaving them to assumption:

* The block average. Julia folds each block into a running mean because `OnlineStats` is
  a streaming accumulator; this port sums, which for uint16 input is exactly rounded.
  Where they differ, this port is the correct one.
* The R-7 interpolation position, which Julia computes with `fma`.

Measured together on a 40x30x945 tile, they move 23 of 27600 written uint16 values, all
by exactly 1 count.
"""

import numpy as np
import pytest

from spotlight.orderstats import (
    N_QUARTILES, OrderStats, block_size, quantile_r7, to_uint16,
)

from golden_io import have_golden, load_bin, load_json

pytestmark = pytest.mark.skipif(
    not have_golden(),
    reason="run tests/gen_golden.jl (needs Julia) to produce the reference values",
)


def _fit1(ys, p):
    """One accumulator over a 1-D sample, shaped as the (1, 1, L) tile the port expects."""
    a = np.asarray(ys, dtype=np.float64).reshape(1, 1, -1)
    return OrderStats.fit(a, p)


def _check(name, os_):
    g = load_json("orderstats.json")[name]
    assert os_.value.shape == (1, 1, g["p"]), name
    np.testing.assert_array_equal(os_.value[0, 0], np.asarray(g["value"]), err_msg=name)
    assert os_.n == g["n"], f"{name}: nobs"
    assert float(os_.vmin[0, 0]) == g["min"], f"{name}: minimum"
    assert float(os_.vmax[0, 0]) == g["max"], f"{name}: maximum"


def _case(name):
    return load_json("orderstats.json")[name]


def test_clean_multiple():
    """Three complete blocks: the plain running mean of sorted blocks."""
    g = _case("clean_multiple")
    _check("clean_multiple", _fit1(g["ys"], g["p"]))


def test_trailing_partial_block_is_discarded():
    """The tail past the last complete block does not enter `value` -- but DOES count
    towards `nobs`, which is what every merge is weighted by."""
    g = _case("trailing_partial")
    os_ = _fit1(g["ys"], g["p"])
    _check("trailing_partial", os_)
    assert os_.n == 14 and os_.value.shape[-1] == 4
    # Same `value` as the clean case, since the extra two observations are dropped.
    clean = _fit1(_case("clean_multiple")["ys"], 4)
    np.testing.assert_array_equal(os_.value, clean.value)


def test_p_exceeds_data_leaves_value_zero():
    """No block ever completes, so every quantile reads 0 while the extrema are real.

    This is the all-zero-qstack failure mode. It is reproduced, not fixed, so a
    misconfigured chunk depth looks the same in both implementations.
    """
    g = _case("p_exceeds_data")
    os_ = _fit1(g["ys"], g["p"])
    _check("p_exceeds_data", os_)
    assert not os_.value.any()
    assert os_.quantiles([50])[0][0, 0] == 0.0


def test_extrema_come_from_the_discarded_tail():
    """minima/maxima are true extrema, not `value[0]`/`value[-1]`."""
    g = _case("extrema_in_tail")
    os_ = _fit1(g["ys"], g["p"])
    _check("extrema_in_tail", os_)
    assert float(os_.vmin[0, 0]) == 1 and float(os_.vmax[0, 0]) == 99
    assert os_.value[0, 0, 0] == 5 and os_.value[0, 0, -1] == 8


@pytest.mark.parametrize("name", ["merge_equal", "merge_unequal", "merge_into_empty"])
def test_merge(name):
    g = _case(name)
    if g["ys"]:
        a = _fit1(g["ys"], g["p"])
    else:
        # A fresh `Extrema` starts at (+Inf, -Inf), not (0, 0), so that the first merge
        # takes the other side's extrema rather than being pinned to zero.
        a = OrderStats(np.zeros((1, 1, g["p"])), 0,
                       np.full((1, 1), np.inf), np.full((1, 1), -np.inf))
    _check(name, a.merge(_fit1(g["ys_b"], g["p"])))


def test_merge_weights_observations_not_blocks():
    """`a` holds 2 complete blocks from 10 observations; `b` holds 2 from 8.

    The weight is 8/18, not 2/4. Weighting by blocks is the likeliest way this port
    silently disagrees with Julia, and it never raises.
    """
    g = _case("merge_partial_block")
    a = _fit1(g["ys"], g["p"]).merge(_fit1(g["ys_b"], g["p"]))
    _check("merge_partial_block", a)
    assert a.n == 18


def test_merge_is_a_left_fold():
    """Float merging is not associative; the stats pass folds strictly in setup order."""
    g = _case("fold_left")
    a, b, c = (_fit1(g[k], g["p"]) for k in ("ys", "ys_b", "ys_c"))
    _check("fold_left", a.merge(b).merge(c))

    r = _case("fold_right")
    a2, b2, c2 = (_fit1(r[k], r["p"]) for k in ("ys", "ys_b", "ys_c"))
    right = a2.merge(b2.merge(c2))
    _check("fold_right", right)
    # The two orders really do differ -- otherwise this test proves nothing.
    assert not np.array_equal(np.asarray(g["value"]), np.asarray(r["value"]))


def test_non_integral_input():
    g = _case("non_integral")
    _check("non_integral", _fit1(g["ys"], g["p"]))


def test_merging_different_block_sizes_raises():
    """Julia warns and silently declines to merge; that is worse than failing, because
    the result looks like a merge that happened."""
    a, b = _fit1(range(8), 4), _fit1(range(8), 2)
    with pytest.raises(ValueError, match="block sizes"):
        a.merge(b)


# ─── quantiles and rounding ───────────────────────────────────────────────────


def test_r7_quantiles_match_julia():
    """The written uint16 must match exactly; the float64 intermediate to within an ulp.

    Bit-exactness on the float is not achievable and not worth chasing: LLVM contracts the
    interpolation inside `Statistics._quantile` into an FMA, which differs by an ulp and
    is not stable across architectures. What the pipeline writes is the rounded value, and
    that is asserted exactly.
    """
    g = load_json("quantiles_r7.json")
    v = np.asarray(g["value"]).reshape(1, 1, -1)
    for level, expected, expected_u16 in zip(g["levels"], g["quantiles"], g["rounded"]):
        got = quantile_r7(v, level / 100.0)[0, 0]
        assert abs(got - expected) <= 4 * np.spacing(abs(expected)), \
            f"q{level:03d}: {got!r} != {expected!r}"
        assert int(to_uint16(np.array(got))) == expected_u16, f"q{level:03d} rounded"


def test_rounding_is_ties_to_even():
    g = load_json("round_ties.json")
    np.testing.assert_array_equal(to_uint16(np.asarray(g["inputs"])),
                                  np.asarray(g["rounded"], dtype=np.uint16))


def test_to_uint16_clips_rather_than_wrapping():
    """A bare `.astype(np.uint16)` would wrap 65536 to 0 -- a saturated voxel coming back
    black is the worst available failure, so clip instead."""
    assert to_uint16(np.array([70000.0, -5.0]))[0] == 65535
    assert to_uint16(np.array([70000.0, -5.0]))[1] == 0


# ─── the whole reduction ──────────────────────────────────────────────────────


def test_block_size():
    assert block_size(64) == 63
    assert block_size(63) == 42
    assert block_size(256) == 252
    with pytest.raises(ValueError):
        block_size(20)          # below N_QUARTILES: raw_stack_mode territory
    with pytest.raises(ValueError):
        block_size(21)          # gives p == 0, a live DivideError in Julia


def test_full_tile_matches_julia():
    """A 5x3x63 tile through the exact reduction the stats pass runs.

    Non-square on purpose: a transposed port fails on shape here rather than passing
    quietly on a square test case.
    """
    g = load_json("tile_stats.json")
    a = load_bin("tile_input")
    assert a.shape == tuple(g["shape"])
    st = OrderStats.fit(a, g["p"])

    np.testing.assert_array_equal(to_uint16(st.vmin).ravel(order="F"),
                                  np.asarray(g["minima"], dtype=np.uint16))
    np.testing.assert_array_equal(to_uint16(st.vmax).ravel(order="F"),
                                  np.asarray(g["maxima"], dtype=np.uint16))
    for level, expected in zip(g["levels"], g["quantiles"]):
        got = to_uint16(quantile_r7(np.sort(st.value, axis=-1), level / 100.0))
        np.testing.assert_array_equal(got.ravel(order="F"),
                                      np.asarray(expected, dtype=np.uint16),
                                      err_msg=f"q{level:03d}")


def test_quantiles_returns_all_levels():
    a = load_bin("tile_input")
    st = OrderStats.fit(a, load_json("tile_stats.json")["p"])
    assert len(st.quantiles()) == N_QUARTILES


# ─── the two places the port deliberately differs from Julia ──────────────────


def test_fit_is_the_exactly_rounded_mean():
    """The strongest statement available, and it needs no Julia reference at all.

    For uint16 input the batch mean is not merely close to the true mean, it IS the
    correctly-rounded one: `nblk * 65535` is far under 2^53, so every partial sum in
    numpy's pairwise reduction is an exactly representable integer and the single division
    is correctly rounded. Better than asserting agreement with Julia, whose running fold
    accumulates ~1e-13 per value -- see `test_julia_fold_is_less_accurate_than_this_port`.
    """
    from fractions import Fraction

    a = load_bin("divergence_input")
    g = load_json("divergence_stats.json")
    p, nblk = g["p"], g["nblk"]
    x, y, _ = a.shape
    st = OrderStats.fit(a, p)

    blocks = np.sort(a.reshape(x, y, nblk, p), axis=-1).astype(np.int64)
    exact = np.array([[[float(Fraction(int(blocks[i, j, :, k].sum()), nblk))
                        for k in range(p)] for j in range(y)] for i in range(x)])
    np.testing.assert_array_equal(st.value, exact)


def test_julia_fold_is_less_accurate_than_this_port():
    """Pins the DIRECTION of the disagreement, so nobody "fixes" it the wrong way.

    Julia folds each block into a running mean because `OnlineStats` is a streaming
    accumulator that never holds the data; this port has the whole chunk resident and sums
    instead. Both are correct implementations of the same statistic; only one is exactly
    rounded.
    """
    from fractions import Fraction

    a = load_bin("divergence_input")
    g = load_json("divergence_stats.json")
    p, nblk = g["p"], g["nblk"]
    julia = np.array(load_json("divergence_value.json")["value"])
    idx = [(x, y) for y in range(2) for x in range(3)]        # the pixels Julia dumped

    st = OrderStats.fit(a, p)
    blocks = np.sort(a.reshape(a.shape[0], a.shape[1], nblk, p), axis=-1).astype(np.int64)
    exact = np.array([[float(Fraction(int(blocks[x, y, :, k].sum()), nblk))
                       for k in range(p)] for (x, y) in idx])
    mine = np.array([st.value[x, y] for (x, y) in idx])

    assert (mine == exact).all(), "this port should be exactly rounded"
    assert not (julia == exact).all(), (
        "Julia's fold is expected to be inexact here; if it now matches exactly, this "
        "test's premise changed and the docstrings in orderstats.py need revisiting")
    assert np.abs(julia - exact).max() < 1e-11


def test_written_uint16_matches_julia_within_one_count():
    """The tolerance a real Julia-vs-Python array diff should use, measured not assumed.

    Two effects survive into the written values: the batch-vs-fold mean above, and Julia's
    `fma` in the R-7 interpolation position, which Python cannot reproduce. Measured here:
    23/27600 differ, always by exactly 1. Isolating them on a subset, the fma accounts for
    all of it and the mean for none -- so matching Julia's fold would buy nothing.

    The bounds are loose enough not to be brittle and tight enough to catch a real porting
    bug, which would move far more than 0.2% of values or more than 1 count.
    """
    a = load_bin("divergence_input")
    g = load_json("divergence_stats.json")
    x, y, _ = a.shape
    st = OrderStats.fit(a, g["p"])
    v = np.sort(st.value, axis=-1)

    differing = total = 0
    for level, julia in zip(g["levels"], g["quantiles"]):
        mine = to_uint16(quantile_r7(v, level / 100.0))
        theirs = np.asarray(julia, dtype=np.uint16).reshape(y, x).T
        d = np.abs(mine.astype(np.int32) - theirs.astype(np.int32))
        assert d.max() <= 1, f"q{level:03d} differs by {d.max()} counts, not 1"
        differing += int((d != 0).sum())
        total += d.size

    # Extrema are integer comparisons -- no interpolation, so these must be EXACT.
    for name, mine in (("minima", to_uint16(st.vmin)), ("maxima", to_uint16(st.vmax))):
        theirs = np.asarray(g[name], dtype=np.uint16).reshape(y, x).T
        np.testing.assert_array_equal(mine, theirs, err_msg=name)

    rate = differing / total
    assert rate < 0.002, f"{rate:.3%} of quantiles differ, expected <0.2%"
