"""The overlap gate, and the config keys that expose it.

The gate is what turns "0/2 usable overlap constraints" into either a solved gain or
unit gains, so the interesting cases are the boundary and the override -- and the
message, because a rejection that does not say by how much it missed cannot be acted on.
"""

import numpy as np
import pytest

from spotlight import aggregate, config, tilestats


def test_config_defaults_match_the_gate_constants():
    """`config.DEFAULTS` restates these as literals to avoid an import cycle; if
    `tilestats` changes and this does not, the documented default becomes a lie.
    """
    assert config.DEFAULTS["min_overlap_foreground"] == tilestats.MIN_FOREGROUND
    assert config.DEFAULTS["min_overlap_fraction"] == tilestats.MIN_FG_FRACTION


def test_every_aggregate_knob_is_in_the_config_list():
    """The point of listing them: `load_config()` is how you find out what exists."""
    for key in ("gain_grouping", "gain_estimator", "gain_lambda",
                "min_overlap_foreground", "min_overlap_fraction"):
        assert key in config.DEFAULTS, f"{key} is read by aggregate but undiscoverable"


# ─── the gate itself ──────────────────────────────────────────────────────────


def _cache(vol_a, vol_b, thr):
    """Two tiles that fully overlap, so `_overlap_ranges` is not what is under test."""
    class Arr:
        def __init__(self, v):
            self.v = v

        def __getitem__(self, k):
            return self

        def read(self, order="C"):
            class F:
                def __init__(self, v):
                    self.v = v

                def result(self):
                    return self.v
            return F(self.v)

    shape = vol_a.shape
    return {0: {"arr": Arr(vol_a), "thr": thr, "basic": None, "factor": (1, 1, 1),
                "shape": shape},
            1: {"arr": Arr(vol_b), "thr": thr, "basic": None, "factor": (1, 1, 1),
                "shape": shape}}


def _sparse(n_fg, gain=1.0, shape=(8, 32, 32)):
    """A volume that is background except for `n_fg` bright voxels -- the shroff_worm
    case, where foreground is a fraction of a percent of the overlap."""
    v = np.full(shape, 100.0, dtype=np.float32)
    flat = v.reshape(-1)
    flat[:n_fg] = 1000.0 * gain
    return v


def _full_overlap(setup, world_bbox, sizes, transforms, factor, shape):
    """Stub the geometry: these tests are about the GATE, and the world-bbox-to-pixel
    transform has its own tests. Two tiles that overlap completely.
    """
    return [(0, shape[0]), (0, shape[1]), (0, shape[2])]


def _pair(cache, estimator="intersection", reject=None, **kw):
    real, aggregate._overlap_ranges = aggregate._overlap_ranges, _full_overlap
    try:
        return aggregate._pair_gain_constraint(0, 1, None, None, None, cache, "zyx",
                                              estimator=estimator, reject=reject, **kw)
    finally:
        aggregate._overlap_ranges = real


def _constraint(n_fg, gain=1.0, reject=None, **kw):
    a, b = _sparse(n_fg), _sparse(n_fg, gain)
    return _pair(_cache(a, b, thr=500.0), reject=reject, **kw)


def test_sparse_overlap_is_rejected_by_default():
    assert _constraint(n_fg=50) is None


def test_lowering_the_floor_admits_it():
    """The override exists for exactly this: real but scant shared tissue."""
    got = _constraint(n_fg=50, gain=2.0, min_fg=16, min_frac=0.0)
    assert got is not None
    a, b, log_ratio, weight = got
    # The gain is recovered from the medians, not invented by the relaxed gate.
    assert log_ratio == pytest.approx(np.log(0.5), rel=1e-4)
    assert (a, b, weight) == (0, 1, 50)


def test_the_absolute_floor_and_the_fraction_are_both_enforced():
    """A big overlap with plenty of absolute foreground can still be unrepresentative,
    and a small overlap that is mostly foreground can still have too few samples."""
    # Passes the fraction (all foreground) but not the count.
    assert _constraint(n_fg=10, min_frac=0.0) is None
    # Passes the count but not the fraction.
    assert _constraint(n_fg=300, min_fg=16, min_frac=0.9) is None


def test_rejection_names_the_shortfall_not_just_the_fact():
    """"0/2 usable" cannot tell you whether to lower a threshold or give up."""
    reject = []
    _constraint(n_fg=50, reject=reject)
    assert len(reject) == 1
    msg = reject[0]
    assert "50" in msg and "min_overlap_foreground" in msg
    assert "estimator intersection" in msg


def test_independent_estimator_sees_more_foreground_than_intersection():
    """Why `independent` is the first thing to try on sparse data: `intersection` counts
    only voxels BOTH tiles call foreground, and misregistered sparse tissue has few."""
    a, b = _sparse(200), _sparse(200)
    # Shift b's foreground so the two sets barely intersect.
    b.reshape(-1)[:200] = 100.0
    b.reshape(-1)[400:600] = 1000.0
    cache = _cache(a, b, thr=500.0)
    kw = dict(min_fg=16, min_frac=0.0)
    assert _pair(cache, "intersection", **kw) is None
    got = _pair(cache, "independent", **kw)
    assert got is not None and got[3] == 200


# ─── the gain_floor override ──────────────────────────────────────────────────


def test_gain_floor_defaults_to_the_per_tile_threshold():
    a, b = _sparse(3000), _sparse(3000, gain=2.0)
    assert _pair(_cache(a, b, thr=1e5), min_fg=256, min_frac=0.0) is None


def test_a_numeric_gain_floor_replaces_the_per_tile_threshold():
    """The rescue case: the per-tile threshold is far above anything in the overlap.
    `_tile_floor` resolves the number for every tile, so the pair's max() is that
    number."""
    a, b = _sparse(3000), _sparse(3000, gain=2.0)
    got = _pair(_cache(a, b, thr=500.0), min_fg=256, min_frac=0.0)
    assert got is not None
    assert got[2] == pytest.approx(np.log(0.5), rel=1e-4)


def test_the_numeric_floor_stays_common_to_both_tiles():
    """The property the whole gate rests on: gating each tile at its OWN threshold
    compares different populations of the same tissue and invents a gain."""
    st = {"threshold": 1e5, "thresholds": {"otsu": 1e5, "li": 500.0}}
    assert aggregate._tile_floor(st, 500.0, 0) == aggregate._tile_floor(st, 500.0, 1)
    a, b = _sparse(3000), _sparse(3000, gain=1.0)
    got = _pair(_cache(a, b, thr=500.0), min_fg=256, min_frac=0.0)
    assert got is not None and got[2] == pytest.approx(0.0, rel=1e-9)


def test_it_is_in_the_config_list_with_tile_as_the_default():
    assert config.DEFAULTS["gain_floor"] == "tile"


def test_gain_floor_accepts_tile_a_method_or_a_number():
    assert aggregate._floor_setting({}) == "tile"
    for m in ("tile", "otsu", "li"):
        assert aggregate._floor_setting({"gain_floor": m}) == m
    assert aggregate._floor_setting({"gain_floor": 250}) == 250.0
    with pytest.raises(ValueError, match="gain_floor"):
        aggregate._floor_setting({"gain_floor": "triangle"})


def test_gain_floor_can_name_a_method_independently_of_tile_threshold():
    """The combination this exists for: `tile_threshold = 0` puts the apply stage in its
    all-pixels `uniform` mode while the gain solve still gates on tissue. The two settings
    answer different questions and must not be forced to agree.
    """
    st = {"threshold": 0.0, "thresholds": {"otsu": 1621.0, "li": 274.0}}
    assert aggregate._tile_floor(st, "tile", 0) == 0.0        # follows tile_threshold
    assert aggregate._tile_floor(st, "li", 0) == 274.0        # ...or does not
    assert aggregate._tile_floor(st, "otsu", 0) == 1621.0
    assert aggregate._tile_floor(st, 300.0, 0) == 300.0


def test_a_stats_file_predating_the_catalogue_says_what_to_re_run():
    """Old stats files have `threshold` but no per-method record, so naming a method has
    no answer -- and guessing one would silently gate at the wrong level.
    """
    old = {"threshold": 1621.0}
    assert aggregate._tile_floor(old, "tile", 7) == 1621.0     # still fine
    with pytest.raises(RuntimeError, match="re-run the stats stage"):
        aggregate._tile_floor(old, "li", 7)


def test_the_header_reports_the_floors_actually_used():
    """The bug this fixes: a run with tile_threshold="li" printed `floor=otsu`, because
    the header echoed the SETTING rather than the values in play.
    """
    cache = {0: {"thr": 274.0}, 1: {"thr": 107.0}}
    label = aggregate._floor_label("tile", cache, [0, 1])
    assert "107-274" in label and "otsu" not in label
    # A numeric setting resolves to the same value for every tile via `_tile_floor`, so
    # the cache it labels holds that one number -- not a range.
    same = {0: {"thr": 250.0}, 1: {"thr": 250.0}}
    numeric = aggregate._floor_label(250.0, same, [0, 1])
    assert "250" in numeric and "250-250" not in numeric


def test_floor_sensitivity_is_reported_for_a_ratio_that_moves():
    """A ratio that changes with intensity is a distribution difference, not a gain, and
    the gate cannot tell them apart -- so it has to be said out loud."""
    shape = (8, 64, 64)
    a = np.full(shape, 100.0, np.float32)
    b = np.full(shape, 100.0, np.float32)
    a.reshape(-1)[:4000] = 600.0
    a.reshape(-1)[4000:8000] = 4000.0
    b.reshape(-1)[:4000] = 200.0
    b.reshape(-1)[4000:8000] = 3800.0
    accept = []
    got = _pair(_cache(a, b, thr=150.0), min_fg=256, min_frac=0.0, accept=accept)
    assert got is not None and len(accept) == 1
    assert "UNSTABLE" in accept[0] and "unreliable" in accept[0], accept[0]


def test_a_true_gain_is_not_flagged_unstable():
    """The false-positive side, and why the check is per-voxel rather than a second median
    above a higher floor: a common floor cuts more off the dimmer tile, so a PERFECT gain
    of 2 reported a 43% drift under the absolute-floor version.
    """
    shape = (8, 64, 64)
    a = np.full(shape, 100.0, np.float32)
    a.reshape(-1)[:4000] = 600.0
    a.reshape(-1)[4000:8000] = 4000.0
    b = a * np.float32(0.5)
    accept = []
    assert _pair(_cache(a, b, thr=150.0), min_fg=256, min_frac=0.0,
                 accept=accept) is not None
    assert "UNSTABLE" not in accept[0]


def test_independent_says_stability_was_not_checked_rather_than_nothing():
    """Silence and "checked, fine" must not look identical in the log."""
    a, b = _sparse(3000), _sparse(3000, gain=2.0)
    accept = []
    _pair(_cache(a, b, thr=500.0), "independent", min_fg=256, min_frac=0.0,
          accept=accept)
    assert "not checked" in accept[0]
