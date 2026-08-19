"""The `tile_threshold` override.

Worth its own file because this one number feeds three unrelated consumers -- the gain
floor, the empty/bimodal/uniform classification, and the apply stage's mask -- so the
thing to test is that overriding it at the PRODUCER reaches all three, not just that
`resolve_threshold` returns what it was told.
"""

import numpy as np
import pytest

from spotlight import config, tilestats


def _vol():
    """Sparse, with THREE populations: background, mid-intensity tissue, and bright
    structure. The middle one is the point -- it is what a too-high threshold discards
    and a corrected one keeps, so a two-level volume would select identically at 250 and
    at 3000 and the test would pass without testing anything."""
    v = np.full((4, 64, 64), 100, dtype=np.uint16)
    v.reshape(-1)[:4000] = 800
    v.reshape(-1)[:400] = 4000
    return v


def test_default_is_per_tile_otsu():
    v = _vol()
    thr, how = tilestats.resolve_threshold({}, 0, v)
    assert how == "otsu"
    assert thr == pytest.approx(float(tilestats.threshold_otsu(v)))


def test_a_number_overrides_otsu_entirely():
    thr, how = tilestats.resolve_threshold({"tile_threshold": 250}, 0, _vol())
    assert thr == 250.0 and "250" in how


def test_a_float_number_survives_as_a_float():
    thr, _ = tilestats.resolve_threshold({"tile_threshold": 137.5}, 0, _vol())
    assert thr == 137.5


def _pooled_cfg(tmp_path, write_measurement=True):
    """A real results_root, so "pooled" is exercised through the same file the emptiness
    stage actually writes rather than through a patched function."""
    if write_measurement:
        d = tmp_path / "intensity_stats"
        d.mkdir(parents=True)
        (d / "setup0.json").write_text('{"empty_threshold": 1114.0}')
    return {"tile_threshold": "pooled", "results_root": str(tmp_path),
            "last_setup": 0, "setups_per_camera": 1, "setups_per_row": 1,
            "setup_ids": []}


def test_pooled_reads_the_emptiness_stage_measurement(tmp_path):
    thr, how = tilestats.resolve_threshold(_pooled_cfg(tmp_path), 0, _vol())
    assert thr == 1114.0 and "pooled" in how


def test_pooled_without_the_emptiness_stage_says_what_to_run(tmp_path):
    """The error has to name the stage to run, or "pooled" just looks broken."""
    with pytest.raises(RuntimeError, match="emptiness"):
        tilestats.resolve_threshold(_pooled_cfg(tmp_path, write_measurement=False), 0,
                                    _vol())


def test_a_typo_is_not_silently_treated_as_otsu():
    """Silently falling back would look like the override simply had no effect, which is
    indistinguishable from a threshold that was already right."""
    with pytest.raises(ValueError, match="tile_threshold"):
        tilestats.resolve_threshold({"tile_threshold": "auto"}, 0, _vol())


def test_the_default_is_a_mode_resolve_threshold_accepts():
    """Membership, not a literal: the default is an operator choice that moves (it was
    "otsu", now "li"), but a default outside THRESHOLD_MODES makes every stage that reads it
    raise, so THAT is the thing worth pinning."""
    assert config.DEFAULTS["tile_threshold"] in tilestats.THRESHOLD_MODES


# ─── it must reach every consumer, not just the stats file ────────────────────


def test_the_override_changes_the_foreground_count_and_so_the_classification():
    """The consumer that matters most: a threshold set too high pushes a tile under
    MIN_FG_FRACTION, and an "empty" tile is dropped from the solve and passed through
    UNCORRECTED. Measured on a real worm: tile 0's Otsu of 1621 gave 0.41% foreground;
    at 250 it gave 2.01%.
    """
    v = _vol()
    high = tilestats._compute_stats(v, thr=3000)
    low = tilestats._compute_stats(v, thr=250)
    assert high["n_foreground"] < low["n_foreground"]
    assert high["threshold"] == 3000 and low["threshold"] == 250
    # n_voxels is what the fraction gate divides by, so it must not move with the floor.
    assert high["n_voxels"] == low["n_voxels"] == v.size


def test_the_override_changes_the_masked_mean_the_apply_stage_rescales_to():
    """`mean`/`std` are computed over the masked foreground, so they move with the
    threshold -- that is the apply stage's rescale target."""
    v = _vol()
    assert (tilestats._compute_stats(v, thr=3000)["mean"]
            > tilestats._compute_stats(v, thr=250)["mean"])


def test_compute_stats_without_an_override_still_does_otsu():
    """The default path must be untouched, or every existing dataset shifts."""
    v = _vol()
    assert (tilestats._compute_stats(v)["threshold"]
            == pytest.approx(float(tilestats.threshold_otsu(v))))


def test_the_typo_message_lists_the_modes():
    with pytest.raises(ValueError, match="'otsu', 'li', 'pooled'"):
        tilestats.resolve_threshold({"tile_threshold": "otso"}, 0, _vol())


def test_triangle_is_no_longer_accepted():
    """Removed after measuring it: best on the sparse worm, actively wrong on the dense
    s7 (0.5-1.9% foreground on tiles that are ~30% tissue). A mode that inverts between
    datasets is worse than no mode."""
    with pytest.raises(ValueError, match="tile_threshold"):
        tilestats.resolve_threshold({"tile_threshold": "triangle"}, 0, _vol())


def test_li_is_available_and_resists_the_tail():
    v = _vol()
    li, how = tilestats.resolve_threshold({"tile_threshold": "li"}, 0, v)
    otsu, _ = tilestats.resolve_threshold({}, 0, v)
    assert how == "li"
    assert li < otsu, f"li {li} should sit below otsu {otsu} on a heavy-tailed tile"


def _heavy_tailed():
    """Poisson background at a few counts plus a lognormal tail on 3% of voxels -- the
    shape a sparse fluorescence tile actually has (421 distinct levels, max ~21000).

    `_vol()` will not do for the identity below: with only three distinct values the
    class means are step functions of the threshold, skimage's iteration terminates on
    its first step, and it returns the image mean. That is a property of a degenerate
    histogram, not of Li, and asserting against it would test the wrong thing.
    """
    rng = np.random.default_rng(0)
    n = 4 * 64 * 64
    a = rng.poisson(4.0, n).astype(np.float64)
    k = int(0.03 * n)
    a[:k] = rng.lognormal(mean=6.0, sigma=1.2, size=k)
    return np.clip(a, 0, 65535).astype(np.uint16).reshape(4, 64, 64)


def test_li_sits_at_the_logarithmic_mean_of_the_class_means():
    """The identity that explains the whole method, and the thing that would break
    silently if skimage swapped implementations: Li's fixed point is the LOGARITHMIC mean
    of the two class means, where isodata's is the arithmetic mean. Holds to 0.0000% on a
    realistic histogram, and to within 0.4 counts on real worm tiles."""
    v = _heavy_tailed()
    a = v.reshape(-1).astype(np.float64)
    t, _ = tilestats.resolve_threshold({"tile_threshold": "li"}, 0, v)
    m0, m1 = a[a <= t].mean(), a[a > t].mean()
    log_mean = (m1 - m0) / (np.log(m1) - np.log(m0))
    assert t == pytest.approx(log_mean, rel=1e-4)
    # Strictly below the arithmetic mean, which is what isodata would have picked. This
    # gap IS the reason Li resists a heavy tail.
    assert log_mean < (m0 + m1) / 2


def test_li_stays_far_below_otsu_on_a_realistic_heavy_tail():
    """The measured effect, in the units that matter: on this histogram Otsu lands at
    ~4400 and Li at ~200, and on the real worm it was 1621 vs 274."""
    v = _heavy_tailed()
    li, _ = tilestats.resolve_threshold({"tile_threshold": "li"}, 0, v)
    otsu, _ = tilestats.resolve_threshold({}, 0, v)
    assert otsu > 10 * li, f"otsu {otsu} vs li {li}"


def test_the_typo_message_lists_li_too():
    with pytest.raises(ValueError, match="li"):
        tilestats.resolve_threshold({"tile_threshold": "lie"}, 0, _vol())


def test_li_takes_the_histogram_path_even_when_basic_correction_made_it_float():
    """`apply_basic` makes `_read_tile_volume` return float32, and skimage's Li reads its
    implementation off the dtype -- float means a full re-scan plus an allocation per
    iteration (1000 ms vs 64 ms on a 19.9M-voxel tile). The integer view must not change
    the answer beyond rounding."""
    v = _heavy_tailed()
    as_float = v.astype(np.float32)
    assert tilestats._histogrammable(as_float).dtype == np.uint16
    assert tilestats._histogrammable(v) is v, "integer input must not be copied"
    t_int, _ = tilestats.resolve_threshold({"tile_threshold": "li"}, 0, v)
    t_flt, _ = tilestats.resolve_threshold({"tile_threshold": "li"}, 0, as_float)
    assert t_flt == pytest.approx(t_int, rel=0.01)


def test_the_integer_view_clips_rather_than_wrapping():
    """BaSiC divides by the flat field, so a corrected value CAN exceed 65535. A bare
    astype would wrap 65536 to 0 and drag the threshold to nonsense."""
    v = np.array([[[0.0, 70000.0, 100.0, 65535.0]]], dtype=np.float32)
    out = tilestats._histogrammable(v)
    assert out.max() == 65535 and out.min() == 0


# ─── one setting, every stage ────────────────────────────────────────────────


def test_threshold_values_is_the_shared_core():
    """`resolve_threshold` (per tile) and the emptiness stage's pooled split must route
    through the same function, or `li` could mean one thing in one stage and another in
    the next -- judging emptiness on a different population than the stats measure."""
    v = _heavy_tailed()
    per_tile, how = tilestats.resolve_threshold({"tile_threshold": "li"}, 0, v)
    shared, how2 = tilestats.threshold_values("li", v)
    assert (per_tile, how) == (shared, how2)


def test_emptiness_uses_the_same_setting(monkeypatch):
    """The point of the change: `tile_threshold = "li"` must reach the emptiness stage's
    pooled threshold too."""
    from spotlight import emptiness
    assert "threshold_values" in emptiness.cmd_emptiness.__globals__
    assert "threshold_mode" in emptiness.cmd_emptiness.__globals__


def test_pooled_resolves_to_otsu_for_the_pooled_split_itself():
    """"pooled" NAMES the emptiness stage's own result, so using it there would be
    circular; it has to mean otsu at that one site."""
    mode = tilestats.threshold_mode({"tile_threshold": "pooled"})
    pooled_mode = "otsu" if mode == "pooled" else mode
    v = _heavy_tailed()
    assert tilestats.threshold_values(pooled_mode, v)[1] == "otsu"


def test_threshold_mode_validates_once_for_everyone():
    assert tilestats.threshold_mode({}) == "otsu"
    assert tilestats.threshold_mode({"tile_threshold": ""}) == "otsu"
    assert tilestats.threshold_mode({"tile_threshold": "li"}) == "li"
    assert tilestats.threshold_mode({"tile_threshold": 250}) == 250.0
    with pytest.raises(ValueError, match="tile_threshold"):
        tilestats.threshold_mode({"tile_threshold": "triangle"})


def test_a_numeric_threshold_skips_the_pooled_sampling_pass(monkeypatch):
    """With a fixed number the emptiness stage can skip its first pass -- a strided read
    of EVERY tile whose only purpose is producing that number. Asserted by counting tile
    reads, not by reading the source: pass 2 still reads each tile once, so the
    signature of the skip is 1 read per tile instead of 2."""
    from spotlight import emptiness

    reads = []

    def fake_read(cfg, setup, level):
        reads.append(setup)
        v = np.full((6, 8, 8), 10, dtype=np.uint16)
        v[0, :4, :4] = 5000
        return v

    monkeypatch.setattr(emptiness, "_read_tile", fake_read)
    monkeypatch.setattr(emptiness, "_select_level", lambda cfg, s, **kw: 0)

    cfg = {"tile_threshold": 250}
    emptiness._empty_areas(cfg, [0, 1, 2])
    assert sorted(reads) == [0, 1, 2], f"one read per tile, got {sorted(reads)}"

    reads.clear()
    emptiness._empty_areas({"tile_threshold": "otsu"}, [0, 1, 2])
    assert len(reads) == 6, f"otsu needs the sampling pass too, got {len(reads)}"


def test_emptiness_honours_li(monkeypatch):
    """The whole point of the change: `li` must reach the pooled split."""
    from spotlight import emptiness

    def fake_read(cfg, setup, level):
        rng = np.random.default_rng(setup)
        v = rng.poisson(4.0, 6 * 32 * 32).astype(np.float64)
        v[:200] = rng.lognormal(6.0, 1.0, 200)
        return np.clip(v, 0, 65535).astype(np.uint16).reshape(6, 32, 32)

    monkeypatch.setattr(emptiness, "_read_tile", fake_read)
    monkeypatch.setattr(emptiness, "_select_level", lambda cfg, s, **kw: 0)
    seen = {}
    real = tilestats.threshold_values
    monkeypatch.setattr(emptiness, "threshold_values",
                        lambda mode, vals: seen.setdefault("mode", mode) and None
                        or real(mode, vals))
    emptiness._empty_areas({"tile_threshold": "li"}, [0, 1])
    assert seen["mode"] == "li"


def test_the_stats_stage_records_every_method_not_just_the_chosen_one():
    """So a later stage can name a different one without re-reading the tile -- the stats
    stage is the only place a whole tile is in memory, and each method costs tens of ms
    against a read of seconds."""
    v = _heavy_tailed()
    cat = tilestats.threshold_catalogue(v)
    assert set(cat) == set(tilestats.THRESHOLD_METHODS)
    assert cat["li"] < cat["otsu"], "on a heavy tail li must sit below otsu"


def test_the_catalogue_reuses_the_already_computed_threshold():
    """`resolve_threshold` has usually just computed one of these; recomputing it would
    double the cost of the mode people actually selected."""
    v = _heavy_tailed()
    cat = tilestats.threshold_catalogue(v, "li", 123.0)
    assert cat["li"] == 123.0
    assert cat["otsu"] == pytest.approx(tilestats.threshold_values("otsu", v)[0])
