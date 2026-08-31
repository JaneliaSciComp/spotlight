"""Tests for the `spotfix` stage.

Each pins a specific bug the prototype actually hit, rather than restating the code:
the empty-overlap slice wrap, the half-cell interpolation offset, the gain floor, and the
precondition that keeps a per-tile gain error out of a local-defect correction.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from spotlight import spotfix


def _xml(tmp_path, tiles, channels=None):
    """A minimal SpimData2 xml: `tiles` is {setup: (size_xyz, origin_xyz)}.

    `channels` optionally gives {setup: channel}; omitted means no <attributes> block at
    all, which must behave as single-channel.
    """
    parts = ['<?xml version="1.0"?>', "<SpimData>", "<SequenceDescription>", "<ViewSetups>"]
    for s, (size, _o) in tiles.items():
        attr = ("<attributes><illumination>0</illumination>"
                f"<channel>{channels[s]}</channel><angle>0</angle></attributes>"
                if channels else "")
        parts += [f"<ViewSetup><id>{s}</id>"
                  f"<size>{size[0]} {size[1]} {size[2]}</size>"
                  "<voxelSize><unit>um</unit><size>0.157 0.157 0.628</size></voxelSize>"
                  f"{attr}</ViewSetup>"]
    parts += ["</ViewSetups>", "</SequenceDescription>", "<ViewRegistrations>"]
    for s, (_size, o) in tiles.items():
        parts += [f'<ViewRegistration timepoint="0" setup="{s}">',
                  '<ViewTransform type="affine"><Name>Translation</Name>',
                  f"<affine>1 0 0 {o[0]} 0 1 0 {o[1]} 0 0 1 {o[2]}</affine>",
                  "</ViewTransform>",
                  '<ViewTransform type="affine"><Name>calibration</Name>',
                  "<affine>1 0 0 0 0 1 0 0 0 0 4 0</affine>",
                  "</ViewTransform>", "</ViewRegistration>"]
    parts += ["</ViewRegistrations>", "</SpimData>"]
    p = tmp_path / "dataset.xml"
    p.write_text("".join(parts))
    return {"dataset_xml": str(p)}


def test_neighbours_come_from_the_geometry_and_exclude_disjoint_tiles(tmp_path):
    # 0 overlaps 1 (shifted half a tile) and not 2 (shifted three tiles away).
    cfg = _xml(tmp_path, {0: ((100, 100, 10), (0, 0, 0)),
                          1: ((100, 100, 10), (50, 0, 0)),
                          2: ((100, 100, 10), (300, 0, 0))})
    assert spotfix.neighbours(cfg, 0) == [1]
    assert spotfix.neighbours(cfg, 2) == []


def test_a_different_channel_is_not_a_neighbour_however_perfectly_it_overlaps(tmp_path):
    """The same tile in another channel occupies exactly the same space, so overlap alone
    admits it -- and it is a different fluorophore, not an independent measurement of this
    signal. In the real dataset those setups also carry only a single-scale `raw/0`, so
    asking one for the analysis level fails outright:

        NOT_FOUND: ... s672-t0.zarr/4/zarr.json does not exist

    on a run that asked for tile 126.
    """
    tiles = {126: ((100, 100, 10), (0, 0, 0)),     # ch 0, the tile being fixed
             127: ((100, 100, 10), (50, 0, 0)),    # ch 0, a real neighbour
             672: ((100, 100, 10), (0, 0, 0)),     # ch 1, exactly co-located
             673: ((100, 100, 10), (50, 0, 0))}    # ch 1, co-located with the neighbour
    cfg = _xml(tmp_path, tiles, channels={126: 0, 127: 0, 672: 1, 673: 1})
    assert spotfix.neighbours(cfg, 126) == [127]
    assert spotfix.neighbours(cfg, 672) == [673]


def test_a_file_without_attributes_behaves_as_single_channel(tmp_path):
    cfg = _xml(tmp_path, {0: ((100, 100, 10), (0, 0, 0)),
                          1: ((100, 100, 10), (50, 0, 0))})
    assert spotfix.neighbours(cfg, 0) == [1]


def test_a_non_overlapping_shift_places_nothing_rather_than_wrapping():
    # An earlier version let a negative stop wrap and silently dropped whole tiles, which
    # reads as a coverage result instead of a bug.
    vals = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    for shift in ((99, 0, 0), (-99, 0, 0), (0, 99, 0), (0, 0, -99)):
        out, n = spotfix._place(vals, shift, (3, 3, 3))
        assert n == 0 and not out.any(), shift
    out, n = spotfix._place(vals, (0, 0, 0), (3, 3, 3))
    assert n == 27 and np.array_equal(out, vals)


def test_the_gain_sampler_is_endpoint_aligned():
    # `scipy.ndimage.zoom(grid_mode=False)` maps output j to coarse j*(nc-1)/(no-1). A
    # cell-width mapping is off by half a cell and lands the gain on the wrong cell at
    # every boundary.
    g = np.array([[[1.0, 5.0]]], dtype=np.float32)          # one z, one y, two x cells
    plane = spotfix._sampler(g, (1, 1, 5))
    got = plane(0, 0, 1, 0, 5)[0]
    assert got[0] == pytest.approx(1.0)                      # first output == first cell
    assert got[-1] == pytest.approx(5.0)                     # last output == last cell
    assert got[2] == pytest.approx(3.0)                      # midpoint interpolates
    from scipy.ndimage import zoom
    assert np.allclose(got, zoom(g[0, 0], 5 / 2, order=1, mode="nearest"), atol=1e-5)


def test_edge_step_and_the_mask_threshold_are_two_views_of_one_number():
    # The mask ends where r == edge_r, so the gain just inside is 1/edge_r: detection
    # sensitivity and boundary visibility cannot be tuned apart.
    cfg = {"spotfix_edge_step": 0.05263157894736842}
    assert spotfix.edge_r(cfg) == pytest.approx(0.95)
    cfg = {"spotfix_edge_step": 1 / 0.75 - 1}
    assert spotfix.edge_r(cfg) == pytest.approx(0.75)


def test_the_precondition_tolerance_scales_with_the_feather_width():
    # A global offset the size of the feather drops half the tile below the threshold on
    # its own, so the bound has to be relative to it, not a fixed percentage.
    r = np.full((4, 4, 4), 0.97, np.float32)
    ev = np.ones((4, 4, 4), bool)
    lev, ok, tol = spotfix._precondition(r, ev, ev, {"spotfix_edge_step": 1 / 0.95 - 1})
    assert lev == pytest.approx(0.97)
    assert tol == pytest.approx(0.025) and not ok      # 3% offset, 2.5% allowed
    # a wider feather tolerates a bigger offset
    _lev, ok2, tol2 = spotfix._precondition(r, ev, ev, {"spotfix_edge_step": 1 / 0.85 - 1})
    assert tol2 == pytest.approx(0.075) and ok2


def test_a_shard_rounds_rather_than_truncates_and_never_darkens():
    # Truncating instead of rounding cost the prototype a whole gray level across 99% of a
    # tile. The fractional part has to straddle 0.5 for the assertion to tell the two
    # apart at all: 100 * 1.004 = 100.4 truncates AND rounds to 100, so it proves nothing.
    canon = np.full((2, 3, 4), 100, np.uint16)
    flat = spotfix._sampler(np.full((2, 1, 1), 1.0, np.float32), (2, 3, 4))
    assert (spotfix._apply_shard(canon.copy(), flat, None, 0, 0, 3, 0, 4, 65535) == 100).all(), \
        "a gain of exactly 1 must not move any voxel"
    up = spotfix._sampler(np.full((2, 1, 1), 1.006, np.float32), (2, 3, 4))
    out = spotfix._apply_shard(canon.copy(), up, None, 0, 0, 3, 0, 4, 65535)
    assert (out == 101).all(), "100 * 1.006 = 100.6 must round UP to 101, not truncate to 100"
    down = spotfix._sampler(np.full((2, 1, 1), 1.004, np.float32), (2, 3, 4))
    out2 = spotfix._apply_shard(canon.copy(), down, None, 0, 0, 3, 0, 4, 65535)
    assert (out2 == 100).all(), "100 * 1.004 = 100.4 must round DOWN to 100"


def test_backup_renames_and_never_overwrites_an_existing_backup(tmp_path):
    root = tmp_path / "out"
    (root / "s7-t0.zarr" / "0").mkdir(parents=True)
    (root / "s7-t0.zarr" / "0" / "zarr.json").write_text("{}")
    cfg = {"output_intensity_path": str(root), "output_format": "zarr3"}
    first = spotfix.backup_tile(cfg, 7)
    assert first.endswith(".prespotfix")
    assert not os.path.exists(str(root / "s7-t0.zarr")), "the tile must be MOVED, not copied"
    assert os.path.exists(f"{first}/0/zarr.json"), "the backup must still hold the data"
    # a second run must not clobber the first backup
    (root / "s7-t0.zarr" / "0").mkdir(parents=True)
    (root / "s7-t0.zarr" / "0" / "zarr.json").write_text("{}")
    second = spotfix.backup_tile(cfg, 7)
    assert second != first and os.path.exists(first)


def test_the_expectation_prefers_a_covering_neighbour_over_the_slice_level():
    # Case 1 of the rule: where a neighbour covers the cell, its value IS the measurement
    # and must win over any statistic derived from the slice.
    obs = np.full((1, 1, 2), 50.0, np.float32)
    nbo = np.array([[[200.0, 0.0]]], np.float32)
    covf = np.array([[[1.0, 0.0]]], np.float32)
    nbfg = np.array([[[1.0, 0.0]]], np.float32)
    signal = np.ones((1, 1, 2), np.float32)
    exp = spotfix.expectation(obs, nbo, covf, nbfg, bg=10.0, signal=signal)
    assert exp[0, 0, 0] == pytest.approx(200.0)
    assert exp[0, 0, 1] == pytest.approx(200.0), "an uncovered cell takes the slice level"


def test_spotfix_is_its_own_pipeline_and_needs_explicit_tiles():
    from spotlight import local
    assert local.PIPELINES["spotfix"] == ["spotfix"]
    assert local.apply_basic_for("spotfix") is False
    with pytest.raises(SystemExit):
        local._units({}, "spotfix", "none", tiles=[])


# ─── end to end on a synthetic two-tile store ────────────────────────────────────


def _store(root, setups, shape_zyx, fill):
    """A zarr3 store with one level-0 array per setup, written through tensorstore so the
    metadata is whatever this codebase's own reader expects."""
    import tensorstore as ts
    for s in setups:
        arr = ts.open({
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": f"{root}/s{s}-t0.zarr/0"},
            "metadata": {"data_type": "uint16",
                         "shape": [1, 1, *shape_zyx],
                         "chunk_grid": {"name": "regular", "configuration":
                                        {"chunk_shape": [1, 1, 8, 8, 8]}}},
        }, create=True, delete_existing=True).result()
        arr[...].write(fill(s)[None, None]).result()


@pytest.fixture
def two_tiles(tmp_path, monkeypatch):
    """Tile 0 with two dark regions and tile 1 covering only half of it.

    PARTIAL coverage on purpose. With a neighbour over every voxel, `need` is exactly 1
    wherever the tile is healthy, so the mask and the gate cannot be told apart from the
    need-clamp -- a fixture that hides them. Half-covered is the real case: in the
    UNcovered half the expectation falls back to the z-slice level, so `need` can be large
    where the tile is merely at background, and only the gate stops that being amplified.

      z 0-7                 healthy, 300 everywhere
      z 8-15, x 8-15        the DEFECT: dark at 30, and a neighbour says 300 is there
      z 8-15, x 0-7         BACKGROUND at 12, uncovered -- must stay dark
    """
    from spotlight import config
    Z, Y, X = 16, 16, 16
    healthy = np.full((Z, Y, X), 300, np.uint16)

    def fill(s):
        a = healthy.copy()
        if s == 0:
            a[8:, :, 8:] = 30            # the defect, where the neighbour can see it
            a[8:, :, :8] = 12            # background, where nothing can
        return a

    out = tmp_path / "corrected"
    _store(str(out), (0, 1), (Z, Y, X), fill)
    # Tile 1 shifted half a tile in x, so tile 0's x 0-7 has no neighbour at all.
    cfg = _xml(tmp_path, {0: ((X, Y, Z), (0, 0, 0)), 1: ((X, Y, Z), (8, 0, 0))})
    results = tmp_path / "results"
    results.mkdir()
    (results / "intensity_target.json").write_text(json.dumps(
        {"setups": {"0": {"bg_mean": 10.0, "bg_std": 3.0, "threshold": 100.0},
                    "1": {"bg_mean": 10.0, "bg_std": 3.0, "threshold": 100.0}}}))
    monkeypatch.chdir(tmp_path)
    return {**cfg,
            "output_intensity_path": str(out), "output_format": "zarr3",
            "results_root": str(results),
            "chunk_size": [8, 8, 8], "shard_size": [8, 8, 8],
            "spotfix_level": 0, "spotfix_cell_um": 0.157 * 4,
            "spotfix_smooth_z_um": 0.628, "spotfix_smooth_lat_um": 0.157 * 4,
            "spotfix_presence_um": 0.157 * 8, "spotfix_contrast_um": 0.157 * 8,
            "n_cores_correction": 2}


def test_fix_tile_brightens_the_defect_backs_up_and_leaves_healthy_voxels_alone(two_tiles):
    import tensorstore as ts
    cfg = two_tiles
    before = np.asarray(ts.open({
        "driver": "zarr3", "kvstore": {"driver": "file",
                                       "path": f"{cfg['output_intensity_path']}/s0-t0.zarr/0"},
    }, open=True).result()[0, 0].read().result())

    report = spotfix.fix_tile(cfg, 0)
    assert not report.get("skipped"), report

    after = np.asarray(ts.open({
        "driver": "zarr3", "kvstore": {"driver": "file",
                                       "path": f"{cfg['output_intensity_path']}/s0-t0.zarr/0"},
    }, open=True).result()[0, 0].read().result())

    # the defect is lifted toward what the neighbour says is there
    assert after[8:, :, 8:].mean() > 3 * before[8:, :, 8:].mean(), \
        (before[8:, :, 8:].mean(), after[8:, :, 8:].mean())
    # the healthy half is untouched
    assert np.array_equal(after[:8], before[:8]), "healthy voxels must not move"
    # and BACKGROUND that no neighbour covers stays dark. This is the assertion the gate
    # exists for: the slice-level expectation there is ~300, so `need` is ~25x, and
    # nothing but the gate stops 12 DN of background being amplified into signal.
    #
    # Away from the boundary, because the trilinear feather DOES reach across it by design
    # -- and on this deliberately tiny grid (4 lateral cells over 16 voxels) one cell is
    # 4 voxels, so the feather is a quarter of the tile rather than the thin rim it is at
    # real cell sizes. The median pins the bulk; the max pins that nothing was amplified
    # toward the 300 the slice level would have demanded.
    bg = after[8:, :, :6]
    assert bg.max() <= 13, f"uncovered background was amplified to {bg.max()}"
    assert np.median(after[8:, :, :8]) == 12, "the bulk of the background must not move"
    # the previous version is kept, not deleted, and still holds the ORIGINAL values
    kept = np.asarray(ts.open({
        "driver": "zarr3", "kvstore": {"driver": "file", "path": f"{report['backup']}/0"},
    }, open=True).result()[0, 0].read().result())
    assert np.array_equal(kept, before), "the backup must hold the pre-fix voxels"
    assert os.path.exists(os.path.join(cfg["results_root"], "spotfix_setup0.json"))


def test_fix_tile_refuses_a_tile_that_is_uniformly_off_its_neighbours(two_tiles, tmp_path):
    """A uniform offset is a per-tile gain error. Correcting it here would smear it into a
    large spatially varying correction, so the stage must refuse rather than guess."""
    import tensorstore as ts
    cfg = two_tiles
    p = f"{cfg['output_intensity_path']}/s0-t0.zarr/0"
    arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": p}},
                  open=True).result()
    dim = np.full((16, 16, 16), 270, np.uint16)      # 10% below the neighbour, everywhere
    arr[...].write(dim[None, None]).result()
    with pytest.raises(RuntimeError, match="per-tile GAIN error"):
        spotfix.fix_tile(cfg, 0)
    # and it must not have moved the tile aside before refusing
    assert os.path.exists(p), "a refused tile must be left exactly where it was"


def test_a_dark_structure_inside_the_defect_keeps_its_own_value(tmp_path, monkeypatch):
    """`local_contrast_weight` is what separates an anatomical HOLE from attenuation.

    Both are dark pointwise; only the ratio to their surroundings tells them apart. Omitting
    this step was why the first packaged run looked worse than the prototype on both tiles:
    every dark structure inside the corrected region took the full gain.

      z 8-15, x 8-15   attenuated to 150 with a neighbour saying 300  -> must be lifted
      inside it        a 2x2 hole at 5 DN against ~150 surroundings   -> must stay dark
    """
    Z, Y, X = 16, 16, 16

    def fill(s):
        a = np.full((Z, Y, X), 300, np.uint16)
        if s == 0:
            a[8:, :, 8:] = 150                     # attenuated, not dead
            a[8:, 6:8, 11:13] = 5                  # a compact hole inside it
        return a

    out = tmp_path / "corrected"
    _store(str(out), (0, 1), (Z, Y, X), fill)
    cfg = _xml(tmp_path, {0: ((X, Y, Z), (0, 0, 0)), 1: ((X, Y, Z), (8, 0, 0))})
    results = tmp_path / "results"
    results.mkdir()
    (results / "intensity_target.json").write_text(json.dumps(
        {"setups": {"0": {"bg_mean": 10.0, "bg_std": 3.0, "threshold": 100.0},
                    "1": {"bg_mean": 10.0, "bg_std": 3.0, "threshold": 100.0}}}))
    monkeypatch.chdir(tmp_path)
    cfg = {**cfg, "output_intensity_path": str(out), "output_format": "zarr3",
           "results_root": str(results), "chunk_size": [8, 8, 8], "shard_size": [8, 8, 8],
           "spotfix_level": 0, "spotfix_cell_um": 0.157 * 4,
           "spotfix_smooth_z_um": 0.628, "spotfix_smooth_lat_um": 0.157 * 4,
           "spotfix_presence_um": 0.157 * 8, "spotfix_contrast_um": 0.157 * 8,
           "n_cores_correction": 2}

    import tensorstore as ts
    read = lambda: np.asarray(ts.open({
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": f"{out}/s0-t0.zarr/0"}},
        open=True).result()[0, 0].read().result())
    before = read()
    spotfix.fix_tile(cfg, 0)
    after = read()

    hole = (slice(8, None), slice(6, 8), slice(11, 13))
    tissue = np.zeros((Z, Y, X), bool)
    tissue[8:, :, 8:] = True
    tissue[hole] = False

    assert after[tissue].mean() > 1.3 * before[tissue].mean(), \
        "the attenuated tissue around the hole must still be lifted"
    assert after[hole].max() <= 7, \
        f"the hole was brightened to {after[hole].max()} -- the contrast weight is not applied"


def test_nbfg_asks_whether_the_NEIGHBOUR_SHOWS_FOREGROUND_not_whether_it_is_present():
    """`nbfg` feeds `local_presence`, i.e. "is there specimen here". Conflating it with
    `covf` ("is a neighbour here") opens the gate everywhere: measured on tile 126, the gate
    fired on 99.3% of cells instead of 37.5% and the post-despeckle gain reached 55x instead
    of 22x. Two independent errors did it -- no foreground threshold, and dividing by the
    cell size rather than by the covered count -- so both are pinned here.
    """
    # one 1x4x4 cell. A neighbour covers half of it; of the covered voxels, a quarter are
    # above the foreground level.
    nb = np.zeros((1, 4, 4), np.float32)
    cov = np.zeros((1, 4, 4), bool)
    cov[0, :2, :] = True                 # 8 of 16 voxels covered
    nb[0, 0, :2] = 300.0                 # 2 of those 8 are foreground
    nb[0, 0, 2:] = 50.0                  # covered but dim -- present, NOT foreground
    nb[0, 1, :] = 50.0
    a = np.full((1, 4, 4), 100, np.float32)
    fgmask = cov & (nb >= 256.0)
    _obs, _nbo, covf, nbfg = spotfix._coarsen(a, nb, cov, fgmask, 4, 4)
    assert covf[0, 0, 0] == pytest.approx(0.5), "half the cell is covered"
    assert nbfg[0, 0, 0] == pytest.approx(0.25), \
        ("of the COVERED voxels, a quarter show foreground -- got "
         f"{nbfg[0, 0, 0]}; 0.5 means it is measuring coverage, 0.125 means it divided by "
         "the cell size instead of the covered count")
    # a cell nothing covers reports 0, not a division by zero
    cov2 = np.zeros((1, 4, 4), bool)
    _o, _n, covf2, nbfg2 = spotfix._coarsen(a, nb, cov2, cov2 & (nb >= 256.0), 4, 4)
    assert covf2[0, 0, 0] == 0.0 and nbfg2[0, 0, 0] == 0.0


def test_a_cell_far_below_background_is_never_corrected():
    """A cell median well BELOW the background mean means no light reached it -- outside the
    illuminated volume rather than attenuated -- so nothing may be extrapolated into it.

    This is the only feature found that separates the sites needing to stay dark from the
    ones needing fill: both are uncovered, and they differ only by 5-6 DN against 10-14 when
    bg is 16. `signal_floor_weight` cannot express it, because it clips at 0 and maps both
    to exactly 0. Six other candidates were measured and separate neither (column plateau,
    live-z fraction, distance to coverage, slice level, local neighbour level, the tile's
    own lateral mean).
    """
    bg, bg_std = 16.0, 8.0
    obs = np.array([[[5.0, 13.0, 20.0]]], np.float32)      # 1.4 sd below / 0.4 below / above
    dead = obs < bg - 1.0 * bg_std                          # the rule under test
    assert dead.tolist() == [[[True, False, False]]], \
        "only the cell more than 1 sd below the background MEAN counts as unilluminated"
    # and the floor weight really does lose the distinction it has to make
    sig = spotfix._signal(obs, bg, bg_std, 4.0)
    assert sig[0, 0, 0] == sig[0, 0, 1] == 0.0, \
        "signal_floor_weight clips at 0, so 5 DN and 13 DN are indistinguishable to it"


def test_fix_tile_leaves_an_unilluminated_region_alone_even_where_a_neighbour_is_bright(
        tmp_path, monkeypatch):
    """End to end: a region well below background must survive untouched even though a
    neighbour covers it and reads 300 DN. Without the dead-cell rule the neighbour's value
    is taken as the target and the region is amplified 100x.

      z 8-15, x 8-15          attenuated to 30, covered, neighbour says 300 -> lifted
      inside it, y 4-7        3 DN, more than 1 sd below bg (10 +- 3)       -> untouched

    The dead region is aligned to the 4-voxel gain cells on purpose: `obs` is a cell MEDIAN,
    so a region straddling cell boundaries mixes 3 DN with 30 and never falls below the cut.
    """
    import tensorstore as ts
    Z, Y, X = 16, 16, 16

    def fill(s):
        a = np.full((Z, Y, X), 300, np.uint16)
        if s == 0:
            a[8:, :, 8:] = 30
            a[8:, 4:8, 8:] = 3            # no light reached here (cells y=1)
        return a

    out = tmp_path / "corrected"
    _store(str(out), (0, 1), (Z, Y, X), fill)
    cfg = _xml(tmp_path, {0: ((X, Y, Z), (0, 0, 0)), 1: ((X, Y, Z), (8, 0, 0))})
    results = tmp_path / "results"
    results.mkdir()
    (results / "intensity_target.json").write_text(json.dumps(
        {"setups": {"0": {"bg_mean": 10.0, "bg_std": 3.0, "threshold": 100.0},
                    "1": {"bg_mean": 10.0, "bg_std": 3.0, "threshold": 100.0}}}))
    monkeypatch.chdir(tmp_path)
    cfg = {**cfg, "output_intensity_path": str(out), "output_format": "zarr3",
           "results_root": str(results), "chunk_size": [8, 8, 8], "shard_size": [8, 8, 8],
           "spotfix_level": 0, "spotfix_cell_um": 0.157 * 4,
           "spotfix_smooth_z_um": 0.628, "spotfix_smooth_lat_um": 0.157 * 4,
           "spotfix_presence_um": 0.157 * 8, "spotfix_contrast_um": 0.157 * 8,
           "n_cores_correction": 2}
    read = lambda: np.asarray(ts.open({
        "driver": "zarr3", "kvstore": {"driver": "file", "path": f"{out}/s0-t0.zarr/0"}},
        open=True).result()[0, 0].read().result())
    before = read()
    spotfix.fix_tile(cfg, 0)
    after = read()

    alive = np.zeros((Z, Y, X), bool); alive[8:, 10:, 8:] = True    # the 30 DN region
    deadr = np.zeros((Z, Y, X), bool); deadr[8:, 5:7, 10:14] = True  # dead interior
    assert after[alive].mean() > 1.5 * before[alive].mean(), \
        "the attenuated region beside it must still be lifted"
    assert after[deadr].max() <= 4, \
        (f"an unilluminated region was amplified to {after[deadr].max()} -- the dead-cell "
         "rule is not being applied to the gate")
