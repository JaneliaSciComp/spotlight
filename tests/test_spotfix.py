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


def _xml(tmp_path, tiles):
    """A minimal SpimData2 xml: `tiles` is {setup: (size_xyz, origin_xyz)}."""
    parts = ['<?xml version="1.0"?>', "<SpimData>", "<SequenceDescription>", "<ViewSetups>"]
    for s, (size, _o) in tiles.items():
        parts += [f"<ViewSetup><id>{s}</id>"
                  f"<size>{size[0]} {size[1]} {size[2]}</size>"
                  "<voxelSize><unit>um</unit><size>0.157 0.157 0.628</size></voxelSize>"
                  "</ViewSetup>"]
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
    assert (spotfix._apply_shard(canon.copy(), flat, 0, 0, 3, 0, 4, 65535) == 100).all(), \
        "a gain of exactly 1 must not move any voxel"
    up = spotfix._sampler(np.full((2, 1, 1), 1.006, np.float32), (2, 3, 4))
    out = spotfix._apply_shard(canon.copy(), up, 0, 0, 3, 0, 4, 65535)
    assert (out == 101).all(), "100 * 1.006 = 100.6 must round UP to 101, not truncate to 100"
    down = spotfix._sampler(np.full((2, 1, 1), 1.004, np.float32), (2, 3, 4))
    out2 = spotfix._apply_shard(canon.copy(), down, 0, 0, 3, 0, 4, 65535)
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
        {"setups": {"0": {"bg_mean": 10.0, "bg_std": 3.0},
                    "1": {"bg_mean": 10.0, "bg_std": 3.0}}}))
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
