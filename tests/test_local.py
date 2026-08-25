"""The local (no-LSF) pipeline driver.

The headline test runs `stats -> qstack -> basic -> correct` for real on a synthetic
store, through the same CLI a user would type. That is the only way to know the driver
iterates the right units in the right order -- a mocked version would pass with the
stages in any order at all.
"""

import numpy as np
import pytest

from make_store import DEPTHS, X, Y, volume, write_store

from spotlight import config, local
from spotlight.__main__ import main


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    store = write_store(tmp_path / "in", "zarr2", setups=(0, 1, 2))
    monkeypatch.chdir(tmp_path)
    config.set_config(
        input_basic_path=store["input_basic_path"],
        output_basic_path=store["output_basic_path"],
        results_root=str(tmp_path / "results"), qstacks_dir=str(tmp_path / "qstacks"),
        input_format="zarr2", last_setup=2, setups_per_camera=3,
        chunk_size=[32, 32, 32], shard_size=[64, 64, 64],
        lsf_project="p", output_stem=str(tmp_path / "o"),
        error_stem=str(tmp_path / "e"), n_cores_stats=2, n_cores_correction=2,
    )
    return tmp_path


def test_basic_pipeline_runs_end_to_end_with_no_lsf(experiment):
    """No bsub, no array job -- one process walks every unit."""
    main(["run", "basic", "--stop-after", "basic"])

    assert (experiment / "qstacks" / "camera1.tiff").is_file()
    flat = experiment / "results" / "camera1" / "Flat-field.tif"
    assert flat.is_file() and (experiment / "results" / "camera1" / "Dark-field.tif").is_file()

    import tifffile
    f = np.asarray(tifffile.imread(str(flat)))
    assert f.shape == (Y, X)
    assert float(f.mean()) == pytest.approx(1.0, rel=1e-4)


def test_correct_stage_writes_every_setup(experiment):
    """The bsub array runs one element per setup; locally the driver must cover all of
    them, not just the first."""
    main(["run", "basic"])
    from spotlight import stores
    cfg = config.load_config()
    out = {**cfg, "input_basic_path": cfg["output_basic_path"]}
    for setup in (0, 1, 2):
        arr = stores.open_source(out, setup, 0)
        got = np.asarray(arr[:, :, :].read().result())
        assert got.shape == (DEPTHS[setup], Y, X), f"setup {setup} not written"


def test_dry_run_touches_nothing(experiment, capsys):
    main(["run", "both", "--dry-run"])
    out = capsys.readouterr().out
    assert "emptiness" in out and "correct" in out
    assert not (experiment / "qstacks").exists(), "dry run wrote output"


def test_stage_windows(experiment, capsys):
    main(["run", "basic", "--stop-after", "qstack", "--dry-run"])
    out = capsys.readouterr().out
    assert "basic" not in out.split("pipeline: basic ->")[1].split("\n")[0].split("qstack")[1]

    main(["run", "basic", "--start-at", "basic", "--dry-run"])
    out = capsys.readouterr().out
    assert "stats" not in out.split("\n")[0].split("->")[1]


def test_start_after_stop_is_rejected(experiment):
    with pytest.raises(ValueError, match="is after"):
        local.run_pipeline(config.load_config(), "basic",
                           start_at="correct", stop_after="stats")


@pytest.mark.parametrize("pipeline,expect_mode", [
    ("basic", "basic"), ("intensity", "intensity"), ("both", "both"),
])
def test_each_pipeline_pins_its_correction_mode(experiment, pipeline, expect_mode,
                                                monkeypatch):
    """`auto` would resolve from whatever files happen to be on disk; the pipeline the
    user named must decide instead."""
    seen = []
    monkeypatch.setattr("spotlight.correct.apply_correction_chunked",
                        lambda cfg, s, mode: seen.append(mode))
    monkeypatch.setattr("spotlight.local._plan", lambda *a, **k: ["correct"])
    local.run_pipeline(config.load_config(), pipeline)
    assert set(seen) == {expect_mode}


def test_stages_cover_every_cli_stage():
    """A stage added to the CLI but not to a pipeline is unreachable locally."""
    assert set(local.STAGES) == {
        "emptiness", "stats", "qstack", "basic", "int-stats", "int-aggregate", "correct"}
    for name, stages in local.PIPELINES.items():
        assert stages[0] == "emptiness", f"{name} must measure emptiness first"
        assert stages[-1] == "correct", f"{name} must end by correcting"


# ─── apply_basic is a property of the PIPELINE, not of what is on disk ────────


def test_apply_basic_follows_the_pipeline_not_the_filesystem():
    """`both` corrects with BaSiC by definition, so every stage must agree from the
    start -- including the stages that run BEFORE the `basic` stage creates the fields
    that `load_config`'s auto-detection looks for."""
    from spotlight import local
    assert local.apply_basic_for("both") is True
    assert local.apply_basic_for("intensity") is False
    assert local.apply_basic_for("basic") is False


def test_apply_basic_matches_what_the_correct_stage_will_demand():
    """The check that fired is `_check_basic_mode(cfg, recorded, ...)` in the apply stage,
    comparing against `correct._view`'s `apply_basic = (mode == "both")`. If those two
    ever disagree the pipeline fails midway, after writing every earlier stage.
    """
    from spotlight import correct, local
    for pipeline, mode in local._CORRECT_MODE.items():
        view = correct._view({"apply_basic": None, "input_basic_path": "",
                              "output_basic_path": ""}, mode)
        if mode != "basic":       # "basic" swaps to the BaSiC I/O view entirely
            assert view["apply_basic"] == local.apply_basic_for(pipeline), pipeline


def test_an_unknown_pipeline_is_rejected():
    from spotlight import local
    with pytest.raises(ValueError, match="unknown pipeline"):
        local.apply_basic_for("everything")


def test_run_pipeline_overrides_a_stale_autodetected_value(capsys, monkeypatch):
    """End to end through the driver: a cfg that arrives claiming apply_basic=True must
    not carry that into an `intensity` run."""
    from spotlight import local
    monkeypatch.setattr(local, "_units", lambda cfg, stage, mode: [])
    local.run_pipeline({"apply_basic": True}, "intensity", dry_run=True)
    assert "apply_basic=False" in capsys.readouterr().out


def test_stats_arrays_are_created_up_front_and_zero_byte_metadata_is_rebuilt(experiment):
    """The driver must leave the workers nothing to create.

    An n5 array whose attributes.json is zero bytes is unopenable and unrepairable --
    tensorstore has to read the metadata to write it -- so a create race does not just
    fail a run, it poisons the array for every run afterwards. Both halves are checked:
    the arrays exist before submission, and a poisoned one is rebuilt rather than handed
    over.
    """
    from spotlight import scripts

    scripts.create_quartile_histograms()
    meta = experiment / "results" / "camera1" / "q050" / "s0" / "attributes.json"
    assert meta.is_file() and meta.stat().st_size, "array not created before submission"
    good = meta.read_bytes()

    # Exactly what a late creator leaves behind: truncated metadata over a chunk that had
    # already been written. The stale chunk must not survive into the rebuilt array.
    meta.write_bytes(b"")
    stale = meta.parent / "0" / "6"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")

    scripts.create_quartile_histograms()
    assert meta.read_bytes() == good, "zero-byte attributes.json was not rebuilt"
    assert not stale.is_file(), "stale chunk survived the rebuild"


def test_the_output_pyramid_matches_a_numpy_downsample(tmp_path, monkeypatch):
    """The default test store has no level 1, so nothing else here runs the pyramid write at
    all. A shifted, duplicated or zeroed band only shows up in a value check."""
    from make_store import volume
    from spotlight.formats import _SPEC, _in_order, canonical_view
    from spotlight.stores import _context, open_output_array, write_group_metadata
    from spotlight import stores

    fmt, setups, Z, y, x = "zarr2", (0,), 63, 48, 64
    store = tmp_path / "in"
    cfg = {"output_intensity_path": str(store), "output_format": fmt,
           "chunk_size": [32, 32, 32], "shard_size": [64, 64, 16]}
    order = _SPEC[fmt]["order"]
    ctx = _context()
    vol = volume(0, Z, y, x)
    # Level 1 is the mean-downsample of level 0, as a real input pyramid is; the stage only
    # reads its SHAPE (to derive the factors), but a lying level 1 would mislead anyone
    # debugging this test.
    half = (vol[: Z // 2 * 2].reshape(Z // 2, 2, y // 2, 2, x // 2, 2)
            .mean(axis=(1, 3, 5)).round().astype(np.uint16))
    for level, data in ((0, vol), (1, half)):
        arr, _, _ = open_output_array(
            cfg, 0, level, _in_order(data.shape, order), "uint16", ctx)
        canonical_view(arr, order)[:, :, :].write(data).result()
    write_group_metadata(cfg, 0, [(1, 1, 1), (2, 2, 2)])

    monkeypatch.chdir(tmp_path)
    config.set_config(
        input_basic_path=str(store), output_basic_path=str(store) + "_out",
        results_root=str(tmp_path / "results"), qstacks_dir=str(tmp_path / "qstacks"),
        input_format=fmt, last_setup=0, setups_per_camera=1,
        chunk_size=[32, 32, 32], shard_size=[64, 64, 16],
        lsf_project="p", output_stem=str(tmp_path / "o"),
        error_stem=str(tmp_path / "e"), n_cores_stats=2, n_cores_correction=2,
    )
    main(["run", "basic"])

    out = {**config.load_config(), "input_basic_path": str(store) + "_out"}
    got0 = np.asarray(stores.open_source(out, 0, 0)[:, :, :].read().result())
    got1 = np.asarray(stores.open_source(out, 0, 1)[:, :, :].read().result())
    # Odd Z: tensorstore rounds the level UP, so the last plane averages the one leftover
    # source plane on its own. Both halves are checked -- the tail is exactly where a slab
    # loop truncates.
    nz = Z // 2
    assert got1.shape == (nz + 1, y // 2, x // 2)
    binned = lambda a: a.reshape(-1, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))
    assert np.abs(got1[:nz].astype(np.int32) - binned(got0[:nz * 2])).max() <= 1
    tail = got0[nz * 2:].reshape(1, 1, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))
    assert np.abs(got1[nz:].astype(np.int32) - tail).max() <= 1


def test_the_apply_basic_banner_does_not_contradict_the_correct_stage(capsys, monkeypatch):
    """`run basic` prints apply_basic=False and then applies BaSiC, which reads as a bug.

    The flag is right -- stats/qstack must fit the fields from RAW voxels -- so the banner
    is what has to say so. Pins that it names `mode`, which is what actually decides.
    """
    from spotlight import correct, local
    monkeypatch.setattr(local, "_units", lambda cfg, stage, mode: [])
    local.run_pipeline({"apply_basic": True}, "basic", dry_run=True)
    out = capsys.readouterr().out
    assert "apply_basic=False" in out
    assert "for every stage" not in out
    assert "mode=basic" in out
    # and the stage really does correct, despite that False
    view = correct._view({"apply_basic": False, "input_basic_path": "",
                          "output_basic_path": ""}, "basic")
    assert view["apply_basic"] is True
