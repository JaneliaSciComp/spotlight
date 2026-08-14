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
        format="zarr2", last_setup=2, setups_per_camera=3,
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
    """The check that fired is `_check_basic_mode(cfg, recorded, ...)` in the apply
    stage, comparing against `correct._view`'s `apply_basic = (mode == "both")`. If these
    two ever disagree the pipeline fails midway, after writing every earlier stage."""
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
