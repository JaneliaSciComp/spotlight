"""Mode selection for the unified correction stage.

Pure resolution logic -- no stores. The point is that asking for something that cannot be
done says so, naming the stage to run, rather than half-correcting.
"""

import json

import pytest

from spotlight import correct


def _cfg(tmp_path):
    return {
        "results_root": str(tmp_path / "results"),
        "input_basic_path": str(tmp_path / "raw.zarr"),
        "output_basic_path": str(tmp_path / "basic.zarr"),
        "input_intensity_path": str(tmp_path / "raw.zarr"),
        "output_intensity_path": str(tmp_path / "int.zarr"),
        "last_setup": 2, "setups_per_camera": 3, "setup_ids": [],
        "input_format": "zarr2", "output_format": "zarr2",
    }


def _write_fields(tmp_path, camera=0):
    d = tmp_path / "results" / f"camera{camera + 1}"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("Flat-field.tif", "Dark-field.tif"):
        (d / name).write_bytes(b"")          # resolve_mode only checks existence
    return d


def _write_target(tmp_path):
    d = tmp_path / "results"
    d.mkdir(parents=True, exist_ok=True)
    (d / "intensity_target.json").write_text(json.dumps(
        {"target_mean": 100.0, "target_std": 10.0, "apply_basic": False, "setups": {}}))


def test_auto_picks_basic_when_only_fields_exist(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fields(tmp_path)
    assert correct.resolve_mode(cfg, 0, "auto") == "basic"


def test_auto_picks_intensity_when_only_the_target_exists(tmp_path):
    cfg = _cfg(tmp_path)
    _write_target(tmp_path)
    assert correct.resolve_mode(cfg, 0, "auto") == "intensity"


def test_auto_picks_both_when_both_exist(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fields(tmp_path)
    _write_target(tmp_path)
    assert correct.resolve_mode(cfg, 0, "auto") == "both"


def test_auto_with_nothing_available_names_both_stages(tmp_path):
    with pytest.raises(RuntimeError, match="basic.*int-aggregate|int-aggregate.*basic"):
        correct.resolve_mode(_cfg(tmp_path), 0, "auto")


@pytest.mark.parametrize("mode,missing", [("basic", "fields"), ("both", "fields")])
def test_explicit_mode_without_fields_says_to_run_basic(tmp_path, mode, missing):
    cfg = _cfg(tmp_path)
    _write_target(tmp_path)
    with pytest.raises(RuntimeError, match="spotlight basic"):
        correct.resolve_mode(cfg, 0, mode)


@pytest.mark.parametrize("mode", ["intensity", "both"])
def test_explicit_mode_without_a_target_says_to_aggregate(tmp_path, mode):
    cfg = _cfg(tmp_path)
    _write_fields(tmp_path)
    with pytest.raises(RuntimeError, match="aggregate"):
        correct.resolve_mode(cfg, 0, mode)


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode must be one of"):
        correct.resolve_mode(_cfg(tmp_path), 0, "flatfield")


# ─── which paths each mode reads and writes ───────────────────────────────────


def test_basic_mode_uses_the_basic_paths(tmp_path):
    v = correct._view(_cfg(tmp_path), "basic")
    assert v["input_intensity_path"].endswith("raw.zarr")
    assert v["output_intensity_path"].endswith("basic.zarr")
    assert v["apply_basic"] is True


def test_intensity_mode_uses_the_intensity_paths_and_no_basic(tmp_path):
    v = correct._view(_cfg(tmp_path), "intensity")
    assert v["output_intensity_path"].endswith("int.zarr")
    assert v["apply_basic"] is False


def test_both_mode_requires_the_raw_store(tmp_path):
    """The flat/dark correction is applied while reading, so pointing the intensity input
    at a previous basic output would apply the fields twice -- silently, and the result
    looks plausible."""
    cfg = _cfg(tmp_path)
    cfg["input_intensity_path"] = cfg["output_basic_path"]   # the corrected store
    with pytest.raises(RuntimeError, match="must be the RAW store"):
        correct._view(cfg, "both")


def test_both_mode_accepts_matching_inputs(tmp_path):
    v = correct._view(_cfg(tmp_path), "both")
    assert v["apply_basic"] is True
    assert v["input_intensity_path"] == v["input_intensity_path"]


def test_int_apply_is_an_alias_for_correct(tmp_path, monkeypatch):
    """There is one correction implementation; the old stage name reaches it."""
    from spotlight import __main__ as m
    seen = {}
    monkeypatch.setattr("spotlight.correct.apply_correction_chunked",
                        lambda cfg, setup, mode: seen.update(setup=setup, mode=mode))
    monkeypatch.setattr("spotlight.config.load_config", lambda: _cfg(tmp_path))
    m.main(["int-apply", "7"])
    assert seen == {"setup": 7, "mode": "auto"}


# ─── emptiness pass 1: the pooled Otsu sample must not scale with tile count ──


def test_otsu_sample_is_capped(tmp_path, monkeypatch):
    """Pass 1 pools a subsample of every tile and holds them all at once to concatenate.
    Uncapped, that grows with the mosaic -- the same pile-up pass 2 documents avoiding.
    """
    import numpy as np
    from make_store import write_store
    from spotlight import emptiness

    store = write_store(tmp_path / "in", "zarr2", setups=(0, 1, 2, 3))
    cfg = {**store, "results_root": str(tmp_path / "results")}

    monkeypatch.setattr(emptiness, "OTSU_SAMPLE_VOXELS", 400)
    sizes = []
    real = np.concatenate

    def spy(arrays, *a, **k):
        out = real(arrays, *a, **k)
        sizes.append(out.size)
        return out

    monkeypatch.setattr(emptiness.np, "concatenate", spy)
    empty, threshold, level, bg, phi = emptiness._empty_areas(cfg, [0, 1, 2, 3])

    assert sizes, "pass 1 never pooled anything"
    assert sizes[0] <= 400, f"pooled {sizes[0]} voxels against a 400 cap"
    # Still a usable threshold from the capped sample.
    assert 0 < threshold < 4096
    assert set(empty) == {0, 1, 2, 3}
    assert all(0.0 <= v <= 1.0 for v in empty.values())


def test_otsu_sample_uses_every_tile(tmp_path, monkeypatch):
    """Capping must not silently drop tiles -- the threshold is meant to be dataset-wide,
    so every tile has to contribute even when each one's share is tiny."""
    from make_store import write_store
    from spotlight import emptiness

    store = write_store(tmp_path / "in", "zarr2", setups=(0, 1, 2, 3))
    cfg = {**store, "results_root": str(tmp_path / "results")}
    monkeypatch.setattr(emptiness, "OTSU_SAMPLE_VOXELS", 8)   # 2 voxels per tile
    seen = []
    real = emptiness._read_tile
    monkeypatch.setattr(emptiness, "_read_tile",
                        lambda c, s, l: seen.append(s) or real(c, s, l))
    emptiness._empty_areas(cfg, [0, 1, 2, 3])
    assert set(seen) == {0, 1, 2, 3}


# ─── the emptiness stage is measured once, not on every script regeneration ───


def _stats_json(tmp_path, setup, **fields):
    d = tmp_path / "results" / "intensity_stats"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"setup{setup}.json").write_text(json.dumps(fields))


def test_emptiness_is_skipped_when_already_measured(tmp_path, monkeypatch, capsys):
    """It rescans every tile. On a large mosaic that is real work to reproduce numbers
    already on disk, and both script generators would otherwise redo it every call."""
    from spotlight import scripts
    cfg = {**_cfg(tmp_path), "last_setup": 2, "setups_per_camera": 3}
    for s in (0, 1, 2):
        _stats_json(tmp_path, s, empty_area=0.01, empty_threshold=170.0)

    ran = []
    monkeypatch.setattr(scripts, "measure_emptiness", lambda c: ran.append(1))
    scripts.ensure_emptiness(cfg)
    assert ran == [], "re-measured despite the values being present"
    assert "skipping" in capsys.readouterr().out

    scripts.ensure_emptiness(cfg, force=True)
    assert ran == [1], "force=True must re-measure"


@pytest.mark.parametrize("missing", ["empty_area", "empty_threshold", "whole file"])
def test_emptiness_runs_when_any_tile_lacks_it(tmp_path, monkeypatch, missing):
    """Partial measurements are the dangerous case: `aggregate` fails on the ONE tile
    that is missing it, after the array job has already run."""
    from spotlight import scripts
    cfg = {**_cfg(tmp_path), "last_setup": 2, "setups_per_camera": 3}
    full = dict(empty_area=0.01, empty_threshold=170.0)
    for s in (0, 1):
        _stats_json(tmp_path, s, **full)
    if missing != "whole file":
        _stats_json(tmp_path, 2, **{k: v for k, v in full.items() if k != missing})

    ran = []
    monkeypatch.setattr(scripts, "measure_emptiness", lambda c: ran.append(1))
    scripts.ensure_emptiness(cfg)
    assert ran == [1], f"did not re-measure when {missing} was absent"


# ─── log directories, and where the emptiness map lives ───────────────────────


def test_log_directories_are_created(tmp_path, monkeypatch):
    """LSF does not create them, and a missing one does not stop the job -- it runs to
    completion and then cannot write its log, so the work is done and the output is
    gone."""
    from spotlight import config, scripts
    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "deep" / "nested" / "output"
    config.set_config(results_root=str(tmp_path / "r"), last_setup=2,
                      setups_per_camera=3, lsf_project="p",
                      output_stem=str(logs / "output"),
                      error_stem=str(tmp_path / "deep" / "err" / "error"),
                      input_basic_path="/in.zarr", output_basic_path="/out.zarr")
    assert not logs.exists()
    scripts.write_correction_script(config.load_config())
    assert logs.is_dir(), "output_stem's parent was not created"
    assert (tmp_path / "deep" / "err").is_dir(), "error_stem's parent was not created"


def test_a_stem_ending_in_a_separator_is_itself_the_log_directory(tmp_path, monkeypatch):
    """`.../out/` means LSF writes `.../out/_correct.txt`, one level deeper than the
    stem's `.parent` -- so that trailing-slash directory has to be created too."""
    from spotlight import config, scripts
    monkeypatch.chdir(tmp_path)
    config.set_config(results_root=str(tmp_path / "r"), last_setup=2,
                      setups_per_camera=3, lsf_project="p",
                      output_stem=f"{tmp_path / 'logs' / 'out'}/",
                      error_stem=f"{tmp_path / 'logs' / 'err'}/",
                      input_basic_path="/in.zarr", output_basic_path="/out.zarr")
    scripts.write_correction_script(config.load_config())
    assert (tmp_path / "logs" / "out").is_dir(), "trailing-slash output dir not created"
    assert (tmp_path / "logs" / "err").is_dir(), "trailing-slash error dir not created"


def test_empty_fraction_map_lives_in_the_camera_folder(tmp_path):
    from spotlight.config import empty_fraction_path
    cfg = {"results_root": str(tmp_path)}
    assert empty_fraction_path(cfg, 0) == tmp_path / "camera1" / "empty_fraction.tif"
    assert empty_fraction_path(cfg, 3) == tmp_path / "camera4" / "empty_fraction.tif"


def test_old_empty_fraction_location_is_still_read(tmp_path, capsys):
    """Results directories written before the map moved must keep working, rather than
    silently losing it and forcing a full re-measure."""
    import numpy as np, tifffile
    from spotlight import qstack
    cfg = {"results_root": str(tmp_path), "input_format": "zarr2"}
    phi = np.full((8, 8), 0.25, np.float32)
    tifffile.imwrite(str(tmp_path / "basic_empty_fraction_camera1.tif"), phi)
    got = qstack.empty_fraction_map(cfg, 0, (8, 8))
    assert got is not None and float(got.mean()) == pytest.approx(0.25)
    assert "old location" in capsys.readouterr().out
