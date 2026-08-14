"""Config round-trip, and the table rename."""

import os
from pathlib import Path

import pytest

from spotlight import config


def _cd(tmp_path):
    os.chdir(tmp_path)


def test_writes_the_new_table_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_config(input_basic_path="/data/x.zarr", results_root="/res",
                      last_setup=8, setups_per_camera=9, format="zarr3")
    text = Path("LocalPreferences.toml").read_text()
    assert "[spotlight]" in text
    assert "BigFlatFieldIlluminator" not in text
    cfg = config.load_config()
    assert cfg["input_basic_path"] == "/data/x.zarr"
    assert cfg["last_setup"] == 8


def test_reads_a_legacy_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("LocalPreferences.toml").write_text(
        "[BigFlatFieldIlluminator]\n"
        'input_basic_path = "/data/y.zarr"\n'
        'results_root = "/res"\n'
        "last_setup = 3\n"
        'format = "n5"\n'
    )
    cfg = config.load_config()
    assert cfg["input_basic_path"] == "/data/y.zarr"
    assert cfg["input_format"] == "n5"


def test_a_legacy_table_is_migrated_not_duplicated(tmp_path, monkeypatch):
    """Leaving both tables in place is the state where an edit lands in the one nothing
    reads, so the first write folds the old one in."""
    monkeypatch.chdir(tmp_path)
    Path("LocalPreferences.toml").write_text(
        '[BigFlatFieldIlluminator]\ninput_basic_path = "/data/y.zarr"\n'
        'results_root = "/res"\nlast_setup = 3\n')
    config.set_config(last_setup=5)
    text = Path("LocalPreferences.toml").read_text()
    assert "BigFlatFieldIlluminator" not in text
    cfg = config.load_config()
    assert cfg["last_setup"] == 5
    assert cfg["input_basic_path"] == "/data/y.zarr"      # the old keys survived


def test_defaults_match_the_julia_ones(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_config(results_root="/res", last_setup=0)
    cfg = config.load_config()
    assert cfg["chunk_size"] == [128, 128, 64]
    assert cfg["shard_size"] == [512, 512, 256]
    assert cfg["stats_scale"] == 2
    assert cfg["basic_stats_level"] == 0
    assert cfg["chunks_per_job"] == 64
    assert cfg["max_concurrent_jobs"] == 100
    assert cfg["qstacks_dir"] == "qstacks"
    assert cfg["basic_unmix_empty"] is False


def test_home_is_expanded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_config(results_root="/res", last_setup=0, output_stem="$HOME/out/output")
    assert config.load_config()["output_stem"] == os.path.expanduser("~") + "/out/output"


def test_basic_table_round_trips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_config(results_root="/res", last_setup=0)
    config.set_basic_config(estimate_darkfield=False, working_size=128)
    text = Path("LocalPreferences.toml").read_text()
    assert "[spotlight.basic]" in text
    p = config.basic_params(config.load_config())
    assert p["estimate_darkfield"] is False
    assert p["working_size"] == 128
    assert p["max_iterations"] == 500          # default still present


def test_a_basic_key_in_the_wrong_table_is_honoured(capsys, tmp_path, monkeypatch):
    """Silently ignoring it is worse: `override_darkfield = true` one level up reads back
    as false, and the run merely looks like it chose not to override."""
    monkeypatch.chdir(tmp_path)
    config.set_config(results_root="/res", last_setup=0, override_darkfield=True)
    p = config.basic_params(config.load_config())
    assert p["override_darkfield"] is True
    assert "override_darkfield" in capsys.readouterr().out


def test_camera_setups(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_config(results_root="/res", last_setup=8, setups_per_camera=3)
    cfg = config.load_config()
    assert config.camera_setups(cfg) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert config.num_cameras(cfg) == 3

    config.set_config(setup_ids=[[171, 172], [201, 202, 203]])
    cfg = config.load_config()
    assert config.camera_setups(cfg) == [[171, 172], [201, 202, 203]]
    assert config.num_cameras(cfg) == 2


def test_basic_view_rebinds_the_io_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_config(results_root="/res", last_setup=0,
                      input_basic_path="/in.zarr", output_basic_path="/out.zarr")
    v = config.basic_view(config.load_config())
    assert v["input_intensity_path"] == "/in.zarr"
    assert v["output_intensity_path"] == "/out.zarr"


def test_negative_stats_level_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_config(results_root="/res", last_setup=0, basic_stats_level=-1)
    with pytest.raises(ValueError, match="basic_stats_level"):
        config.load_config()
