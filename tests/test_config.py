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
                      last_setup=8, setups_per_camera=9, input_format="zarr3")
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
        'input_format = "n5"\n'
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
    assert cfg["max_concurrent_cores"] == 2000
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


def test_a_broken_toml_names_the_file_it_read(tmp_path, monkeypatch):
    """The traceback tomllib produces on its own says neither which file nor which
    directory, so a run from the wrong directory is indistinguishable from a corrupt toml --
    and a byte-order mark, which reads as an error at line 1 column 1, looks like neither."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "LocalPreferences.toml").write_bytes(
        b"\xef\xbb\xbf[spotlight]\ninput_basic_path = \"/x\"\n")     # BOM ahead of the table
    with pytest.raises(ValueError) as e:
        config.load_config()
    assert str(tmp_path / "LocalPreferences.toml") in str(e.value)
    assert "byte-order mark" in str(e.value)


def test_stage_cores_resolves_for_every_stage_that_asks(monkeypatch):
    """A mode missing from the map, or a key DEFAULTS lacks, raises inside the stage rather
    than at submit time."""
    from spotlight import correct
    for mode in correct.MODES:
        if mode != "auto":                        # resolved to one of the others before use
            assert mode in config.CORES_KEY, f"mode {mode!r} has no core count"
    for stage, key in config.CORES_KEY.items():
        assert key in config.DEFAULTS, f"{stage} -> {key}, which is not in DEFAULTS"
        assert isinstance(config.stage_cores({}, stage), int)

    assert config.stage_cores({"n_cores_int_correct": 7}, "both") == 7     # cfg wins
    assert config.stage_cores({}, "both") == config.DEFAULTS["n_cores_int_correct"]
    # The two correction pipelines genuinely differ, which is the reason for the map.
    assert config.CORES_KEY["basic"] != config.CORES_KEY["both"]


def test_windows_roots_reach_tensorstore_with_forward_slashes(tmp_path, monkeypatch):
    """Every kvstore path is a `/`-joined suffix on a configured root, so a Windows root
    has to be normalised at the root or the key comes out mixed-separator. Driven through
    `_input_location`/`stats_array_path` rather than `_slashes` directly -- those are what
    tensorstore is actually handed, and testing the helper alone would not catch a root
    that never goes through it."""
    from spotlight import stores
    from spotlight.formats import _input_location

    monkeypatch.setattr(config.os, "sep", "\\")          # pretend to be Windows
    monkeypatch.chdir(tmp_path)
    config.set_config(input_basic_path=r"C:\data\exp.n5", output_basic_path=r"C:\out.n5",
                      results_root=r"\\prfs\lab\res", last_setup=0, input_format="n5")
    cfg = config.load_config()

    assert cfg["input_basic_path"] == "C:/data/exp.n5"
    assert cfg["results_root"] == "//prfs/lab/res"       # UNC keeps its double separator
    path, _ = _input_location(config.basic_view(cfg), 0, 0)
    assert "\\" not in path


def test_the_stats_array_kvstore_path_is_posix(monkeypatch, tmp_path):
    """The one kvstore path built by `Path` joining rather than by `/`-joining a root, so
    `config._slashes` cannot reach it -- on Windows `str()` would hand tensorstore
    `C:\\res\\camera1\\minima\\s0`. `PureWindowsPath` is what makes that reachable from a
    POSIX host; without it the mutation `.as_posix()` -> `str()` survives."""
    from pathlib import PureWindowsPath
    from spotlight import stores

    seen = {}
    monkeypatch.setattr(stores, "Path", PureWindowsPath)
    monkeypatch.setattr(stores, "_open", lambda spec, ctx, **kw: seen.update(spec))
    stores.open_stats_array({"results_root": r"C:\res", "chunk_size": [32, 32, 32]},
                            0, "minima", (64, 64))
    assert seen["kvstore"]["path"] == "C:/res/camera1/minima/s0"


def test_slashes_is_a_noop_off_windows(monkeypatch):
    """`\\` is a legal filename character on Linux and macOS, so normalising there would
    corrupt a real path rather than fix one."""
    monkeypatch.setattr(config.os, "sep", "/")
    assert config._slashes(r"/data/weird\name.n5") == r"/data/weird\name.n5"
