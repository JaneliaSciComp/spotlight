"""`copy` / `copy-basic`: rewrite a tile into the corrected layout, applying nothing.

The channels of a dataset that need no correction still have to live in the corrected
output store, or the dataset.xml the corrected tiles are indexed by cannot resolve them.
So the claim under test is narrow and total: same voxels, corrected layout.

Mutation-checked: returning early from `resolve_mode` for the wrong modes, dropping the
`copy-basic` branch of `_view`, letting a copy default to every setup, skipping the xml
check, and pointing the cluster array at the config's setup list instead of the named
tiles all fail at least one of these.
"""

import json
import os
import subprocess

import numpy as np
import pytest

from make_store import DEPTHS, X, Y, volume, write_store

from spotlight import config, correct, local, scripts
from spotlight.formats import _SPEC, canonical_view
from spotlight.stores import _context
from spotlight.__main__ import main

import tensorstore as ts


XML = """<?xml version="1.0"?>
<SpimData version="0.2"><SequenceDescription><ViewSetups>
{setups}
</ViewSetups></SequenceDescription>
<ViewRegistrations/><MissingViews>{missing}</MissingViews></SpimData>
"""


def _xml(path, ids, missing=()):
    path.write_text(XML.format(
        setups="\n".join(f"<ViewSetup><id>{i}</id><size>64 48 63</size></ViewSetup>"
                         for i in ids),
        missing="".join(f'<MissingView timepoint="0" setup="{i}"/>' for i in missing)))
    return str(path)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """A store with setups 0-2, and an xml that also knows about 3 (absent) and 4 (missing)."""
    store = write_store(tmp_path / "in", "zarr2", setups=(0, 1, 2))
    monkeypatch.chdir(tmp_path)
    config.set_config(
        input_basic_path=store["input_basic_path"],
        output_basic_path=store["output_basic_path"],
        input_intensity_path=store["input_intensity_path"],
        output_intensity_path=store["output_intensity_path"],
        results_root=str(tmp_path / "results"), qstacks_dir=str(tmp_path / "qstacks"),
        input_format="zarr2", output_format="zarr2", last_setup=2, setups_per_camera=3,
        chunk_size=[32, 32, 32], shard_size=[64, 64, 64],
        dataset_xml=_xml(tmp_path / "dataset.xml", (0, 1, 2, 4), missing=(4,)),
        lsf_project="p", output_stem=str(tmp_path / "o"),
        error_stem=str(tmp_path / "e"), n_cores_int_correct=2,
    )
    return tmp_path


def _read(root, setup, fmt="zarr2", level=0):
    arr = ts.open({"driver": _SPEC[fmt]["driver"],
                   "kvstore": {"driver": "file", "path": f"{root}/s{setup}-t0.zarr/{level}"}},
                  context=_context(), open=True, read=True).result()
    return np.asarray(canonical_view(arr, _SPEC[fmt]["order"])[:, :, :].read().result())


# ─── the claim: identical voxels, corrected layout ───────────────────────────

def test_a_copy_writes_the_input_voxels_unchanged(experiment):
    """Not 'close' -- IDENTICAL. A copy that quietly rescaled would make the uncorrected
    channels disagree with the raw data in a way nothing downstream could detect.
    """
    main(["run", "copy", "1"])
    out = _read(str(experiment / "in_out"), 1)
    assert out.shape == (DEPTHS[1], Y, X)
    np.testing.assert_array_equal(out, volume(1, DEPTHS[1]))


def test_a_copy_lands_in_the_corrected_layout_not_a_raw_recreation(experiment):
    """The CONFIGURED chunking and the multiscales metadata, not the input store's -- the
    copied channels sit in the same store as the corrected ones and are read by the same
    viewer, so a tile chunked like the raw data is the failure worth catching.
    """
    main(["run", "copy", "0"])
    copied = experiment / "in_out" / "s0-t0.zarr"
    meta = json.loads((copied / "0" / ".zarray").read_text())
    # 5-D (t, c, z, y, x) with the configured chunk on the spatial axes -- the corrected
    # store's shape, not a bare 3-D recreation of the input.
    assert meta["chunks"] == [1, 1, 32, 32, 32], meta
    assert meta["shape"][:2] == [1, 1] and meta["shape"][2:] == [DEPTHS[0], Y, X], meta
    assert meta["dtype"].endswith("u2"), meta
    attrs = json.loads((copied / ".zattrs").read_text())
    assert "multiscales" in attrs, attrs
    assert len(attrs["multiscales"][0]["datasets"]) == 1, attrs


def test_a_copy_needs_neither_the_fields_nor_the_target(experiment):
    """The point of the mode. Requiring them would make copying an uncorrected channel
    depend on a correction it never applies -- and on this experiment neither exists.
    """
    assert not (experiment / "results" / "camera1" / "Flat-field.tif").exists()
    assert not (experiment / "results" / "intensity_target.json").exists()
    for mode in correct.COPY_MODES:
        assert correct.resolve_mode({}, 0, mode) == mode
    with pytest.raises(RuntimeError, match="nothing to correct"):
        correct.resolve_mode({"results_root": str(experiment / "results"),
                              "setups_per_camera": 3, "last_setup": 2, "setup_ids": []},
                             0, "auto")


@pytest.mark.parametrize("mode,key", [("copy", "output_intensity_path"),
                                      ("copy-basic", "output_basic_path")])
def test_the_two_copy_modes_differ_only_in_which_configured_pair_they_use(mode, key):
    cfg = {"input_basic_path": "/b/in", "output_basic_path": "/b/out",
           "input_intensity_path": "/i/in", "output_intensity_path": "/i/out"}
    view = correct._view(cfg, mode)
    assert view["output_intensity_path"] == cfg[key]
    assert view["apply_basic"] is False       # a copy applies no flat/dark either


# ─── tile selection ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("words,want", [
    (["126"], [126]),
    (["200-203"], [200, 201, 202, 203]),
    (["1,2,7-9"], [1, 2, 7, 8, 9]),
    (["5", "5", "4-5"], [4, 5]),             # deduped and sorted, so a tile copies once
    ([], []),
])
def test_setup_ranges_are_inclusive_deduped_and_sorted(words, want):
    assert config.parse_setups(words) == want


@pytest.mark.parametrize("bad", ["abc", "5-1", "3-x"])
def test_an_unreadable_range_is_refused(bad):
    with pytest.raises(SystemExit):
        config.parse_setups([bad])


def test_a_copy_with_no_tiles_is_refused_rather_than_copying_everything(experiment):
    """The dangerous default: `copy` over the whole setup list would overwrite the
    corrected channels with uncorrected voxels.
    """
    with pytest.raises(SystemExit, match="tiles to copy"):
        main(["run", "copy"])
    with pytest.raises(SystemExit, match="tiles to copy"):
        main(["run", "copy", "--cluster"])


@pytest.mark.parametrize("tile,why", [(3, "not ViewSetups"), (4, "MissingViews")])
def test_a_tile_the_xml_does_not_offer_is_refused_before_any_work(experiment, tile, why):
    """A typo in a hand-entered range is otherwise 200 array elements each dying on a store
    that was never written -- after the array has been submitted.
    """
    with pytest.raises(SystemExit, match=why):
        main(["run", "copy", str(tile), "--cluster"])
    assert not (experiment / "bsub_pipeline_copy.sh").exists()


# ─── the cluster script ──────────────────────────────────────────────────────

@pytest.mark.skipif(os.name == "nt", reason="drives bash; the cluster is always Linux")
def test_the_cluster_script_covers_exactly_the_named_tiles(experiment):
    """The array must index the NAMED tiles, not the config's setup list -- `last_setup` is
    2 here while the copy targets 0 and 2, so a wrong selector still produces a valid-looking
    3-element array.
    """
    main(["run", "copy", "0", "2", "--cluster"])
    text = (experiment / "bsub_pipeline_copy.sh").read_text()
    assert subprocess.run(["bash", "-n"], input=text, text=True).returncode == 0
    line = next(l for l in text.splitlines() if l.startswith("J_CORRECT="))
    assert "S=(0 2);" in line, line
    assert "[1-2]" in line, line
    assert "--mode copy" in line, line
    assert "spotlight-copy" in line, line     # its own name, not the correction's
    assert "_cp_%I.txt" in line, line         # and its own log namespace


def test_a_copy_run_submits_only_the_copy(experiment):
    """No emptiness, no stats, no aggregate: a copy measures nothing, and on a fresh dataset
    the emptiness pass would rescan every tile to produce numbers this run never reads.
    """
    main(["run", "copy", "1", "--cluster"])
    text = (experiment / "bsub_pipeline_copy.sh").read_text()
    assert local.PIPELINES["copy"] == ["correct"]
    for absent in ("J_STATS", "J_QSTACK", "J_BASIC", "J_INT_STATS", "J_INT_AGGREGATE"):
        assert absent not in text, absent
    assert text.count("jsub -J") == 1, text


def _write_real_fields(root, flat_value=2.0, dark_value=0.0):
    """Loadable flat/dark TIFFs at the level-0 frame, so `basic_model` would really apply
    them if the copy path let it. flat=2 halves every voxel -- unmissable.
    """
    import tifffile
    d = root / "results" / "camera1"
    d.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(d / "Flat-field.tif", np.full((Y, X), flat_value, np.float32))
    tifffile.imwrite(d / "Dark-field.tif", np.full((Y, X), dark_value, np.float32))
    return d


def test_a_copy_ignores_existing_basic_fields_and_an_apply_basic_toml(experiment):
    """The case that actually matters: the corrected channels of this dataset went through
    BaSiC, so the fields ARE on disk and the toml may well say apply_basic = true. A copy
    must still write the raw voxels -- `_view` forces the flag off for both copy modes, and
    with flat=2 a leak would halve every value.
    """
    _write_real_fields(experiment)
    config.set_config(apply_basic=True)
    assert config.load_config()["apply_basic"] is True
    for mode in correct.COPY_MODES:
        assert correct._view(config.load_config(), mode)["apply_basic"] is False

    main(["run", "copy", "2"])
    np.testing.assert_array_equal(_read(str(experiment / "in_out"), 2),
                                  volume(2, DEPTHS[2]))


def test_a_copy_ignores_the_apply_basic_environment_override(experiment):
    """`run <pipeline> --cluster` sets SPOTLIGHT_APPLY_BASIC in every job's environment, and
    it beats the toml by design -- so the copy path has to override it too, not inherit it.
    """
    _write_real_fields(experiment)
    os.environ["SPOTLIGHT_APPLY_BASIC"] = "1"
    try:
        main(["run", "copy", "2"])
    finally:
        del os.environ["SPOTLIGHT_APPLY_BASIC"]
    np.testing.assert_array_equal(_read(str(experiment / "in_out"), 2),
                                  volume(2, DEPTHS[2]))


@pytest.mark.parametrize("tile,why", [(3, "not ViewSetups"), (4, "MissingViews")])
def test_the_local_route_checks_the_xml_too(experiment, tile, why):
    """Same guard on both routes: `run copy 3` locally must refuse before opening a store,
    not fail per tile inside the loop.
    """
    with pytest.raises(SystemExit, match=why):
        main(["run", "copy", str(tile)])
