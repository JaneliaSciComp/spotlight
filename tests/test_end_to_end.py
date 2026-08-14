"""The whole pipeline through the CLI, on a synthetic store.

stats -> qstack -> basic -> correct, invoked the way the generated bsub scripts invoke
it, so the argument plumbing and the on-disk names are exercised and not just the
library calls. Small enough to run in a couple of seconds.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from make_store import DEPTHS, X, Y, volume, write_store

from spotlight import config
from spotlight.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """A working directory with a store and a LocalPreferences.toml pointing at it."""
    store = write_store(tmp_path / "in", "zarr2", setups=(0, 1, 2))
    monkeypatch.chdir(tmp_path)
    config.set_config(
        input_basic_path=store["input_basic_path"],
        output_basic_path=store["output_basic_path"],
        results_root=str(tmp_path / "results"),
        qstacks_dir=str(tmp_path / "qstacks"),
        format="zarr2",
        last_setup=2,
        setups_per_camera=3,
        chunk_size=[32, 32, 32],
        shard_size=[64, 64, 64],
        chunks_per_job=2,
        lsf_project="testproj",
        output_stem=str(tmp_path / "out"),
        error_stem=str(tmp_path / "err"),
        n_cores_stats=2,
        n_cores_correction=2,
    )
    return tmp_path


def test_full_pipeline(experiment):
    main(["stats", "0", "1", "4"])

    qdir = experiment / "qstacks"
    main(["qstack"])
    tiff = qdir / "camera1.tiff"
    assert tiff.is_file()

    import tifffile
    pages = np.asarray(tifffile.imread(str(tiff)))
    assert pages.shape == (21, Y, X)          # 21 quantile planes, (Y, X) for zarr
    assert pages.dtype == np.uint16
    # Monotone in quantile at every pixel -- q000 <= q005 <= ... is what an order
    # statistic means, and it is the cheapest end-to-end sanity check there is.
    assert (np.diff(pages.astype(np.int32), axis=0) >= 0).all()

    main(["basic", "0"])
    flat_p = experiment / "results" / "camera1" / "Flat-field.tif"
    dark_p = experiment / "results" / "camera1" / "Dark-field.tif"
    assert flat_p.is_file() and dark_p.is_file()
    flat = np.asarray(tifffile.imread(str(flat_p)))
    assert flat.shape == (Y, X)               # published at the level-0 frame
    assert flat.dtype == np.float32
    assert float(flat.mean()) == pytest.approx(1.0, rel=1e-4)

    main(["correct", "0"])
    from spotlight import stores
    cfg = config.load_config()
    out = stores.open_source({**cfg, "input_basic_path": cfg["output_basic_path"]}, 0, 0)
    got = np.asarray(out[:, :, :].read().result())
    assert got.shape == (DEPTHS[0], Y, X)
    dark = np.asarray(tifffile.imread(str(dark_p)))
    from spotlight.correct import FLAT_FLOOR
    want = np.round(np.clip((volume(0, DEPTHS[0]) - dark) / np.maximum(flat, FLAT_FLOOR),
                            0, 65535)).astype(np.uint16)
    # Within one gray level everywhere -- see test_stores.py for why the kernel folds the
    # arithmetic. No bound on HOW MANY voxels differ here, unlike test_stores: this flat
    # field comes from a real fit on tiny synthetic data, which leaves ~23% of the frame
    # at or below zero and therefore floored. Those pixels sit where `1/flat` is largest
    # and the folded form is least precise, so the differing fraction says more about the
    # degenerate fit than about the kernel. The max is the invariant that matters.
    diff = np.abs(got.astype(np.int32) - want.astype(np.int32))
    assert diff.max() <= 1, f"{diff.max()} counts, {(diff != 0).mean():.2%} of voxels"


def test_generated_scripts(experiment):
    main(["submit", "stats"])
    main(["submit", "correct"])
    main(["submit", "intensity"])

    stats_sh = (experiment / "bsub_command.sh").read_text()
    # 4 chunks at chunks_per_job=2 -> 2 array elements, one camera.
    assert 'bsub -J "spotlight-stats[1-2]%100"' in stats_sh
    assert "-n 2 -P testproj" in stats_sh
    assert "python -m spotlight stats 0" in stats_sh
    # The shell arithmetic must survive quoting -- these expand in the JOB's shell, and
    # a script that expanded them at generation time would give every element the same
    # chunk range.
    assert "$LSB_JOBINDEX" in stats_sh
    # The manifest path, so the job uses THIS checkout's environment. `spotlight` itself
    # comes from that environment (installed editable), not from PYTHONPATH.
    assert f"--manifest-path {ROOT}" in stats_sh
    assert "PYTHONPATH" not in stats_sh

    corr_sh = (experiment / "bsub_correction.sh").read_text()
    assert 'bsub -J "spotlight-correct[1-3]%100"' in corr_sh     # last_setup=2 -> 3 setups
    # Pinned, not `auto`: a stale intensity_target.json must not silently turn the
    # flat/dark stage into a joint one.
    assert "python -m spotlight correct $(($LSB_JOBINDEX-1)) --mode basic" in corr_sh

    for name, stage, array in (("stats", "int-stats", True),
                               ("aggregate", "int-aggregate", False),
                               ("correct", "int-apply", True)):
        text = (experiment / f"bsub_int_{name}.sh").read_text()
        assert f"python -m spotlight {stage}" in text
        assert ("[1-3]" in text) == array


def test_generated_scripts_with_explicit_setup_ids(experiment):
    """Non-contiguous ids cannot be derived from the array index, so the script carries
    them in a shell array and indexes into it."""
    config.set_config(setup_ids=[[171, 172], [201, 202, 203]])
    main(["submit", "correct"])
    text = (experiment / "bsub_correction.sh").read_text()
    assert "S=(171 172 201 202 203)" in text
    assert 'bsub -J "spotlight-correct[1-5]%100"' in text
    assert "${S[$(($LSB_JOBINDEX-1))]}" in text


def test_module_entry_point_runs(experiment):
    """`python -m spotlight` must work with nothing but the package importable.

    Run from the repo root with a bare environment -- no PYTHONPATH -- so this checks the
    same thing the generated bsub lines rely on: that `python -m spotlight` resolves
    without any path plumbing. (cwd is on sys.path, which stands in for the editable
    install the environment provides on the cluster.)
    """
    r = subprocess.run([sys.executable, "-m", "spotlight", "--help"],
                       capture_output=True, text=True, cwd=ROOT,
                       env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    for stage in ("stats", "qstack", "basic", "correct", "emptiness", "submit"):
        assert stage in r.stdout


def test_background_quantile_partials_are_cleared_on_resubmit(experiment):
    """They are summed blind, so a stale partial from a previous run would skew the
    profile rather than fail."""
    from spotlight.quantiles import background_quantile_dir
    cfg = config.load_config()
    d = background_quantile_dir(cfg, 0)
    d.mkdir(parents=True, exist_ok=True)
    (d / "job99.json").write_text(json.dumps({"sum": [0.0] * 21, "count": 1}))
    main(["submit", "stats"])
    assert not d.exists()
