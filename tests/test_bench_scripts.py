"""The benchmark scripts, exercised as shell rather than inspected as strings.

The first version of these was generated but never run, and shipped a quoting bug: the
whole command sat inside double quotes, so `$LSB_JOBINDEX` expanded at SUBMIT time (to
nothing) and the job died on `$(( * 64 + 1 ))`. String assertions would not have caught
it -- the string looked right. So these tests run `run_arm.sh` for real, with a stub
`python` on PATH, and check the arguments it would have passed.
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_ARM = ROOT / "bench" / "run_arm.sh"


@pytest.fixture
def stub_python(tmp_path):
    """A `python` on PATH that reports its arguments and environment, and runs nothing."""
    d = tmp_path / "bin"
    d.mkdir()
    (d / "python").write_text(
        '#!/bin/bash\n'
        'echo "ARGS: $*"\n'
        'echo "SEM: ${SPOTLIGHT_CORRECT_CONCURRENCY:-unset}"\n'
        'echo "THREADS: ${JULIA_NUM_THREADS:-unset}/${OMP_NUM_THREADS:-unset}"\n'
    )
    (d / "python").chmod(0o755)
    return d


def _run(stub, index, *args, **env):
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "LSB_JOBINDEX": str(index)}
    e.update(env)
    r = subprocess.run(["bash", str(RUN_ARM), *args], capture_output=True, text=True,
                       env=e, stdin=subprocess.DEVNULL, timeout=60)
    assert r.returncode == 0, r.stderr
    out = {k: v.strip() for k, _, v in
           (l.partition(": ") for l in r.stdout.splitlines() if ": " in l)}
    out["argv"] = out["ARGS"].split()
    return out


@pytest.mark.parametrize("index,impl,start,stop", [
    (1, "python", 1, 64),      # pair 0
    (2, "julia", 1, 64),       # same chunk range as index 1 -- that pairing is the point
    (3, "python", 65, 128),    # pair 1
    (4, "julia", 65, 128),
    (5, "python", 129, 192),
])
def test_stats_arm_selection_and_arithmetic(stub_python, index, impl, start, stop):
    got = _run(stub_python, index, "stats", "0", "64")
    assert got["argv"][0].endswith("bench.py")
    assert got["argv"][1:] == [impl, "stats", "0", str(start), str(stop)]


def test_stats_pairs_share_a_chunk_range(stub_python):
    """Every julia element must have a python element on the same work, or `collect.py`
    has nothing to pair and the whole design is pointless."""
    seen = {}
    for index in range(1, 13):
        impl, _, _cam, start, stop = _run(stub_python, index, "stats", "0", "64")["argv"][1:]
        seen.setdefault((start, stop), set()).add(impl)
    assert all(v == {"julia", "python"} for v in seen.values()), seen
    assert len(seen) == 6


@pytest.mark.parametrize("index,impl,sem", [
    (1, "julia", "0"),
    (2, "python", "4"),
    (3, "python", "16"),
    (4, "python", "64"),
    (5, "julia", "0"),          # wraps back round for the next repeat
])
def test_correct_arm_selection(stub_python, index, impl, sem):
    got = _run(stub_python, index, "correct", "7",
               "julia:0", "python:4", "python:16", "python:64")
    assert got["argv"][1:] == [impl, "correct", "7"]
    # The semaphore limit reaches the job as an env var -- the knob the maxRSS sweep turns.
    assert got["SEM"] == sem


def test_thread_counts_are_pinned_from_the_lsf_allocation(stub_python):
    """Both arms drive the same tensorstore, so unpinned thread counts would benchmark
    tensorstore's defaults rather than the port."""
    got = _run(stub_python, 1, "stats", "0", "64", LSB_DJOB_NUMPROC="12")
    assert got["THREADS"] == "12/12"


def test_run_arm_is_valid_shell():
    assert subprocess.run(["bash", "-n", str(RUN_ARM)]).returncode == 0


def test_generated_bsub_lines_do_not_expand_job_variables(tmp_path, monkeypatch):
    """The generated line must keep `$LSB_JOBINDEX` out of the submitting shell's reach.

    Here that means it must not appear at all: the arithmetic moved into run_arm.sh
    precisely so no layer of quoting has to protect it.
    """
    from make_store import write_store
    from spotlight import config

    store = write_store(tmp_path / "in", "zarr2", setups=(0, 1, 2))
    monkeypatch.chdir(tmp_path)
    config.set_config(
        input_basic_path=store["input_basic_path"],
        output_basic_path=store["output_basic_path"],
        results_root=str(tmp_path / "results"), input_format="zarr2",
        last_setup=2, setups_per_camera=3, chunk_size=[32, 32, 32],
        shard_size=[64, 64, 64], chunks_per_job=2, lsf_project="testproj",
        output_stem=str(tmp_path / "out"), error_stem=str(tmp_path / "err"),
        n_cores_stats=2, n_cores_correction=2,
    )
    r = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "make_bench_scripts.py"),
         "--julia-project", "/some/where/BigFlatFieldIlluminator.jl"],
        capture_output=True, text=True,
        # PYTHONPATH here is HARNESS plumbing, not product behaviour: this runs the
        # generator from a temp directory, and until the package is installed editable
        # that is the only way it can import `spotlight`. Harmless once it is (same
        # code). What the test actually asserts is that the GENERATED scripts contain no
        # PYTHONPATH -- see below.
        env={**os.environ, "PYTHONPATH": str(ROOT)}, cwd=tmp_path)
    assert r.returncode == 0, r.stderr

    for name in ("bsub_bench_stats.sh", "bsub_bench_correct.sh"):
        text = (tmp_path / name).read_text()
        assert "$LSB_JOBINDEX" not in text, f"{name} still inlines the job index"
        assert "run_arm.sh" in text
        assert "PYTHONPATH" not in text, "the package comes from the environment now"
        # The command must be single-quoted, so nothing in it expands at submit time.
        assert text.count("'") == 2, text


def test_the_runner_does_not_cap_blas_pools():
    """Pinned as a decision, not an omission. The caps are the obvious-looking fix -- BLAS
    sizes its pool from the host, not the allocation -- but those threads sleep
    interruptibly and the load average ignores them. Measured at 30 slots on `correct`:
    1157 peak runnable-or-blocked uncapped against 1245 capped, while the tensorstore pool
    split alone reached 70. Re-measure with `bench/sweep_threads.py --arms legacy,blas-only`
    before putting them back.
    """
    from spotlight import scripts
    cmd = scripts.runner()
    assert "MALLOC_ARENA_MAX" in cmd, "the arena cap IS measured and must stay"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "OMP_WAIT_POLICY"):
        assert var not in cmd, f"{var} moved no load number; see bench/sweep_threads.py"
