"""The concurrency sweep harness, exercised for real.

The sweep's OUTPUT is a measurement with no right answer, so nothing here asserts which
concurrency wins. What is testable is the harness: that it pins the right environment
variable, honours the budget, runs every value the requested number of times, survives a
stage that fails, and picks the minimum out of the table it built.

The stage is stubbed, so these run in milliseconds and do not touch a store.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import sweep_concurrency as sweep


@pytest.fixture
def fake_stage(monkeypatch):
    """Replace the subprocess with a model whose wall time is a known function of
    concurrency, so the harness's arithmetic has a right answer to find.

    Modelled on what was actually measured: time falls as concurrency rises until the
    reads saturate, then flattens. A knee, not a peak.
    """
    calls = []

    def fake_run(stage, args, concurrency, budget=None):
        calls.append({"stage": stage, "args": args, "concurrency": concurrency,
                      "budget": budget})
        wall = 60.0 / min(concurrency, 16)          # saturates at 16
        return {"concurrency": concurrency, "t_wall": wall, "returncode": 0,
                "maxrss_bytes": concurrency * 2**30, "maxrss_highwater_bytes": 0,
                "bytes_in": 9 * 2**30, "t_read": wall * 3, "t_compute": wall}

    monkeypatch.setattr(sweep, "run_once", fake_run)
    return calls


def test_sweep_runs_every_value_the_requested_number_of_times(fake_stage, capsys):
    sweep.main(["stats", "0", "1", "64", "--values", "4,8,16", "--repeats", "3"])
    got = sorted(c["concurrency"] for c in fake_stage)
    assert got == sorted([4, 8, 16] * 3)
    assert all(c["stage"] == "stats" and c["args"] == ["0", "1", "64"]
               for c in fake_stage)


def test_sweep_reports_the_fastest_value(fake_stage, capsys):
    sweep.main(["stats", "0", "--values", "4,8,16,32", "--repeats", "2"])
    assert "fastest: concurrency 16" in capsys.readouterr().out


def test_sweep_names_the_knee_when_the_minimum_sits_above_it(monkeypatch, capsys):
    """The realistic shape: throughput saturates, then noise makes some higher value
    marginally "fastest" while costing twice the memory for nothing. Reporting the raw
    minimum there would talk you into 32; the knee is the number you want.
    """
    def model(stage, args, concurrency, budget=None):
        wall = 60.0 / min(concurrency, 16) * (1 - 0.001 * concurrency)   # tiny drift
        return {"concurrency": concurrency, "t_wall": wall, "returncode": 0,
                "maxrss_bytes": concurrency * 2**30, "bytes_in": 9 * 2**30}
    monkeypatch.setattr(sweep, "run_once", model)
    sweep.main(["stats", "0", "--values", "4,8,16,32", "--repeats", "1"])
    out = capsys.readouterr().out
    assert "fastest: concurrency 32" in out            # the raw minimum...
    assert "within 5% of the best from concurrency 16 onward" in out   # ...but use 16
    assert "prefer the lower one" in out


def test_sweep_stays_quiet_when_the_fastest_is_already_the_cheapest(fake_stage, capsys):
    """No advice to give when the minimum IS the knee -- an unconditional suggestion
    would just be noise."""
    sweep.main(["stats", "0", "--values", "4,8,16", "--repeats", "1"])
    assert "prefer the lower one" not in capsys.readouterr().out


def test_sweep_pins_the_budget_so_only_concurrency_varies(fake_stage, capsys):
    budget = 12 * 2**30
    sweep.main(["correct", "0", "--values", "1,2", "--repeats", "1",
                "--budget", str(budget)])
    assert {c["budget"] for c in fake_stage} == {budget}
    assert "pinned" in capsys.readouterr().out


def test_sweep_shuffles_by_default_so_the_page_cache_is_not_one_value_s_advantage(
        fake_stage):
    """Run in order, the first value reads from cold and looks slowest whatever it is."""
    sweep.main(["stats", "0", "--values", "1,2,4,8,16,32", "--repeats", "2", "--seed", "1"])
    order = [c["concurrency"] for c in fake_stage]
    assert order != sorted(order), "ran in ascending order; cold cache would bias value 1"


def test_sweep_ascending_order_is_available(fake_stage):
    sweep.main(["stats", "0", "--values", "1,2,4", "--repeats", "1",
                "--order", "ascending"])
    assert [c["concurrency"] for c in fake_stage] == [1, 2, 4]


def test_sweep_survives_a_value_that_fails(monkeypatch, capsys):
    """One concurrency OOMing must not lose the rest of the table -- that is often the
    result you wanted, since it brackets the ceiling."""
    def flaky(stage, args, concurrency, budget=None):
        if concurrency == 32:
            return {"concurrency": 32, "t_wall": 1.0, "returncode": 137,
                    "maxrss_bytes": 0, "tail": ["Killed"]}
        return {"concurrency": concurrency, "t_wall": 60.0 / concurrency,
                "returncode": 0, "maxrss_bytes": 2**30, "bytes_in": 2**30}
    monkeypatch.setattr(sweep, "run_once", flaky)
    rc = sweep.main(["stats", "0", "--values", "8,32", "--repeats", "1"])
    out = capsys.readouterr().out
    assert "ALL RUNS FAILED" in out and "Killed" in out
    assert "fastest: concurrency 8" in out
    assert rc == 0, "a partial table is still a result"


def test_sweep_writes_raw_records(fake_stage, tmp_path):
    out = tmp_path / "sweep.json"
    sweep.main(["stats", "0", "--values", "4,8", "--repeats", "1", "--out", str(out)])
    recs = json.loads(out.read_text())
    assert {r["concurrency"] for r in recs} == {4, 8}
    assert all("t_wall" in r for r in recs)


def test_run_once_pins_the_right_env_var_per_stage(monkeypatch):
    """`stats` and `correct` read different variables; pinning the wrong one would sweep
    nothing and silently report the default six times."""
    seen = {}

    class P:
        returncode = 0
        stdout = "SPOTLIGHT_TIMING " + json.dumps({"bytes_in": 1})

    def fake_subprocess(cmd, env=None, **kw):
        seen.update(env)
        seen["cmd"] = cmd
        return P()

    monkeypatch.setattr(sweep.subprocess, "run", fake_subprocess)
    sweep.run_once("stats", ["0"], 7)
    assert seen["SPOTLIGHT_STATS_CONCURRENCY"] == "7"
    sweep.run_once("correct", ["0"], 3, budget=99)
    assert seen["SPOTLIGHT_CORRECT_CONCURRENCY"] == "3"
    assert seen["SPOTLIGHT_MEMORY_BYTES"] == "99"
    assert seen["cmd"][1:4] == ["-m", "spotlight", "correct"]


# ─── running outside LSF ──────────────────────────────────────────────────────


def test_budget_off_cluster_reads_the_machine(monkeypatch):
    """No allocation to derive from, so use what the box has. A flat default is wrong in
    both directions -- far too small on a workstation, far too large on a laptop."""
    from spotlight import stores
    for k in ("SPOTLIGHT_MEMORY_BYTES", "LSB_CG_MEMLIMIT", "LSB_DJOB_NUMPROC"):
        monkeypatch.delenv(k, raising=False)
    assert stores.memory_budget() == int(stores._machine_memory() * stores.MEMORY_FRACTION)
    assert stores.memory_budget() < stores._machine_memory(), "must leave the box headroom"


def test_budget_falls_back_when_sysconf_is_unavailable(monkeypatch):
    from spotlight import stores
    monkeypatch.setattr(stores.os, "sysconf",
                        lambda *_: (_ for _ in ()).throw(ValueError("nope")))
    assert stores._machine_memory() == 8 * 2**30


def test_scratch_workdir_redirects_results_root(tmp_path, monkeypatch):
    """The sweep re-runs the stage a dozen times; pointed at a live results_root it would
    rewrite those chunks and the reference set stops being one."""
    import tomllib
    real = tmp_path / "real_results"
    (real / "intensity_stats").mkdir(parents=True)
    (real / "intensity_stats" / "setup0.json").write_text('{"empty_threshold": 170.0}')
    exp = tmp_path / "experiment"
    exp.mkdir()
    (exp / "LocalPreferences.toml").write_text(
        f'[spotlight]\nresults_root = "{real}"\nlast_setup = 0\n')
    monkeypatch.chdir(exp)

    scratch = tmp_path / "scratch"
    work = sweep._scratch_workdir(str(scratch))

    with open(work / "LocalPreferences.toml", "rb") as f:
        got = tomllib.load(f)["spotlight"]
    assert got["results_root"] == str(scratch)
    assert got["last_setup"] == 0, "the rest of the config must survive"
    # empty_threshold has to come across or the pass skips the background profile and
    # stops doing the work a real run does.
    assert (scratch / "intensity_stats" / "setup0.json").is_file()


def test_scratch_workdir_says_so_when_there_are_no_measurements(tmp_path, monkeypatch,
                                                                capsys):
    exp = tmp_path / "experiment"
    exp.mkdir()
    (exp / "LocalPreferences.toml").write_text(
        f'[spotlight]\nresults_root = "{tmp_path / "missing"}"\n')
    monkeypatch.chdir(exp)
    sweep._scratch_workdir(str(tmp_path / "scratch"))
    assert "skip the background profile" in capsys.readouterr().out


# ─── sweep_threads ────────────────────────────────────────────────────────────
#
# What it measures has no right answer to assert; a parser and an environment reset do, and
# both would corrupt every number in the table.

import sweep_threads as threads_sweep


@pytest.mark.parametrize("line, want", [
    ("1234 (python3.11) S 1 1234 1234 0 -1 4194560 12", "S"),
    ("1234 (ts_file_io) D 1 1234 1234 0 -1 4194560 12", "D"),
    # A comm containing spaces and parens is the case that breaks a naive split(): the
    # state is field 3, but `.split()[2]` here returns "worker)" and every thread reads as
    # neither R nor D, silently reporting a load of zero.
    ("1234 (my prog (v2)) R 1 1234 1234 0 -1 4194560 12", "R"),
])
def test_the_thread_state_is_read_past_the_executable_name(line, want):
    assert threads_sweep._state_char(line) == want


def test_an_arm_pins_every_knob_it_is_measuring(monkeypatch):
    """Arms run in one shell, so anything exported there would benchmark something other
    than the arm's name. `LSB_DJOB_NUMPROC` is included because the STAGE reads it too."""
    monkeypatch.setenv("SPOTLIGHT_IO_CONCURRENCY", "999")
    monkeypatch.setenv("SPOTLIGHT_COPY_CONCURRENCY", "999")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "8")

    table = threads_sweep.arms(64)
    legacy = threads_sweep.arm_env(table["legacy"], 64)
    assert legacy["SPOTLIGHT_IO_CONCURRENCY"] == "4096"      # the arm's, not the shell's
    assert legacy["SPOTLIGHT_COPY_CONCURRENCY"] == "64"
    assert legacy["LSB_DJOB_NUMPROC"] == "64"

    shipped = threads_sweep.arm_env(table["shipped"], 64)
    assert (shipped["SPOTLIGHT_IO_CONCURRENCY"],
            shipped["SPOTLIGHT_COPY_CONCURRENCY"]) == ("32", "64")

    # Unrelated environment still passes through -- the stage needs it to run at all.
    monkeypatch.setenv("SPOTLIGHT_GB_PER_SLOT", "15")
    assert threads_sweep.arm_env(table["legacy"], 64)["SPOTLIGHT_GB_PER_SLOT"] == "15"


def test_it_refuses_to_invent_the_slot_count(monkeypatch, capsys):
    """Every ratio and every arm's pool size derives from this number, so guessing it
    silently measures a configuration nobody asked for. See CLAUDE.md, "Gotchas"."""
    monkeypatch.delenv("LSB_DJOB_NUMPROC", raising=False)
    with pytest.raises(SystemExit) as e:
        threads_sweep.main(["correct", "0"])
    assert "LSB_DJOB_NUMPROC is not set" in str(e.value)
    assert "--slots" in str(e.value)
