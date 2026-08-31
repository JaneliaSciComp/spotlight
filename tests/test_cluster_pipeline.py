"""`run <pipeline> --cluster`: one script, every stage chained on LSF job ids.

The chain is the whole feature, so what these pin is that it cannot come apart quietly:
a stage that does not wait, a `-w` expression the shell never expands, a job id that
failed to parse. Each of those submits successfully and then does the wrong thing hours
later, which is exactly the failure mode a fire-and-forget script has to rule out.

Mutation-checked: dropping `dep=dep` from `_stage_jobs`, `shlex.quote`-ing the `-w`
expression instead of double-quoting it, dropping the SPOTLIGHT_APPLY_BASIC prefix,
reversing `local.PIPELINES["both"]`, and removing `jsub`'s `exit 1` on a bad id all fail
at least one of these.
"""

import os
import subprocess

import pytest

from spotlight import config, local, scripts

pytestmark = pytest.mark.skipif(os.name == "nt",
                                reason="drives bash; the cluster is always Linux")

# The variable each stage's id lands in, so a test can name the edge it expects.
VAR = {"stats": "J_STATS_0", "qstack": "J_QSTACK", "basic": "J_BASIC",
       "int-stats": "J_INT_STATS", "int-aggregate": "J_INT_AGGREGATE",
       "correct": "J_CORRECT"}


@pytest.fixture
def generate(tmp_path, monkeypatch):
    """Write a pipeline script into a tmp cwd, with the two stages that need a real store
    stubbed out: `_stats_prep` reads the source pyramid and creates arrays on disk, and
    `ensure_emptiness` rescans every tile. Both are covered by their own tests.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scripts, "ensure_emptiness", lambda cfg, **kw: None)
    monkeypatch.setattr(scripts, "_stats_prep", lambda cfg: ([0, 1], 5, 64))
    monkeypatch.setattr(scripts._config, "num_cameras", lambda cfg: 2)

    def go(pipeline, **over):
        cfg = dict(config.DEFAULTS, lsf_project="p", last_setup=3,
                   output_stem=str(tmp_path / "out"), error_stem=str(tmp_path / "err"),
                   **over)
        scripts.write_pipeline_script(cfg, pipeline)
        return (tmp_path / f"bsub_pipeline_{pipeline}.sh").read_text()
    return go


@pytest.mark.parametrize("pipeline", ["basic", "intensity", "both"])
def test_the_generated_script_is_valid_bash(generate, pipeline):
    """`bash -n` parses without executing -- an unquoted `(` in a path, or a stray quote in
    the `-w` expression, is a parse error that submits nothing at all.
    """
    text = generate(pipeline)
    r = subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True)
    assert r.returncode == 0, r.stderr + "\n" + text


@pytest.mark.parametrize("pipeline", ["basic", "intensity", "both"])
def test_every_stage_but_the_first_waits_on_the_one_before_it(generate, pipeline):
    """The chain, edge by edge, against `local.PIPELINES` -- the same stage order a local
    run walks. A missing `-w` submits everything at once, which is the old three-script
    workflow with the waiting silently removed.
    """
    text = generate(pipeline)
    # `emptiness` is measured at generation time and submits nothing, so it is not an edge.
    stages = [s for s in local.PIPELINES[pipeline] if s != "emptiness"]
    for after, before in zip(stages[1:], stages[:-1]):
        line = next(l for l in text.splitlines() if l.startswith(f"{VAR[after]}="))
        assert f"done(${VAR[before]})" in line, f"{after} does not wait on {before}: {line}"
    first = next(l for l in text.splitlines() if l.startswith(f"{VAR[stages[0]]}="))
    assert " -w " not in first, first


def test_the_stats_pass_is_waited_on_per_camera(generate):
    """One array per camera, and qstack needs ALL of them: BaSiC fits from the assembled
    stack, so a camera still running is a stack missing a camera.
    """
    line = next(l for l in generate("both").splitlines() if l.startswith("J_QSTACK="))
    assert 'done($J_STATS_0) && done($J_STATS_1)' in line, line


@pytest.mark.parametrize("pipeline", ["basic", "intensity", "both"])
def test_the_dependency_is_double_quoted_so_the_shell_expands_it(generate, pipeline):
    """Single quotes would submit the literal `done($J_QSTACK)` and LSF would wait forever
    on a job id that is not a number. `shlex.quote` produces exactly that.
    """
    text = generate(pipeline)
    assert '-w "done($' in text, text
    assert "-w 'done(" not in text, text


@pytest.mark.parametrize("pipeline", ["basic", "intensity", "both"])
def test_the_stages_that_read_apply_basic_are_told_it_not_left_to_detect_it(generate,
                                                                           pipeline):
    """The flag has to travel to the jobs, because each is a separate process that would
    otherwise auto-detect from whether the fields exist -- True for `--cluster intensity`
    on an experiment with leftover fields, which measures every tile from flat-fielded
    voxels and only fails in `correct`. See config._load_toml_config.
    """
    text = generate(pipeline)
    want = f"SPOTLIGHT_APPLY_BASIC={int(local.apply_basic_for(pipeline))}"
    for stage in ("int-stats", "int-aggregate"):
        if stage not in local.PIPELINES[pipeline]:
            continue
        line = next(l for l in text.splitlines() if l.startswith(f"{VAR[stage]}="))
        assert want in line, line
    correct = next(l for l in text.splitlines() if l.startswith("J_CORRECT="))
    assert f"--mode {local._CORRECT_MODE[pipeline]}" in correct, correct


def test_the_script_cds_to_the_directory_it_was_generated_in(generate, tmp_path):
    """Every stage re-reads LocalPreferences.toml from its job's working directory, and LSF
    hands a job the submitting shell's cwd. Launched from the wrong directory this is a
    tomllib parse error in every element at once.
    """
    assert f"cd {tmp_path}" in generate("both")


def test_spotfix_is_refused_rather_than_submitted():
    with pytest.raises(SystemExit, match="local-only"):
        scripts.write_pipeline_script({}, "spotfix")


def _run_jsub(tmp_path, bsub_body, args="-J x"):
    """Run the generated `jsub` against a stub `bsub`, and return (rc, stdout)."""
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    (stub / "bsub").write_text("#!/bin/bash\n" + bsub_body + "\n")
    (stub / "bsub").chmod(0o755)
    script = f"set -eu\n{scripts._JSUB}\nID=$(jsub {args})\nprintf 'got:%s' \"$ID\"\n"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={**os.environ, "PATH": f"{stub}:{os.environ['PATH']}"})
    return r.returncode, r.stdout


def test_jsub_captures_only_the_job_id(tmp_path):
    """LSF's own line has to reach the operator, and only the id may reach the variable --
    it is about to be interpolated into the next stage's `-w`.
    """
    rc, out = _run_jsub(tmp_path, 'echo "Job <8675309> is submitted to queue <normal>."')
    assert (rc, out) == (0, "got:8675309")


@pytest.mark.parametrize("body", [
    "echo 'Submitted.'; exit 0",          # succeeded, but nothing to parse
    "echo 'Batch system not available'; exit 1",
])
def test_jsub_stops_the_chain_when_it_cannot_read_an_id(tmp_path, body):
    """An empty variable makes the next `-w "done()"` an expression LSF accepts and never
    satisfies, so every later stage would park in PEND. Failing here is the only point at
    which the operator finds out.
    """
    rc, out = _run_jsub(tmp_path, body)
    assert rc == 1 and "got:" not in out


def test_the_whole_camera_stages_get_a_longer_run_limit_than_the_per_tile_ones(generate):
    """`-Q 140` has no retry cap, so a `-W` shorter than a healthy run is an infinite
    requeue loop -- and in a chain, nothing after it ever starts. qstack and basic process
    a whole camera per element; the config value is tuned against a per-tile stage.
    """
    text = generate("both")
    per_tile = config.DEFAULTS["lsf_runlimit_minutes"]
    assert scripts.WHOLE_CAMERA_RUNLIMIT > 1
    for stage in ("qstack", "basic"):
        line = next(l for l in text.splitlines() if l.startswith(f"{VAR[stage]}="))
        assert f" -W {per_tile * scripts.WHOLE_CAMERA_RUNLIMIT} " in line, line
    for stage in ("int-stats", "correct"):
        line = next(l for l in text.splitlines() if l.startswith(f"{VAR[stage]}="))
        assert f" -W {per_tile} " in line, line


@pytest.mark.parametrize("value,want", [("1", True), ("true", True), ("0", False),
                                        ("false", False), ("", False)])
def test_the_env_var_the_script_sets_beats_the_toml_and_the_detection(tmp_path, monkeypatch,
                                                                     value, want):
    """The other half of the chain: setting SPOTLIGHT_APPLY_BASIC on int-stats is only
    worth anything if config honours it, and it has to WIN -- `local.run_pipeline` overrides
    the flag in-process from the pipeline name, so a cluster run of the same pipeline has to
    reach the same answer over a toml that says otherwise.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPOTLIGHT_APPLY_BASIC", value)
    (tmp_path / "LocalPreferences.toml").write_text(
        '[spotlight]\ninput_basic_path = "/d/y.zarr"\nresults_root = "/res"\n'
        f"last_setup = 3\napply_basic = {str(not want).lower()}\n")
    assert config.load_config()["apply_basic"] is want


def test_without_the_env_var_the_toml_still_decides(tmp_path, monkeypatch):
    """Mutation guard: an override that applied unconditionally would silently pin every
    local run and every hand-run stage to one answer.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPOTLIGHT_APPLY_BASIC", raising=False)
    (tmp_path / "LocalPreferences.toml").write_text(
        '[spotlight]\ninput_basic_path = "/d/y.zarr"\nresults_root = "/res"\n'
        "last_setup = 3\napply_basic = true\n")
    assert config.load_config()["apply_basic"] is True
