"""LSF submission scripts.

Every stage -- the BaSiC ones and the per-tile intensity ones -- is reachable through the
same `python -m spotlight <stage>` entry point, so there is one runner string here instead
of one per pipeline.

The scripts are written to the CURRENT WORKING DIRECTORY, and every stage re-reads
`LocalPreferences.toml` from its own job's working directory at run time. So the LSF job's
working directory has to be the one the scripts were generated from; run them from there.

Two shapes, same stages:

* `write_pipeline_script` -- ONE script for a whole pipeline, every stage chained on the
  previous stage's LSF job ids. `python -m spotlight run both --cluster`, then run it once
  and walk away. This is the one to use.
* the per-stage generators (`create_quartile_histograms`, `write_correction_script`,
  `create_intensity_correction_script`) -- one script per stage, submitted by hand in
  order. Still here because they are how you rerun a single stage without regenerating a
  chain, and because the chained script is a driver over exactly these lines.
"""

import json
import os
import shlex
import shutil
import sys
from pathlib import Path

from .emptiness import cmd_emptiness
from . import config as _config
from . import qstack as _qstack

__all__ = [
    "create_quartile_histograms", "write_correction_script",
    "create_intensity_correction_script", "measure_emptiness", "ensure_emptiness",
    "emptiness_is_measured", "ensure_log_dirs", "runner", "write_pipeline_script",
]

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def runner():
    """The command that runs a spotlight stage inside this checkout's pixi environment.

    The manifest path is derived from `__file__` rather than configured, so a moved
    checkout fixes itself the next time the scripts are regenerated. `spotlight` comes
    from the environment -- a `[pypi-dependencies]` entry installed editable, so `pixi
    run` alone puts it on the path; run `pixi install` after moving or first cloning.

    MALLOC_ARENA_MAX caps glibc's per-thread malloc arenas, which otherwise retain every
    thread's freed numpy temporaries. It has to be in the ENVIRONMENT ahead of the
    process: glibc reads it when it creates the first arena, long before `__main__`. The
    `:-` default lets the submitting shell override it. There are deliberately no
    BLAS/OpenMP caps beside it -- those were measured and rejected. Both, plus the
    numbers: CLAUDE.md.
    """
    return ('MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-4} '
            f"pixi run --manifest-path {PACKAGE_ROOT} python -m spotlight")


def _throttle(cfg, cores, n_jobs, n_arrays=1):
    """The LSF job-array throttle suffix: how many elements may run at once.

    Budgeted in CORES, not jobs -- what the script holds is `cores * throttle * n_arrays`,
    so the budget is divided by both the cores one element asks for AND the number of
    arrays the script submits together (the stats pass submits one per camera, and they
    all run at once). A throttle written in jobs means a 48-core stats stage and a 20-core
    apply stage hold wildly different amounts of the cluster for the same number. Capped
    at the array size (a throttle above it does nothing) and floored at 1, so a single
    element of an over-budget stage still runs rather than the array never starting.
    """
    return f"%{max(1, min(n_jobs, cfg['max_concurrent_cores'] // (cores * n_arrays)))}"


def ensure_log_dirs(cfg):
    """Create the directories `output_stem` / `error_stem` point into.

    LSF does not create them. A missing one does not stop the job -- it runs and then
    fails to write its log, so the work is done but the output is gone and the failure
    reads as "Fail to open stderr file ... No such file or directory" at the very end.

    The stem is a PREFIX, not a directory (`.../output/output` -> files named
    `output_correct_1.txt`), so what has to exist is the parent of the FORMATTED path, not
    of the stem. Taking the parent of the stem itself is off by one level whenever the
    stem ends in a separator: `.../logs/output/` gives logs at
    `.../logs/output/_correct.txt`, one directory deeper than `Path(stem).parent` says.
    """
    for key in ("output_stem", "error_stem"):
        stem = cfg.get(key)
        if not stem:
            continue
        d = Path(f"{stem}_x.txt").parent
        if not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            print(f"created log directory {d} (for {key})")


# `bsub -Q`: requeue the element instead of failing it. 140 is the run-limit kill and
# nothing else -- LSF sends SIGUSR2 and 128+12 = 140. Not configurable because it does not
# vary by experiment the way the run limit does; the two forms worth hand-editing are
# `140 137` (SIGKILL, for a job too wedged to take SIGUSR2) and `EXCLUDE(140)` (keeps the
# retry off the host that just wedged -- valid only for a 1-slot stage). See CLAUDE.md.
REQUEUE_EXIT_CODES = "140"


# `qstack` and `basic` each process a whole camera in one element rather than one chunk
# range or one tile, so a run limit tuned against the apply stage -- 60 min against a 209 s
# worst case, ~17x -- can be SHORTER than a healthy run instead of a multiple of it. Both
# were run by hand until the chained script existed, so neither has ever been measured
# under LSF; until they are, they get a multiple of whatever the operator set.
#
# The asymmetry is deliberate. Overshooting costs a wedged host a few idle hours.
# Undershooting costs an infinite requeue loop: `-Q 140` has no job-level retry cap (see
# CLAUDE.md, MAX_JOB_REQUEUE is a queue setting), so an element that legitimately exceeds
# `-W` is killed and resubmitted forever -- and in a chained run nothing downstream of it
# ever starts.
WHOLE_CAMERA_RUNLIMIT = 8


def _watchdog(cfg, factor=1):
    """The ` -W <minutes> -Q "140"` suffix: kill a wedged element, then requeue it.

    An element that blocks on a wedged NFS mount does not fail -- it holds its slots until
    someone notices. `-W` is the only thing that bounds that, and `-Q` is what turns the
    kill into another attempt instead of a hole in the output. Each is pointless without
    the other: `-W` alone loses the tile, `-Q` alone never fires.

    `factor` scales the limit for a stage whose element is much larger than the one the
    config value was set against; see `WHOLE_CAMERA_RUNLIMIT`.
    """
    minutes = int(cfg.get("lsf_runlimit_minutes", _config.DEFAULTS["lsf_runlimit_minutes"]))
    minutes *= factor
    return f" -W {minutes} -Q \"{REQUEUE_EXIT_CODES}\"" if minutes > 0 else ""


def _bsub(cfg, name, cores, out_suffix, command, array=None, n_arrays=1, dep=None,
          submit="bsub", runlimit=1):
    """`array` is the element COUNT (the array is always 1-N), or None for a single job.

    `n_arrays` is how many such arrays the generated script submits together, so the core
    budget is split between them rather than handed to each.

    `dep` is a `bsub -w` expression -- `done(<id>) && ...` -- and `submit` names the
    command the line starts with, so the chained pipeline script can route it through its
    own `jsub` wrapper. `runlimit` scales `-W` (see `WHOLE_CAMERA_RUNLIMIT`). All three
    default to the standalone scripts' behaviour.
    """
    job = f"{name}[1-{array}]{_throttle(cfg, cores, array, n_arrays)}" if array else name
    index = "_%I" if array else ""
    # DOUBLE quotes, deliberately, and so not `shlex.quote`: the expression holds `&&` and
    # parentheses, which need quoting, but also `$J_STATS_0`, which the shell has to
    # expand -- single quotes would submit the literal string and LSF would wait on a
    # dependency that can never be satisfied. Safe because the only thing that ever
    # reaches `dep` is `_dep()`'s own generated variable names.
    wait = f' -w "{dep}"' if dep else ""
    return (f'{submit} -J "{job}"{wait}'
            f" -n {cores} -P {cfg['lsf_project']}{_watchdog(cfg, runlimit)}"
            # Quoted: an experiment directory can contain `(`, spaces, `&` -- shell
            # metacharacters that make the generated script a syntax error. `%I` is
            # substituted by LSF, not the shell, so single-quoting it costs nothing.
            f" -o {shlex.quote(cfg['output_stem'] + f'_{out_suffix}{index}.txt')}"
            f" -e {shlex.quote(cfg['error_stem'] + f'_{out_suffix}{index}.txt')}"
            f" '{command}'")


def _write(path, text):
    Path(path).write_text(text)
    os.chmod(path, 0o755)
    print(f"wrote {path}")


def _setup_selector(cfg, setups=None):
    """(N, prefix, arg) for a per-setup array job.

    With explicit `setup_ids` -- or an explicit `setups` list, which a copy run passes -- the
    array index selects from a shell array, because the ids need not be contiguous;
    otherwise it is just the index minus one.
    """
    flat = list(setups) if setups else [s for group in cfg.get("setup_ids", []) for s in group]
    if flat:
        return len(flat), f"S=({' '.join(str(s) for s in flat)}); ", "${S[$(($LSB_JOBINDEX-1))]}"
    return cfg["last_setup"] + 1, "", "$(($LSB_JOBINDEX-1))"


def emptiness_is_measured(cfg):
    """Whether every tile already carries the emptiness stage's measurements.

    Checks the two fields its consumers actually read: `empty_threshold` (the quantile
    stats pass measures its background profile against it) and `empty_area` (`_classify`
    needs it to tell a tile with real background from one that is signal everywhere).
    """
    from .config import stats_path, tile_list
    for setup in tile_list(cfg):
        path = stats_path(cfg, setup)
        if not path.is_file():
            return False
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        if d.get("empty_area") is None or d.get("empty_threshold") is None:
            return False
    return True


def ensure_emptiness(cfg, force=False):
    """Run the emptiness stage unless its measurements are already on disk.

    Skipped by default because it rescans EVERY tile: on a large mosaic that is real work
    to reproduce numbers that are already there, and both script generators would
    otherwise redo it every call. Pass `force=True` after changing the data or the tile
    set, since a stale `empty_area` is reused silently.
    """
    if not force and emptiness_is_measured(cfg):
        print("emptiness: measurements already on disk for every tile; skipping "
              "(pass force=True to re-measure)")
        return
    measure_emptiness(cfg)


def measure_emptiness(cfg=None):
    """Run the emptiness stage in this process.

    Measures, per camera, the empty frame area of every tile, the dataset's background
    level and intensity threshold, and the per-frame-pixel empty fraction. Four consumers
    want different parts of that: `basic_unmix_empty` wants the fraction map (and the
    threshold, so the stats pass can measure its own per-quantile profile),
    `override_darkfield` wants the scalar background level, a raw-mode un-mix wants the
    same scalar, and the intensity pipeline's tile classification wants the per-tile area.

    Failure is not fatal: the stages that need the measurements fail loudly on their own
    if they are missing, naming this stage.
    """
    cfg = _config.load_config() if cfg is None else cfg
    try:
        cmd_emptiness(cfg)
    except Exception as err:                       # noqa: BLE001 - advisory stage
        print(f"warning: the emptiness stage failed ({err!r}); stages that need its "
              "measurements will say so when they run", file=sys.stderr)


def create_quartile_histograms(cfg=None):
    """Write `bsub_command.sh`: the quantile stats pass, one array job per camera."""
    cfg = _config.load_config() if cfg is None else cfg
    ensure_log_dirs(cfg)
    ensure_emptiness(cfg)
    cameras, num_jobs, per_job = _stats_prep(cfg)
    if not cameras:
        return
    lines = [_stats_line(cfg, c, num_jobs, per_job, len(cameras)) for c in cameras]
    _write("bsub_command.sh", "\n".join(lines) + "\n")


def _stats_line(cfg, camera, num_jobs, per_job, n_cameras, dep=None, submit="bsub"):
    """One camera's quantile-stats array job; each element covers `per_job` chunks."""
    cmd = (f"{runner()} stats {camera}"
           f" $((($LSB_JOBINDEX-1)*{per_job}+1)) $(($LSB_JOBINDEX*{per_job}))")
    return _bsub(cfg, "spotlight-stats", cfg["n_cores_stats"], f"qstack_{camera + 1}",
                 cmd, array=num_jobs, n_arrays=n_cameras, dep=dep, submit=submit)


def _stats_prep(cfg):
    """Everything the quantile stats pass needs on disk before its array goes out.

    Shared by `create_quartile_histograms` and `write_pipeline_script`. Neither half is
    optional -- clearing stale partials and pre-creating the statistic arrays are both
    races that silently corrupt a camera's finished work, see the comments below -- and a
    second copy of them would drift.

    Returns `(cameras, num_jobs, per_job)`; `cameras` is empty when every camera is too
    shallow in Z to have quantiles at all.
    """
    lvl = cfg["basic_stats_level"]
    # Sized at basic_stats_level, matching the stats pass: a coarser level has a smaller
    # frame, hence fewer X/Y chunks and fewer array elements. The two must agree, or the
    # workers' chunk indices address a different tiling than the arrays were sized for.
    from . import stores
    size_xy = stores.source_size_xyz(cfg, scale=lvl)[:2]
    n_chunks = len(stores.xy_chunks(size_xy, cfg["chunk_size"][:2]))
    per_job = cfg["chunks_per_job"]
    num_jobs = -(-n_chunks // per_job)
    n_cam = _config.num_cameras(cfg)

    # Cameras too shallow in Z have no quantiles to compute -- save_qstack reads their
    # slices straight from the input instead.
    cameras = [c for c in range(n_cam) if not _qstack.raw_stack_mode(cfg, c, scale=lvl)]
    if not cameras:
        print(f"warning: every camera is too shallow in Z for quantiles; no stats jobs "
              "needed. Run save_qstack() then run_basic() directly.")
        return [], 0, per_job
    if len(cameras) < n_cam:
        skipped = sorted(set(range(n_cam)) - set(cameras))
        print(f"warning: skipping cameras too shallow in Z for quantiles: "
              f"{[c + 1 for c in skipped]}")

    # Clear a camera's background-quantile partials only when THIS run would not reproduce
    # them. They are summed blind, so a partial from a different tiling or a different setup
    # list skews the profile rather than failing -- but clearing unconditionally threw away
    # finished cameras every time an unrelated knob (`n_cores_stats`, a log stem) sent
    # someone back through here to regenerate the script, which is the common case.
    #
    # The stamp records what the SUM depends on: which partial files get written
    # (`chunks_per_job` and the chunk count set the `job{start}.json` keys), and what went
    # into them (the setup list, the threshold columns are tested against, the pixel
    # stride). Anything else -- cores, log stems, the LSF project -- leaves the partials
    # valid, so they survive. A missing or stale stamp clears, so partials from before this
    # check existed are never trusted.
    from .quantiles import (BACKGROUND_PIXEL_STRIDE, background_quantile_dir,
                            empty_threshold)
    for c in cameras:
        d = background_quantile_dir(cfg, c)
        fp = {"chunks_per_job": per_job, "n_chunks": n_chunks, "level": lvl,
              "frame": [int(v) for v in size_xy],
              "chunk_size": [int(v) for v in cfg["chunk_size"][:2]],
              "setups": [int(x) for x in _config.camera_setups(cfg)[c]],
              "empty_threshold": empty_threshold(cfg, c),
              "stride": BACKGROUND_PIXEL_STRIDE}
        stamp = d / "fingerprint.json"
        try:
            unchanged = json.loads(stamp.read_text()) == fp
        except (OSError, ValueError):
            unchanged = False
        if unchanged:
            print(f"camera {c + 1}: keeping "
                  f"{len(list(d.glob('job*.json')))} background-quantile partial(s)")
            continue
        if d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        stamp.write_text(json.dumps(fp, sort_keys=True))

    # Create every statistic array HERE, once, before the array job goes out. A worker
    # opens with `create=True`, which is harmless when the array already exists but not
    # when 40 elements race to create the same one: tensorstore's file kvstore writes
    # attributes.json with O_TRUNC, so a late creator truncates an ALREADY VALID one to
    # zero bytes, and every element afterwards dies `Invalid JSON` -- permanently, because
    # tensorstore cannot open the file to repair it. Seen on camera 9 of an 11-camera run:
    # q010 and q015 were clobbered after chunk 0/6 had already been written to them, and
    # all 40 elements failed while the other ten cameras were fine. Creating them from one
    # process leaves the workers nothing to race over. They still create on demand, so a
    # direct `python -m spotlight stats` outside this driver keeps working.
    from .orderstats import LEVELS
    stat_names = ("minima", "maxima", *(f"q{q:03d}" for q in LEVELS))
    ctx = stores.context()
    for c in cameras:
        for name in stat_names:
            # Metadata that exists but will not parse is unopenable AND unrepairable in
            # place -- tensorstore has to read it to write it -- so hand that array to
            # `rebuild`, which deletes and recreates it. Narrow on purpose: only a file
            # that is present and broken. Rebuilding on any open failure would let one
            # flaky network read discard a camera's finished chunks.
            meta = stores.stats_array_path(cfg, c, name, lvl) / "attributes.json"
            broken = False
            if meta.is_file():
                try:
                    broken = not json.loads(meta.read_text())
                except (OSError, ValueError):
                    broken = True
            if broken:
                print(f"rebuilding {meta.parent}: attributes.json is unreadable "
                      "(a create race truncated it); its chunks go with it")
            stores.open_stats_array(cfg, c, name, size_xy, scale=lvl, ctx=ctx,
                                    rebuild=broken)
    print(f"{len(cameras)} camera(s) x {len(stat_names)} statistic array(s) ready")
    print(f"{num_jobs} job(s) x {len(cameras)} camera(s), {n_chunks} chunks, "
          f"level {lvl}, frame {size_xy}")
    return cameras, num_jobs, per_job


def write_correction_script(cfg=None, mode="basic"):
    """Write `bsub_correction.sh`: the flat/dark apply, one array element per setup.

    Pinned to `--mode basic` rather than left on `auto`. This script exists for the
    flat/dark stage specifically -- the joint route is
    `create_intensity_correction_script` -- and `auto` would silently upgrade to `both` if
    an `intensity_target.json` happened to be sitting in `results_root` from an earlier
    experiment. Pass `mode=` to override.
    """
    cfg = _config.load_config() if cfg is None else cfg
    ensure_log_dirs(cfg)
    n, prefix, arg = _setup_selector(cfg)
    cmd = f"{prefix}{runner()} correct {arg} --mode {mode}"
    _write("bsub_correction.sh",
           _bsub(cfg, "spotlight-correct", cfg["n_cores_correction"], "correct", cmd,
                 array=n) + "\n")


def create_intensity_correction_script(cfg=None):
    """Write the three per-tile intensity-correction scripts.

    Run them in order, waiting for each stage's jobs to finish before submitting the next:
    there is no `-w` dependency between them, since LSF name-based dependencies proved
    unreliable here. `write_pipeline_script` is the version that does the waiting for you,
    chaining on job IDS instead of names -- prefer it unless you are rerunning one stage.

    1. `bsub_int_stats.sh`     -- per-tile foreground mean/std behind a background mask
    2. `bsub_int_aggregate.sh` -- one job: solves per-tile gains from overlapping pairs,
                                  reduces every tile's stats into a single target
    3. `bsub_int_correct.sh`   -- rescales each setup to the target, writing the pyramid

    With `apply_basic` on, all three also apply this package's flat/dark fields to the
    voxels they read, so `write_correction_script`'s stage is not run at all and the data
    is read once, written once, and rounded to uint16 once.
    """
    cfg = _config.load_config() if cfg is None else cfg
    ensure_log_dirs(cfg)
    # `aggregate` refuses to run without `empty_area` in every tile's stats, and only
    # the emptiness stage measures it. `create_quartile_histograms` covers the BaSiC
    # route; an intensity-only run never touches that function, so cover it here too --
    # skipped when the measurements are already on disk.
    ensure_emptiness(cfg)
    n, prefix, arg = _setup_selector(cfg)
    for stage, script, name, suffix, cores, is_array in (
        ("stats", "stats", "spotlight-int-stats", "is", cfg["n_cores_int_stats"], True),
        ("aggregate", "aggregate", "spotlight-int-aggregate", "ia", cfg["n_cores_int_aggregate"], False),
        ("apply", "correct", "spotlight-int-correct", "ic", cfg["n_cores_int_correct"], True),
    ):
        if is_array:
            cmd = f"{prefix}{runner()} int-{stage} {arg}"
            line = _bsub(cfg, name, cores, suffix, cmd, array=n)
        else:
            line = _bsub(cfg, name, cores, suffix, f"{runner()} int-{stage}")
        _write(f"bsub_int_{script}.sh", line + "\n")


# ─── the whole pipeline as one chained submission ─────────────────────────────

# The wrapper every stage of the chained script submits through. `bsub` writes
# "Job <12345> is submitted to queue <normal>." to stdout, so the id goes to stdout and
# LSF's own line to stderr -- `V=$(jsub ...)` then captures the id alone while the
# operator still sees the message.
#
# The `exit 1` on an id that will not parse is load-bearing. An empty variable turns the
# next stage's `-w "done()"` into an expression LSF accepts and never satisfies, so a
# single failed submission would silently detach every stage after it. Stopping at the
# first bad one is the only way that surfaces at submit time.
_JSUB = '''jsub() {
  local out id
  out=$(bsub "$@") || { printf '%s\\n' "$out" >&2; exit 1; }
  printf '%s\\n' "$out" >&2
  id=${out#Job <}; id=${id%%>*}
  case "$id" in
    ''|*[!0-9]*) printf 'spotlight: no job id in bsub output: %s\\n' "$out" >&2; exit 1 ;;
  esac
  printf '%s' "$id"
}'''


def _dep(ids):
    """The `bsub -w` expression waiting on every job id in `ids`, or None for none."""
    return " && ".join(f"done(${v})" for v in ids) or None


def _stage_jobs(cfg, stage, mode, apply_basic, dep, tiles=None):
    """`(bsub lines, shell variable names)` for one pipeline stage.

    `dep` is the `-w` expression this stage waits on. A stage that submits nothing returns
    no variable names, and the caller carries the previous stage's ids forward across it so
    the chain stays connected.

    The job names and log suffixes are the same ones the standalone scripts use, so the
    log-scraping recipes in CLAUDE.md (`output/_ic_*.txt`) keep working.
    """
    sub = {"dep": dep, "submit": "jsub"}
    # Only int-stats and int-aggregate read the flag; `correct` sets its own from --mode.
    # Stated in the environment rather than detected per job -- see the SPOTLIGHT_APPLY_BASIC
    # comment in config._load_toml_config for what detection gets wrong here.
    env = f"SPOTLIGHT_APPLY_BASIC={int(apply_basic)} "

    if stage == "emptiness":
        # Measured at generation time, not submitted. `_stats_prep`'s fingerprint -- which
        # decides whether a camera's finished partials survive -- is computed HERE and reads
        # `empty_threshold`, so the measurement has to be on disk before anything goes out.
        # `ensure_emptiness` skips the rescan when it already is.
        return [], []
    if stage == "stats":
        cameras, num_jobs, per_job = _stats_prep(cfg)
        return ([_stats_line(cfg, c, num_jobs, per_job, len(cameras), **sub)
                 for c in cameras],
                [f"J_STATS_{c}" for c in cameras])
    if stage == "qstack":
        return ([_bsub(cfg, "spotlight-qstack", cfg["n_cores_qstack"], "qs",
                       f"{runner()} qstack", runlimit=WHOLE_CAMERA_RUNLIMIT, **sub)],
                ["J_QSTACK"])
    if stage == "basic":
        # One element per camera: the fits are independent, and the SVD is the slow part.
        n = _config.num_cameras(cfg)
        return ([_bsub(cfg, "spotlight-basic", cfg["n_cores_basic_fit"], "bf",
                       f"{runner()} basic $(($LSB_JOBINDEX-1))", array=n,
                       runlimit=WHOLE_CAMERA_RUNLIMIT, **sub)],
                ["J_BASIC"])
    if stage == "int-stats":
        n, prefix, arg = _setup_selector(cfg)
        return ([_bsub(cfg, "spotlight-int-stats", cfg["n_cores_int_stats"], "is",
                       f"{prefix}{env}{runner()} int-stats {arg}", array=n, **sub)],
                ["J_INT_STATS"])
    if stage == "int-aggregate":
        return ([_bsub(cfg, "spotlight-int-aggregate", cfg["n_cores_int_aggregate"], "ia",
                       f"{env}{runner()} int-aggregate", **sub)],
                ["J_INT_AGGREGATE"])
    if stage == "correct":
        n, prefix, arg = _setup_selector(cfg, tiles)
        # Its own job name and log suffix, so a copy run's logs never sit in the same
        # `_ic_*.txt` namespace as the correction whose output it is writing into.
        name, suffix = {"basic": ("spotlight-correct", "correct"),
                        "copy": ("spotlight-copy", "cp"),
                        "copy-basic": ("spotlight-copy", "cp"),
                        }.get(mode, ("spotlight-int-correct", "ic"))
        return ([_bsub(cfg, name, _config.stage_cores(cfg, mode), suffix,
                       f"{prefix}{runner()} correct {arg} --mode {mode}", array=n, **sub)],
                ["J_CORRECT"])
    raise ValueError(f"unknown stage {stage!r}")


def write_pipeline_script(cfg=None, pipeline="both", start_at=None, stop_after=None,
                          tiles=None):
    """Write one script that submits a whole pipeline, chained on LSF job ids.

    The same stage list `local.run_pipeline` walks in one process, submitted instead: each
    stage waits on the previous stage's ids with `bsub -w done(<id>)`, so the operator runs
    one script and LSF does all the waiting. Job IDS and not job names -- name-based
    dependencies proved unreliable here, which is why the three intensity scripts were
    separate files with "wait for each stage to finish" in the docstring.

    `done()` rather than `ended()` on purpose. A stage that starts on a partial previous
    stage does not fail, it produces a quietly wrong answer: `int-aggregate` will solve a
    target from 195 of 196 tiles and report `n_present` without complaint. The price is
    that a genuinely failed element leaves everything downstream in PEND with "Dependency
    condition never satisfied" -- visible in `bjobs -l`, and the script's own trailer says
    so. It is a one-word edit in the generated file for anyone who wants the other trade.
    """
    from . import local
    if pipeline == "spotfix":
        raise SystemExit("spotfix is local-only and takes the tiles to repair: "
                         "python -m spotlight run spotfix 126 158")
    cfg = _config.load_config() if cfg is None else cfg
    # Imported rather than restated: `PIPELINES` is the one definition of stage order, and
    # `_plan`/`_CORRECT_MODE`/`apply_basic_for` are what make a cluster run and a local run
    # of the same pipeline name mean the same thing.
    stages = local._plan(pipeline, start_at, stop_after)
    mode = local._CORRECT_MODE[pipeline]
    apply_basic = local.apply_basic_for(pipeline)

    from .correct import COPY_MODES
    if mode in COPY_MODES:
        if not tiles:
            raise SystemExit(f"{pipeline} needs the tiles to copy, as ids or ranges: "
                             f"python -m spotlight run {pipeline} 200-395 --cluster")
        # Checked HERE, before anything is submitted: a typo in a hand-entered range is
        # otherwise 200 array elements each dying on a store that was never written.
        tiles = _config.check_setups_in_xml(cfg, tiles)
        print(f"{pipeline}: {len(tiles)} tile(s) {tiles[0]}..{tiles[-1]}, verified against "
              f"{cfg['dataset_xml']}")
    elif tiles:
        tiles = _config.check_setups_in_xml(cfg, tiles)

    ensure_log_dirs(cfg)
    # A copy measures nothing, so it needs no emptiness pass -- and on a fresh dataset that
    # would be a full rescan of every tile to produce numbers this run never reads.
    if mode not in COPY_MODES:
        ensure_emptiness(cfg)

    out = [
        "#!/bin/bash",
        f"# The spotlight `{pipeline}` pipeline as one chained LSF submission:",
        f"#     {' -> '.join(stages)}",
        "#",
        "# `emptiness`, if listed, is not submitted -- it was measured when this script was",
        "# generated, because the stats pass needs its threshold to decide which partials to",
        "# keep. Everything else below is a job.",
        "#",
        f"# Generated by `python -m spotlight run {pipeline} --cluster`. Every path in here",
        "# is absolute and baked in, so regenerate after editing LocalPreferences.toml,",
        "# changing the tile set, or moving the checkout. Re-running submits the whole chain",
        "# again -- it is not idempotent.",
        "#",
        "# Each stage waits on the previous stage's job ids. If one element fails, the",
        "# stages after it stay PEND rather than running on partial input:",
        "#     bjobs -l <id> | grep -i depend     # 'Dependency condition never satisfied'",
        "# Fix the cause, bkill the parked jobs, and rerun with --start-at <stage>.",
        "set -eu",
        "",
        # Baked in rather than left to the caller: every stage re-reads
        # LocalPreferences.toml from its own job's working directory, LSF hands a job the
        # submitting shell's cwd, and the failure when that is wrong is a `tomllib` parse
        # error in 196 elements at once.
        f"cd {shlex.quote(str(Path.cwd()))} || exit 1",
        "",
        _JSUB,
        "",
    ]

    prev, summary = [], []
    for stage in stages:
        lines, ids = _stage_jobs(cfg, stage, mode, apply_basic, _dep(prev), tiles)
        if not lines:
            continue                      # emptiness, or a stats pass with no camera
        out.append(f"# ─── {stage} " + "─" * max(0, 60 - len(stage)))
        out.append(f'echo "spotlight: submitting {stage}" >&2')
        out += [f"{var}=$({line})" for var, line in zip(ids, lines)]
        out.append("")
        summary.append((stage, ids))
        prev = ids

    if not prev:
        raise SystemExit(f"the {pipeline} pipeline has no cluster stages to submit")

    every = [v for _, ids in summary for v in ids]
    out += ['echo "" >&2',
            f'echo "spotlight: submitted {pipeline} -- {len(every)} job(s)" >&2']
    out += [f'echo "  {stage:<14} {" ".join(f"${v}" for v in ids)}" >&2'
            for stage, ids in summary]
    out += ['echo "  watch   bjobs -A -J \'spotlight-*\'" >&2',
            f'echo "  cancel  bkill {" ".join(f"${v}" for v in every)}" >&2']

    path = f"bsub_pipeline_{pipeline}.sh"
    _write(path, "\n".join(out) + "\n")
    if mode in COPY_MODES:
        margin = int(cfg.get("crop_margin") or 0)
        print(f"correct --mode {mode}: no arithmetic, "
              f"{'BaSiC' if mode == 'copy-basic' else 'intensity'} I/O pair"
              + (f", crop_margin={margin} (every tile loses {2 * margin} voxels per axis; "
                 f"regenerate the output store's dataset.xml with "
                 f"`python -m spotlight.crop_xml`)" if margin else ""))
    else:
        print(f"apply_basic={apply_basic} for int-stats/int-aggregate; "
              f"correct --mode {mode}")
    print(f"run it once, from this directory:  ./{path}")
