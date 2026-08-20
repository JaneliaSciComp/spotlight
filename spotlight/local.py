"""Run the whole pipeline in one process, without LSF.

`scripts.py` generates bsub arrays: one element per chunk range, per setup. That is the
right shape on a cluster and useless on a workstation. This walks the same units in a
single process instead, so a dataset small enough to sit on one machine needs no queue.

It is a driver, not a second implementation -- every stage below is the same function the
bsub scripts invoke. What changes is only who loops over the units.

    python -m spotlight run basic
    python -m spotlight run intensity --stop-after aggregate
    python -m spotlight run both --dry-run

Sequential over units, on purpose. Each stage is already concurrent inside itself
(asyncio over reads, a thread pool for the numpy) and sized to a memory budget derived
from the whole machine -- so running two units at once would double the memory while
competing for the same saturated I/O. The parallelism that pays here is already inside
the unit.
"""

import time

from . import config as _config

__all__ = ["PIPELINES", "STAGES", "run_pipeline", "apply_basic_for"]

# Each entry is (name, what it iterates). The order is the dependency order; there is no
# scheduler, because the dependencies are a straight line.
PIPELINES = {
    "basic": ["emptiness", "stats", "qstack", "basic", "correct"],
    "intensity": ["emptiness", "int-stats", "int-aggregate", "correct"],
    "both": ["emptiness", "stats", "qstack", "basic", "int-stats", "int-aggregate",
             "correct"],
}

STAGES = sorted({s for v in PIPELINES.values() for s in v})

# Which correction each pipeline ends with. `both` reads the raw store once and applies
# flat/dark and the per-tile gain together, which is the whole reason to prefer it.
_CORRECT_MODE = {"basic": "basic", "intensity": "intensity", "both": "both"}


def apply_basic_for(pipeline):
    """Whether every stage of `pipeline` works on BaSiC-corrected voxels.

    `load_config` auto-detects this by asking whether camera 1's flat/dark fields exist.
    That is the right question for a single stage and the WRONG one for a pipeline,
    because a pipeline can create the very files the detection looks at: in `both`, the
    `basic` stage writes Flat-field.tif three steps before `int-stats` needs the answer.
    Detected once at the start, `both` therefore ran `int-stats`/`int-aggregate` as raw
    (the fields did not exist yet) and then hit the `correct` stage, which sets
    `apply_basic = (mode == "both")` unconditionally -- and the two disagreed:

        RuntimeError: intensity_target.json was written with apply_basic=False but
        this run has apply_basic=True

    It also fails the other way: with fields left over from an earlier run, `intensity`
    would detect True, write stats from corrected voxels, and then correct as raw.

    The pipeline name already states the intent, so take it from there and hold it for
    the whole run. This matches `correct._view` exactly, which is the file the check
    compares against.
    """
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {sorted(PIPELINES)}")
    return _CORRECT_MODE[pipeline] == "both"


def _plan(pipeline, start_at, stop_after):
    stages = PIPELINES[pipeline]
    lo = stages.index(start_at) if start_at else 0
    hi = stages.index(stop_after) + 1 if stop_after else len(stages)
    if lo >= hi:
        raise ValueError(f"--start-at {start_at} is after --stop-after {stop_after}")
    return stages[lo:hi]


def _units(cfg, stage, mode):
    """What this stage iterates, as a list of (label, callable)."""
    from . import basic, correct, qstack, quantiles, scripts, tilestats, aggregate

    cameras = range(_config.num_cameras(cfg))
    setups = [s for group in _config.camera_setups(cfg) for s in group]

    if stage == "emptiness":
        return [("all tiles", lambda: scripts.ensure_emptiness(cfg))]
    if stage == "stats":
        # One call per camera covering EVERY chunk -- the bsub array splits this only so
        # LSF can spread it; in one process the split would just add overhead.
        return [(f"camera {c + 1}", lambda c=c: quantiles.calculate_camera_stats(cfg, c))
                for c in cameras if not qstack.raw_stack_mode(cfg, c,
                                                              scale=cfg["basic_stats_level"])]
    if stage == "qstack":
        return [("all cameras", lambda: qstack.save_qstack(cfg))]
    if stage == "basic":
        return [(f"camera {c + 1}", lambda c=c: basic.run_basic_camera(cfg, c))
                for c in cameras]
    if stage == "int-stats":
        return [(f"setup {s}", lambda s=s: tilestats.cmd_stats(cfg, s)) for s in setups]
    if stage == "int-aggregate":
        return [("all tiles", lambda: aggregate.cmd_aggregate(cfg))]
    if stage == "correct":
        return [(f"setup {s}", lambda s=s: correct.apply_correction_chunked(cfg, s, mode))
                for s in setups]
    raise ValueError(f"unknown stage {stage!r}")


def run_pipeline(cfg=None, pipeline="basic", start_at=None, stop_after=None,
                 dry_run=False):
    """Run `pipeline` end to end in this process.

    `--start-at` / `--stop-after` exist because the natural checkpoints are inspection
    points: you look at the qstack before fitting, and at the fields before correcting a
    whole dataset. Re-running from a stage re-does that stage; nothing is skipped because
    its output happens to exist, except `emptiness`, which is expensive and idempotent.
    """
    cfg = _config.load_config() if cfg is None else cfg
    mode = _CORRECT_MODE[pipeline]
    cfg = {**cfg, "apply_basic": apply_basic_for(pipeline)}
    stages = _plan(pipeline, start_at, stop_after)

    # On its own line: `test_stage_windows` parses the stage list off the line above by
    # splitting on stage names, and a suffix containing "basic" reads as a stage.
    print(f"pipeline: {pipeline} -> {' -> '.join(stages)}")
    # "for every stage" was a lie in both directions and it reads as a bug report: only
    # int-stats and int-aggregate consult this flag, and `correct` overrides it from `mode`
    # (mode=basic forces it True, see `correct._view`). Under `run basic` the old wording
    # announced False for a pipeline whose whole point is applying BaSiC.
    print(f"apply_basic={cfg['apply_basic']} for the stages that read it "
          f"(int-stats, int-aggregate); the correct stage sets its own from mode={mode}")
    plan = [(s, _units(cfg, s, mode)) for s in stages]
    total = sum(len(u) for _, u in plan)
    print(f"{total} unit(s) across {len(stages)} stage(s), sequential, in this process")
    if dry_run:
        for stage, units in plan:
            print(f"  {stage:14} {len(units):4} unit(s): "
                  f"{', '.join(l for l, _ in units[:4])}"
                  f"{' ...' if len(units) > 4 else ''}")
        return

    t0 = time.perf_counter()
    done = 0
    for stage, units in plan:
        for label, fn in units:
            done += 1
            t = time.perf_counter()
            print(f"\n[{done}/{total}] {stage}: {label}", flush=True)
            fn()
            print(f"[{done}/{total}] {stage}: {label} done in "
                  f"{time.perf_counter() - t:.1f}s", flush=True)
    print(f"\npipeline {pipeline} finished in {time.perf_counter() - t0:.1f}s")
