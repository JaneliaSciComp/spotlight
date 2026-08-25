"""Run one benchmarked stage and append a JSON record.

Invoked by the generated bsub scripts. Emits the same schema whether it ran the Python
implementation or shelled out to the Julia one, so `collect.py` can pair them.

    python bench.py <impl> <stage> <camera-or-setup> [start] [stop]

`impl` is "python" or "julia"; `stage` is "stats" or "correct".
"""

import json
import os
import resource
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

BENCH_DIR = Path(os.environ.get("BENCH_DIR", "bench_results"))
JULIA_PROJECT = os.environ.get("SPOTLIGHT_JULIA_PROJECT", "")
JULIA = os.environ.get("SPOTLIGHT_JULIA", "julia")


def _maxrss_bytes():
    """Peak RSS of this process AND its children.

    Children matter: the julia arm runs as a subprocess, so RUSAGE_SELF alone would report
    the harness rather than the thing being measured. Linux reports kilobytes.
    """
    self_ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scale = 1024 if sys.platform.startswith("linux") else 1
    return max(self_, kids) * scale


def _geometry(stage, target):
    """Bytes this job will read, from the config rather than measured -- deterministic,
    and it makes MB/s comparable across differently-sized jobs."""
    from spotlight import config, stores
    cfg = config.load_config()
    if stage == "stats":
        lvl = cfg["basic_stats_level"]
        setups = config.camera_setups(cfg)[target]
        x, y, z = stores.camera_source_size_xyz(cfg, setups, scale=lvl)
        tile = cfg["chunk_size"][:2]
        return dict(n_setups=len(setups), tile=tile, cam_size=[x, y, z])
    x, y, z = stores.source_size_xyz(cfg, setup=target)
    return dict(n_setups=1, tile=cfg["chunk_size"][:2], cam_size=[x, y, z])


def _isolated_workdir(stage, arm):
    """A working directory whose config writes somewhere only this arm writes.

    Every element of the `correct` array corrects the SAME setup -- that is the point, and
    it is what makes the arms comparable. But they would then all write the same output
    store, and concurrent writers to one sharded zarr race on the shard lock files
    ("Failed to rename ...__lock"), failing the job and invalidating the timing.

    Both implementations read their config from the CURRENT DIRECTORY, so isolating them
    needs no code change in either: copy the toml into a per-arm directory with the output
    paths suffixed, and run there. `stats` needs none of this -- it writes per-camera
    statistic arrays at disjoint chunk offsets.
    """
    if stage != "correct":
        return None
    src = Path.cwd() / "LocalPreferences.toml"
    with open(src, "rb") as f:
        tables = tomllib.load(f)
    key = "spotlight" if "spotlight" in tables else "BigFlatFieldIlluminator"
    d = Path.cwd() / f".bench_arm_{arm}"
    d.mkdir(exist_ok=True)
    text = src.read_text()
    for k in ("output_basic_path", "output_intensity_path"):
        old = tables[key].get(k)
        if old:
            text = text.replace(f'"{old}"', f'"{old}.bench_{arm}"')
    (d / "LocalPreferences.toml").write_text(text)
    return d


def main():
    impl, stage, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rest = [int(a) for a in sys.argv[4:]]
    root = Path(__file__).resolve().parents[1]

    if impl == "python":
        cmd = [sys.executable, "-m", "spotlight", stage, str(target), *map(str, rest)]
        env = dict(os.environ)
    else:
        app = "Stats" if stage == "stats" else "Correct"
        # Julia's stats CLI takes a 1-BASED camera; spotlight's is 0-based.
        arg = target + 1 if stage == "stats" else target
        cmd = [JULIA, f"--project={JULIA_PROJECT}", "-t", "auto",
               "-m", f"BigFlatFieldIlluminator.{app}", str(arg), *map(str, rest)]
        env = dict(os.environ)

    arm = f'{impl}_{os.environ.get("SPOTLIGHT_CORRECT_CONCURRENCY", "0")}'
    workdir = _isolated_workdir(stage, arm)

    t0 = time.perf_counter()
    # Tee rather than swallow: the LSF log stays useful, and the stage's own
    # SPOTLIGHT_TIMING line gets folded into this record so the next question after
    # "which is faster" -- I/O bound or CPU bound? -- is already answered.
    proc = subprocess.run(cmd, env=env, cwd=workdir, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    print(proc.stdout, end="")
    phases = {}
    for line in proc.stdout.splitlines():
        if line.startswith("SPOTLIGHT_TIMING "):
            try:
                phases = json.loads(line[len("SPOTLIGHT_TIMING "):])
            except ValueError:
                pass

    rec = {
        "impl": impl, "stage": stage, "target": target, "args": rest,
        "t_wall": round(wall, 3), "returncode": proc.returncode,
        "maxrss_bytes": _maxrss_bytes(),
        "host": os.uname().nodename,
        "jobid": os.environ.get("LSB_JOBID", ""),
        "jobindex": os.environ.get("LSB_JOBINDEX", ""),
        "cores": os.environ.get("LSB_DJOB_NUMPROC", ""),
        "sem_limit": os.environ.get("SPOTLIGHT_STATS_CONCURRENCY")
                     or os.environ.get("SPOTLIGHT_CORRECT_CONCURRENCY", ""),
    }
    rec.update(phases)
    try:
        rec.update(_geometry(stage, target))
    except Exception as err:                                  # noqa: BLE001
        rec["geometry_error"] = repr(err)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    name = f'{rec["jobid"] or "local"}_{rec["jobindex"] or os.getpid()}_{impl}.json'
    (BENCH_DIR / name).write_text(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
