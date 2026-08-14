"""Find where throughput peaks as concurrency rises, under a fixed memory budget.

    python sweep_concurrency.py stats 0 1 64 --values 4,8,18,32,64
    python sweep_concurrency.py correct 0    --values 1,2,4,8

Runs the stage once per concurrency value in a fresh subprocess, and reports wall clock,
read throughput and peak RSS for each -- plus what `stores.memory_budget()` would have
chosen on its own, which is the number the sweep is really auditing.

Why this is a script and not a pytest: the answer depends on the filesystem, the node, and
what else is running on it. There is no assertion to make, only a measurement to read.
The harness itself IS tested (tests/test_sweep.py).

Two confounds it handles explicitly, because both will otherwise invent a knee that is
not there:

  * PAGE CACHE. The first value read from cold looks slowest whatever it is. Values are
    run in shuffled order by default and each is repeated, so the cache advantage is
    spread across them rather than landing on one. `--warmup` reads the data once first,
    which measures warm-cache behaviour instead -- honest, but not what a production job
    sees.
  * MEMORY. Raising concurrency raises peak RSS, and LSF kills a job that exceeds its
    reservation. The table reports RSS beside throughput so the fastest value that still
    fits is visible, rather than just the fastest.
"""

import argparse
import json
import os
import random
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_VAR = {"stats": "SPOTLIGHT_STATS_CONCURRENCY",
           "correct": "SPOTLIGHT_CORRECT_CONCURRENCY"}


def run_once(stage, args, concurrency, budget=None):
    """One stage run at a pinned concurrency. Returns its timing record."""
    env = {**os.environ, ENV_VAR[stage]: str(concurrency)}
    if budget is not None:
        env["SPOTLIGHT_MEMORY_BYTES"] = str(budget)
    cmd = [sys.executable, "-m", "spotlight", stage, *map(str, args)]
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scale = 1024 if sys.platform.startswith("linux") else 1
    rec = {"concurrency": concurrency, "t_wall": wall, "returncode": p.returncode,
           # ru_maxrss for CHILDREN is a high-water mark across all of them, so it only
           # rises; the delta is this run's contribution when it set a new peak, and 0
           # when an earlier, larger run still holds the mark.
           "maxrss_bytes": max(after - before, 0) * scale,
           "maxrss_highwater_bytes": after * scale}
    for line in p.stdout.splitlines():
        if line.startswith("SPOTLIGHT_TIMING "):
            try:
                rec.update(json.loads(line[len("SPOTLIGHT_TIMING "):]))
            except ValueError:
                pass
    if p.returncode:
        rec["tail"] = p.stdout.strip().splitlines()[-3:]
    return rec


def _scratch_workdir(results_root):
    """A working directory whose config writes to `results_root` instead of the real one.

    The sweep RE-RUNS the stage, once per value per repeat, so pointed at a live
    `results_root` it would rewrite those chunks a dozen times. Same values each time, so
    nothing is corrupted -- but a reference set you are comparing against stops being one
    the moment the sweep touches it.

    Both implementations read their config from the current directory, so redirecting it
    needs no code change: copy the toml with `results_root` swapped and run there. The
    emptiness measurements are copied across too -- the stats pass reads
    `empty_threshold` from them, and without it the pass skips the background profile and
    stops doing the same work the real one does.
    """
    import shutil
    import tomllib

    src = Path.cwd() / "LocalPreferences.toml"
    with open(src, "rb") as f:
        tables = tomllib.load(f)
    key = "spotlight" if "spotlight" in tables else "BigFlatFieldIlluminator"
    old = tables[key]["results_root"]

    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    work = root / "_sweep_cwd"
    work.mkdir(exist_ok=True)
    (work / "LocalPreferences.toml").write_text(
        src.read_text().replace(f'"{old}"', f'"{results_root}"'))

    stats_src, stats_dst = Path(old) / "intensity_stats", root / "intensity_stats"
    if stats_src.is_dir() and not stats_dst.exists():
        shutil.copytree(stats_src, stats_dst)
        print(f"copied {stats_src.name} so empty_threshold is found")
    elif not stats_src.is_dir():
        print(f"note: no intensity_stats under {old}; the stats pass will skip the "
              "background profile, so it does slightly less work than a real run")
    return work


def summarise(records, auto):
    """Print the table and name the peak. Returns the best concurrency."""
    by_conc = {}
    for r in records:
        by_conc.setdefault(r["concurrency"], []).append(r)

    print(f"\n{'conc':>5} {'runs':>5} {'wall (s)':>9} {'MiB/s':>8} "
          f"{'peak RSS':>10} {'t_read':>9} {'t_compute':>10}")
    rows = []
    for c in sorted(by_conc):
        rs = [r for r in by_conc[c] if not r.get("returncode")]
        if not rs:
            print(f"{c:>5} {'--':>5}   ALL RUNS FAILED: {by_conc[c][0].get('tail')}")
            continue
        wall = statistics.median(r["t_wall"] for r in rs)
        byt = statistics.median(r.get("bytes_in", 0) for r in rs)
        rss = max(r["maxrss_bytes"] for r in rs)
        rd = statistics.median(r.get("t_read", 0) for r in rs)
        cp = statistics.median(r.get("t_compute", 0) for r in rs)
        thru = byt / 2**20 / wall if wall and byt else 0.0
        rows.append((c, wall, thru, rss))
        print(f"{c:>5} {len(rs):>5} {wall:>9.1f} {thru:>8.1f} "
              f"{rss/2**30:>8.2f}Gi {rd:>9.1f} {cp:>10.1f}")

    if not rows:
        return None
    best = min(rows, key=lambda r: r[1])
    print(f"\nfastest: concurrency {best[0]} at {best[1]:.1f}s "
          f"({best[2]:.1f} MiB/s, {best[3]/2**30:.2f} GiB peak)")
    if auto is not None:
        hit = [r for r in rows if r[0] == auto]
        if hit:
            cost = (hit[0][1] / best[1] - 1) * 100
            print(f"the derivation would pick {auto}: {hit[0][1]:.1f}s "
                  f"({cost:+.0f}% vs the best value here)")
        else:
            print(f"the derivation would pick {auto}, which was not swept -- add it to "
                  f"--values to see whether the default is well placed")
    # A knee, not a peak, is the usual shape: throughput flattens once the reads are
    # saturated and only RSS keeps climbing. Say where it flattens.
    flat = [r for r in rows if r[1] <= best[1] * 1.05]
    if flat and min(f[0] for f in flat) != best[0]:
        cheap = min(flat, key=lambda r: r[0])
        print(f"within 5% of the best from concurrency {cheap[0]} onward "
              f"({cheap[3]/2**30:.2f} GiB vs {best[3]/2**30:.2f} GiB) -- prefer the "
              f"lower one unless the memory is free")
    return best[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(ENV_VAR))
    ap.add_argument("args", nargs="*", help="the stage's own arguments")
    ap.add_argument("--values", default="1,2,4,8,16,32",
                    help="comma-separated concurrencies to try")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--budget", type=int, default=None,
                    help="pin SPOTLIGHT_MEMORY_BYTES, so every value sees the same "
                         "budget and only the concurrency differs")
    ap.add_argument("--order", choices=("shuffled", "ascending"), default="shuffled")
    ap.add_argument("--warmup", action="store_true",
                    help="read once before timing; measures WARM-cache behaviour")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the raw records here as JSON")
    ap.add_argument("--results-root", default=None,
                    help="run against a scratch results_root instead of the configured "
                         "one, so the sweep does not rewrite real outputs")
    a = ap.parse_args(argv)

    values = [int(v) for v in a.values.split(",")]
    sys.path.insert(0, str(ROOT))
    from spotlight import stores

    if a.results_root:
        workdir = _scratch_workdir(a.results_root)
        os.chdir(workdir)
        print(f"running in {workdir} against results_root={a.results_root}")
    auto = None
    if a.budget:
        os.environ["SPOTLIGHT_MEMORY_BYTES"] = str(a.budget)
    print(f"memory budget: {stores.memory_budget()/2**30:.1f} GiB "
          f"({'pinned' if a.budget else 'derived from the allocation'})")

    if a.warmup:
        print("warmup run (not timed)...")
        run_once(a.stage, a.args, max(values), a.budget)

    plan = [(c, i) for i in range(a.repeats) for c in values]
    if a.order == "shuffled":
        random.Random(a.seed).shuffle(plan)
    records = []
    for n, (c, rep) in enumerate(plan, 1):
        print(f"[{n}/{len(plan)}] concurrency {c} (repeat {rep + 1})...", flush=True)
        records.append(run_once(a.stage, a.args, c, a.budget))

    best = summarise(records, auto)
    if a.out:
        Path(a.out).write_text(json.dumps(records, indent=2))
        print(f"wrote {a.out}")
    return 0 if best is not None else 1


if __name__ == "__main__":
    sys.exit(main())
