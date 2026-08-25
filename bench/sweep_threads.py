"""Does a stage stay inside its LSF allocation, and what does staying inside it cost?

    python bench/sweep_threads.py stats 0 1 64
    python bench/sweep_threads.py correct 0
    python bench/sweep_threads.py correct 0 --arms legacy,shipped --repeat 2

Scicomp's contract is 1-minute load average <= 2x RESERVED SLOTS, so what has to be
measured is not the thread COUNT but how many of those threads are runnable or in
uninterruptible sleep at once -- that sum is what the load average is. A thread parked in
an NFS read is charged exactly like one burning a core, which is why an idle-looking pool
of 1920 file-I/O threads is not free.

Read `/proc/loadavg` and you measure the whole host, everyone else's jobs included. This
samples `/proc/<pid>/task/*/stat` and counts only THIS job's threads in state R or D: its
own contribution to the load, uncontaminated. (`/proc/loadavg` is reported alongside
anyway -- it is literally the number in the email, and on a quiet node it corroborates.)

Each arm is a full stage run in a fresh subprocess with a different environment, because
every knob under test is read at process start: OpenBLAS sizes its pool when the shared
library loads, and tensorstore builds its thread pools from the first `ts.Context`.

    legacy     file_io slots*64, data_copy slots  -- the behaviour that drew the email
    shipped    file_io slots/2,  data_copy slots  -- the current defaults
    io-half    file_io slots/2   } the file_io curve at data_copy = slots/2. Measured flat
    io-1x      file_io slots     } then worse, which is how the two pools were told apart:
    io-4x      file_io slots*4   } all the wall clock was in data_copy, none in file_io.
    io-16x     file_io slots*16  }
    both-full  both slots        -- one step past `shipped`; measured 2.2x, over the ceiling

Run it on a QUIET node (`bsub -n <slots> -Is`), or the load column measures the
neighbours. Linux only: /proc is the whole measurement.

The arms run in the order given, so on a source small enough to fit the page cache the
later ones read warm and look faster. `--shuffle` spreads that across arms instead of
letting it land on one; with `--repeat 2` the spread is visible in the table. It does not
matter for a 70 GiB correction input, which cannot be cached.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_SECONDS = 0.2

# The yardstick the verdict column judges against: scicomp's ceiling on a host's 1-minute
# load average as a multiple of the slots LSF reserved. The bench's own, not a production
# knob -- `spotlight` sizes its pools from measurements taken with this script, not from a
# constant.
LOAD_CEILING = 2


def arms(n):
    """Environment overlay per arm: the two tensorstore pool limits, in threads."""
    half = max(2, n // 2)
    return {
        "legacy":    dict(io=n * 64, copy=n),
        "shipped":   dict(io=half,   copy=n),
        "io-half":   dict(io=half,   copy=half),
        "io-1x":     dict(io=n,      copy=half),
        "io-4x":     dict(io=4 * n,  copy=half),
        "io-16x":    dict(io=16 * n, copy=half),
        "both-full": dict(io=n,      copy=n),
    }


def arm_env(spec, n):
    """Build one arm's environment.

    Starts from a COPY with every knob cleared, so a value inherited from the submitting
    shell cannot silently leak into an arm that means to leave it at its default.

    `LSB_DJOB_NUMPROC` is pinned to `n` because the STAGE reads it too -- it sizes its own
    kernel ThreadPoolExecutor from `stores.slots()`. Judging a ratio against 30 slots
    while the child quietly builds an 8-thread pool measures neither configuration.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("SPOTLIGHT_IO_CONCURRENCY", "SPOTLIGHT_COPY_CONCURRENCY")}
    env["LSB_DJOB_NUMPROC"] = str(n)
    env["SPOTLIGHT_IO_CONCURRENCY"] = str(spec["io"])
    env["SPOTLIGHT_COPY_CONCURRENCY"] = str(spec["copy"])
    return env


def _state_char(line):
    """The state field of a `/proc/<pid>/task/<tid>/stat` line.

    Read after the LAST ')' because field 2 is the executable name, unescaped: it may
    itself contain spaces and parentheses, so splitting on whitespace picks the wrong
    field for exactly the processes whose names are interesting.
    """
    return line[line.rindex(")") + 2]


def _hwm_kib(pid):
    """The child's own peak RSS, sampled while it is still alive.

    Read from /proc rather than `getrusage(RUSAGE_CHILDREN)`, whose `ru_maxrss` is a
    monotonic high-water mark across every child ever waited on: the first arm's peak
    sticks and each later arm reports a delta of 0, which reads as "this arm used no
    memory" rather than "not measured". Hence here, and not after `communicate()`.
    """
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _states(pid):
    """(threads, in R or D) for this instant, or None once the process is gone."""
    try:
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
    except OSError:
        return None
    n = busy = 0
    for t in tasks:
        try:
            line = (t / "stat").read_text()
        except OSError:
            continue                      # thread exited between listing and reading
        try:
            state = _state_char(line)
        except (ValueError, IndexError):
            continue
        n += 1
        busy += state in "RD"
    return n, busy


def _sampler(pid, out, stop):
    peak_t = peak_b = peak_rss = 0
    busies, loads = [], []
    while not stop.is_set():
        s = _states(pid)
        if s is not None:
            n, busy = s
            peak_t, peak_b = max(peak_t, n), max(peak_b, busy)
            peak_rss = max(peak_rss, _hwm_kib(pid))
            busies.append(busy)
        try:
            loads.append(float(Path("/proc/loadavg").read_text().split()[0]))
        except OSError:
            pass
        time.sleep(SAMPLE_SECONDS)
    out.update(peak_threads=peak_t, peak_busy=peak_b, rss_gib=round(peak_rss / 2**20, 2),
               mean_busy=round(sum(busies) / len(busies), 1) if busies else 0.0,
               peak_host_load=max(loads) if loads else 0.0, n_samples=len(busies))


def run_arm(name, spec, stage, args, n):
    env = arm_env(spec, n)
    cmd = [sys.executable, "-m", "spotlight", stage, *map(str, args)]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, cwd=os.getcwd(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    got, stop = {}, threading.Event()
    th = threading.Thread(target=_sampler, args=(p.pid, got, stop), daemon=True)
    th.start()
    out = p.communicate()[0]
    stop.set(); th.join()
    wall = time.perf_counter() - t0
    rec = {"arm": name, "t_wall": round(wall, 1), "rc": p.returncode, **got}
    m = re.search(r"SPOTLIGHT_TIMING (\{.*\})", out)
    if m:
        t = json.loads(m.group(1))
        rec["gib_in"] = round(t.get("bytes_in", 0) / 2**30, 1)
    if p.returncode:
        rec["tail"] = out.strip().splitlines()[-1:] or ["(no output)"]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", help="a `python -m spotlight` stage, e.g. stats or correct")
    ap.add_argument("args", nargs="*", help="that stage's own arguments")
    ap.add_argument("--arms", default=None, help="comma-separated subset, in this order")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--shuffle", action="store_true",
                    help="randomise arm order, so page-cache warmth does not all land on "
                         "whichever arm happens to run last")
    ap.add_argument("--slots", type=int, default=None,
                    help="the reservation to judge against, when not running under LSF. "
                         "Also pinned into each arm's LSB_DJOB_NUMPROC so the stage sizes "
                         "its own pools from the same number.")
    a = ap.parse_args(argv)

    # NOT `slots()`: its default exists for off-cluster library use, and here a guessed
    # denominator invents every ratio in the table AND every arm's pool sizes. On a 64-core
    # node with no reservation exported it would quietly measure an 8-slot configuration
    # and call it OK.
    reserved = os.environ.get("LSB_DJOB_NUMPROC")
    if a.slots is None and not reserved:
        sys.exit(
            f"LSB_DJOB_NUMPROC is not set, so there is no reservation to measure against "
            f"and nothing here would mean anything (this host has {os.cpu_count()} logical "
            f"cores, which is NOT the number -- the contract is load vs SLOTS).\n"
            f"Either run under LSF -- `bsub -n <slots> -Is ...` -- or say what to assume: "
            f"--slots <n>.")
    n = a.slots or int(reserved)
    if reserved and a.slots and int(reserved) != a.slots:
        print(f"note: --slots {a.slots} overrides the {reserved} LSF reserved; the arms "
              f"will run as if {a.slots} were the allocation")
    if not Path("/proc/self/task").is_dir():
        sys.exit("this needs Linux /proc -- run it on the cluster, not a laptop")

    table = arms(n)
    chosen = a.arms.split(",") if a.arms else list(table)
    unknown = [c for c in chosen if c not in table]
    if unknown:
        sys.exit(f"unknown arm(s) {unknown}; have {list(table)}")

    ceiling = LOAD_CEILING * n
    print(f"{n} slots ({'LSF' if reserved and not a.slots else '--slots'}) on a host with "
          f"{os.cpu_count()} logical cores -> ceiling {ceiling} threads "
          f"runnable-or-blocked ({LOAD_CEILING}x), "
          f"stage `{a.stage} {' '.join(map(str, a.args))}`")
    print("`peak busy` is this job's own contribution to the load average; `host` is "
          "/proc/loadavg,\nwhich includes every other job on the node.\n")
    hdr = f"{'arm':<10} {'wall s':>7} {'threads':>8} {'peak busy':>10} {'/slots':>7} " \
          f"{'mean busy':>10} {'RSS GiB':>8} {'host':>7}  verdict"
    print(hdr); print("-" * len(hdr))

    order = [n for n in chosen for _ in range(a.repeat)]
    if a.shuffle:
        random.shuffle(order)

    rows = []
    for name in order:
        r = run_arm(name, table[name], a.stage, a.args, n)
        rows.append(r)
        if r["rc"]:
            print(f"{name:<10} FAILED rc={r['rc']} {r.get('tail')}")
            continue
        ratio = r["peak_busy"] / n
        print(f"{name:<10} {r['t_wall']:>7.1f} {r['peak_threads']:>8} "
              f"{r['peak_busy']:>10} {ratio:>6.1f}x {r['mean_busy']:>10} "
              f"{r['rss_gib']:>8} {r['peak_host_load']:>7.0f}  "
              f"{'OK' if ratio <= LOAD_CEILING else 'OVER'}")

    ok = [r for r in rows if not r["rc"]]
    if len(ok) > 1:
        base = min((r for r in ok if r["arm"] == "legacy"), default=None,
                   key=lambda r: r["t_wall"])
        fastest = min(ok, key=lambda r: r["t_wall"])
        print()
        if base:
            print(f"legacy: {base['t_wall']:.0f}s at {base['peak_busy'] / n:.1f}x slots")
        compliant = [r for r in ok if r["peak_busy"] / n <= LOAD_CEILING]
        if compliant:
            best = min(compliant, key=lambda r: r["t_wall"])
            cost = f" ({best['t_wall'] / base['t_wall']:.2f}x legacy)" if base else ""
            print(f"fastest arm inside {LOAD_CEILING}x: {best['arm']} at "
                  f"{best['t_wall']:.0f}s{cost}")
        else:
            print(f"NO arm stayed inside {LOAD_CEILING}x slots -- the ceiling needs a "
                  f"smaller pool than any arm here tries")
        print(f"fastest overall: {fastest['arm']} at {fastest['t_wall']:.0f}s, "
              f"{fastest['peak_busy'] / n:.1f}x slots")
    Path("sweep_threads.json").write_text(json.dumps(rows, indent=2))
    print("\nrows -> sweep_threads.json")


if __name__ == "__main__":
    main()
