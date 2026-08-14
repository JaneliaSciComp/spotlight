"""Summarise a benchmark run: `python collect.py <bench_dir>`.

Reports the PAIRED per-index ratio, not just an aggregate. The cluster is heterogeneous
and the page cache is warm or cold depending on what ran before, so a mean over all
julia jobs against a mean over all python jobs mostly measures which hosts each landed
on. The pairing is what makes the number mean something -- and any pair that did land on
different hosts is flagged rather than averaged in.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _mb(n):
    return n / (1 << 20)


def main(bench_dir):
    records = [json.loads(p.read_text()) for p in sorted(Path(bench_dir).glob("*.json"))]
    if not records:
        raise SystemExit(f"no benchmark records in {bench_dir}")

    failed = [r for r in records if r.get("returncode")]
    if failed:
        print(f"## {len(failed)} job(s) exited non-zero\n")
        for r in failed:
            print(f"- {r['impl']} {r['stage']} index {r['jobindex']} "
                  f"rc={r['returncode']} host={r['host']}")
        print()

    ok = [r for r in records if not r.get("returncode")]

    for stage in ("stats", "correct"):
        rows = [r for r in ok if r["stage"] == stage]
        if not rows:
            continue
        print(f"## {stage}\n")
        by_arm = defaultdict(list)
        for r in rows:
            arm = r["impl"] if not r.get("sem_limit") or r["impl"] == "julia" \
                else f'python(sem={r["sem_limit"]})'
            by_arm[arm].append(r)

        print("| arm | n | median wall (s) | median maxRSS (MiB) | conc | read | compute | MiB/s |")
        print("|---|---|---|---|---|---|---|---|")
        for arm, rs in sorted(by_arm.items()):
            walls = sorted(r["t_wall"] for r in rs)
            rss = sorted(r["maxrss_bytes"] for r in rs)
            # Phase totals are summed over concurrent tasks, so they exceed the wall
            # clock; their RATIO is what says whether a stage is I/O or CPU bound.
            def med(k):
                v = [r[k] for r in rs if r.get(k)]
                return statistics.median(v) if v else 0.0
            rd, cp = med("t_read"), med("t_compute")
            byt = med("bytes_in")
            thru = byt / 2**20 / rd if rd else 0.0
            conc = {r.get("concurrency") for r in rs if r.get("concurrency")}
            print(f"| {arm} | {len(rs)} | {statistics.median(walls):.1f} | "
                  f"{_mb(statistics.median(rss)):.0f} | "
                  f"{','.join(str(c) for c in sorted(conc)) or '-'} | "
                  f"{rd:.0f}s | {cp:.0f}s | {thru:.0f} |")
        print()

        # Paired ratios, keyed on the work each element actually did.
        pairs = defaultdict(dict)
        for r in rows:
            pairs[(r["target"], tuple(r["args"]))][r["impl"]] = r
        ratios = []
        print("| target | args | julia (s) | python (s) | python/julia | same host |")
        print("|---|---|---|---|---|---|")
        for (target, args), arm in sorted(pairs.items()):
            j, p = arm.get("julia"), arm.get("python")
            if not (j and p):
                continue
            ratio = p["t_wall"] / j["t_wall"]
            ratios.append(ratio)
            same = "yes" if j["host"] == p["host"] else f"NO ({j['host']} vs {p['host']})"
            print(f"| {target} | {list(args)} | {j['t_wall']:.1f} | {p['t_wall']:.1f} | "
                  f"{ratio:.2f}x | {same} |")
        if ratios:
            print(f"\n**median python/julia wall ratio: {statistics.median(ratios):.2f}x** "
                  f"({len(ratios)} pairs; below 1.0 means the port is faster)\n")
        else:
            print("\n(no complete julia/python pairs)\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_results")
