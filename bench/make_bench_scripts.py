"""Write the benchmark submission scripts for this experiment.

    cd <experiment dir>            # the one with LocalPreferences.toml
    python <spotlight>/bench/make_bench_scripts.py --julia-project <BigFlatFieldIlluminator.jl>

Both implementations run inside ONE LSF array job, with the element index choosing which:
even -> Julia, odd -> Python, on the SAME camera and chunk range. That is the whole point.
Two separate submissions would land on different hosts at different times against a
differently-warm page cache, and no amount of averaging fixes that; interleaving them in
one array puts each pair on the same host pool in the same window, so the per-index ratio
is meaningful on its own.

Thread pools are pinned identically on both sides, tensorstore's own context included:
both implementations drive the same tensorstore, so leaving those at their defaults would
benchmark tensorstore's defaults rather than the port.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from spotlight import config, stores

    ap = argparse.ArgumentParser()
    ap.add_argument("--julia-project", required=True,
                    help="path to the BigFlatFieldIlluminator.jl checkout")
    ap.add_argument("--julia", default="julia")
    ap.add_argument("--camera", type=int, default=0, help="0-based")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--bench-dir", default="bench_results")
    ap.add_argument("--sem-sweep", default="4,16,64",
                    help="semaphore limits to sweep for the correct stage")
    args = ap.parse_args()

    cfg = config.load_config()
    cores = args.cores or cfg["n_cores_stats"]
    per_job = cfg["chunks_per_job"]
    lvl = cfg["basic_stats_level"]
    n_chunks = len(stores.xy_chunks(stores.source_size_xyz(cfg, scale=lvl)[:2],
                                    cfg["chunk_size"][:2]))
    setups = config.camera_setups(cfg)[args.camera]

    # Everything that needs $LSB_JOBINDEX lives in run_arm.sh, so nothing here has to
    # survive a round of shell quoting. The command stays inside SINGLE quotes, which is
    # what stops the job's own variables from expanding at submit time -- the bug that
    # turned `$(( $PAIR * 64 + 1 ))` into a syntax error when this was inlined.
    env = (f"BENCH_DIR={args.bench_dir} "
           f"SPOTLIGHT_JULIA_PROJECT={args.julia_project} "
           f"SPOTLIGHT_JULIA={args.julia}")
    runner = f"pixi run --manifest-path {ROOT} env {env} bash {ROOT}/bench/run_arm.sh"

    # ── stats ──────────────────────────────────────────────────────────────
    # 2 * repeats elements: index parity picks the implementation, and each PAIR of
    # elements shares one chunk range, so `collect.py` can diff them directly.
    n = 2 * args.repeats
    stats_line = (
        f'bsub -J "spotlight-bench-stats[1-{n}]" -n {cores} -P {cfg["lsf_project"]}'
        f' -R "span[hosts=1]"'
        f' -o {cfg["output_stem"]}_bench_stats_%I.txt'
        f' -e {cfg["error_stem"]}_bench_stats_%I.txt'
        f" '{runner} stats {args.camera} {per_job}'"
    )
    _write("bsub_bench_stats.sh", stats_line)

    # ── correct ───────────────────────────────────────────────────────────
    # Julia holds every chunk's read future at once; the port bounds them. So the Python
    # arm sweeps its semaphore limit and the deliverable is a maxRSS-vs-wall-time curve,
    # not a single "N% faster". The Julia arm has no such knob -- it gets one arm.
    sems = [int(s) for s in args.sem_sweep.split(",")]
    arms = ["julia:0"] + [f"python:{s}" for s in sems]
    n2 = len(arms) * args.repeats
    correct_line = (
        f'bsub -J "spotlight-bench-correct[1-{n2}]" -n {cfg["n_cores_correction"]}'
        f' -P {cfg["lsf_project"]} -R "span[hosts=1]"'
        f' -o {cfg["output_stem"]}_bench_correct_%I.txt'
        f' -e {cfg["error_stem"]}_bench_correct_%I.txt'
        f" '{runner} correct {setups[0]} {' '.join(arms)}'"
    )
    _write("bsub_bench_correct.sh", correct_line)


    print(f"\ncamera {args.camera} has {len(setups)} setups; the frame tiles into "
          f"{n_chunks} chunks at level {lvl}, {per_job} per job.")
    print("Submit ./bsub_bench_stats.sh, wait for it, then ./bsub_bench_correct.sh.")
    print(f"Then: python {ROOT}/bench/collect.py {args.bench_dir}")


def _write(path, line):
    Path(path).write_text(line + "\n")
    os.chmod(path, 0o755)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
