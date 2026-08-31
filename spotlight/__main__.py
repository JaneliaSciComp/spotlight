"""`python -m spotlight <stage> [args]` -- the single entry point every job runs.

Both pipelines dispatch from here, the BaSiC stages and the per-tile intensity ones
(`int-*`). That is what lets `scripts.py` have one runner string instead of one per
pipeline.

Camera and setup arguments are 0-BASED, matching the intensity stages. On disk cameras
stay 1-based (`camera1/`, `Flat-field.tif`), as the Julia package wrote them. Nobody types
these by hand -- the `submit` stage generates them.
"""

import argparse
import os
import sys


def _index_fallback(value):
    """A missing positional falls back to this job's LSF array index, as the intensity
    stages already do, so a bsub line can omit it.
    """
    if value is not None:
        return value
    idx = os.environ.get("LSB_JOBINDEX")
    if idx is None:
        raise SystemExit("no argument given and LSB_JOBINDEX is not set")
    return int(idx) - 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spotlight", description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("stats", help="quantile statistics for one camera's chunk range")
    p.add_argument("camera", type=int)
    p.add_argument("start", type=int, nargs="?", default=1)
    p.add_argument("stop", type=int, nargs="?", default=None)

    sub.add_parser("qstack", help="assemble the per-camera BaSiC input TIFFs")

    p = sub.add_parser("basic", help="fit flat/dark fields (all cameras, or those named)")
    p.add_argument("cameras", type=int, nargs="*")

    p = sub.add_parser("correct", help="correct one setup: flat/dark, intensity, or both")
    p.add_argument("setup", type=int, nargs="?", default=None)
    p.add_argument("--mode", choices=("auto", "basic", "intensity", "both"), default="auto",
                   help="auto (default) picks `both` when the BaSiC fields and the "
                        "intensity target are both present, else whichever one is")

    sub.add_parser("emptiness", help="measure background level, threshold, empty fractions")

    p = sub.add_parser("int-stats", help="per-tile intensity stats for one setup")
    p.add_argument("setup", type=int, nargs="?", default=None)
    sub.add_parser("int-aggregate", help="reduce per-tile intensity stats to a target")
    # Kept as an alias so existing bsub scripts keep working; there is only one
    # correction implementation now, and this is it with the mode pinned.
    p = sub.add_parser("int-apply", help="alias for `correct --mode auto`")
    p.add_argument("setup", type=int, nargs="?", default=None)

    p = sub.add_parser("submit", help="write the bsub scripts")
    p.add_argument("which", choices=("stats", "correct", "intensity"))

    p = sub.add_parser("run", help="run a whole pipeline here, without LSF")
    p.add_argument("pipeline", choices=("basic", "intensity", "both", "spotfix"))
    p.add_argument("tiles", type=int, nargs="*",
                   help="for `spotfix`: the setups to repair, e.g. `spotfix 126 158`")
    p.add_argument("--start-at", default=None,
                   help="resume from this stage (it is re-run, not skipped)")
    p.add_argument("--stop-after", default=None,
                   help="stop after this stage -- the checkpoints worth inspecting are "
                        "`qstack` and `basic`")
    p.add_argument("--dry-run", action="store_true",
                   help="list the stages and units without running them")

    args = parser.parse_args(argv)

    from . import config

    if args.stage == "run":
        from . import local
        local.run_pipeline(config.load_config(), args.pipeline, args.start_at,
                           args.stop_after, args.dry_run, args.tiles)
        return

    if args.stage == "submit":
        from . import scripts
        cfg = config.load_config()
        {"stats": scripts.create_quartile_histograms,
         "correct": scripts.write_correction_script,
         "intensity": scripts.create_intensity_correction_script}[args.which](cfg)
        return

    if args.stage in ("int-stats", "int-aggregate"):
        from .aggregate import cmd_aggregate
        from .config import _load_toml_config
        from .tilestats import cmd_stats
        icfg = _load_toml_config()
        if args.stage == "int-aggregate":
            cmd_aggregate(icfg)
        else:
            cmd_stats(icfg, _index_fallback(args.setup))
        return

    cfg = config.load_config()
    if args.stage == "stats":
        from . import quantiles
        quantiles.calculate_camera_stats(cfg, args.camera, args.start, args.stop)
    elif args.stage == "qstack":
        from . import qstack
        qstack.save_qstack(cfg)
    elif args.stage == "basic":
        from . import basic
        basic.run_basic(cfg, args.cameras or None)
    elif args.stage in ("correct", "int-apply"):
        from . import correct
        mode = getattr(args, "mode", "auto")
        correct.apply_correction_chunked(cfg, _index_fallback(args.setup), mode)
    elif args.stage == "emptiness":
        from . import scripts
        scripts.measure_emptiness(cfg)


if __name__ == "__main__":
    sys.exit(main())
