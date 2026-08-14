"""Compare `tile_threshold` estimators on a real dataset, without writing anything.

    cd <experiment dir with LocalPreferences.toml>
    python /path/to/spotlight/bench/compare_thresholds.py            # every setup
    python /path/to/spotlight/bench/compare_thresholds.py 0 1 2      # just these

Reads each tile once at `stats_scale` and reports what each estimator would choose, what
fraction of the tile it would mark foreground, and -- the part that decides whether a
tile participates in the gain solve at all -- whether it would be classified `empty`.

Why this is a script and not a test: there is no right answer to assert. Which threshold
is correct is a judgement about the specimen, and the point here is to put the numbers
side by side so that judgement can be made on evidence.

The column that explains everything else is `r`, the ratio of the two class means at
Li's threshold. Li's fixed point is the LOGARITHMIC mean of the class means and
Otsu/isodata's is the ARITHMETIC mean, so their ratio is a pure function of r:

    li/otsu ~ 2(r-1) / ((r+1) ln r)

which is ~1 when the classes are close and falls away as they separate (0.71 at r=10,
0.43 at r=100). That is the whole difference between the two methods: on a sparse tile
with a long bright tail r is large and Otsu is dragged into the tail; on an almost-all-
tissue tile r is small and the two agree. If this run shows r below ~5 everywhere, Otsu
is not failing on this dataset and switching to Li will change little.
"""

import sys
from pathlib import Path

import numpy as np
from skimage.filters import threshold_isodata, threshold_li, threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spotlight import config as _config          # noqa: E402
from spotlight.tilestats import (MIN_FG_FRACTION, MIN_FOREGROUND,  # noqa: E402
                                 _read_tile_volume)

METHODS = {"otsu": threshold_otsu, "li": threshold_li, "isodata": threshold_isodata}


def _would_be_empty(n_fg, n_vox):
    """The `_classify` foreground gate, restated. A tile that fails it is called `empty`,
    dropped from the gain solve, and passed through UNCORRECTED -- so a threshold that
    pushes a real tile under this bar is not a cosmetic problem."""
    return n_fg < MIN_FOREGROUND or n_fg < MIN_FG_FRACTION * n_vox


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = _config.load_config()
    setups = ([int(a) for a in argv] if argv
              else [s for g in _config.camera_setups(cfg) for s in g])

    print(f"stats_scale={cfg['stats_scale']}  {len(setups)} setup(s)")
    print(f"empty gate: n_fg >= {MIN_FOREGROUND} AND n_fg >= {MIN_FG_FRACTION:.1%} "
          f"of the tile\n")
    header = f"{'setup':>5} {'shape':>18} {'max':>7} {'r':>7} "
    for m in METHODS:
        header += f"{m:>9} {'fg%':>7} {'':>2}"
    print(header)
    print(f"{'':>5} {'':>18} {'':>7} {'':>7} " +
          "".join(f"{'':>9} {'':>7} {'E?':>2}" for _ in METHODS))

    rows = {m: [] for m in METHODS}
    for s in setups:
        vol = _read_tile_volume(cfg, s)
        a = vol.reshape(-1).astype(np.float64)
        line = f"{s:>5} {str(vol.shape):>18} {a.max():>7.0f} "
        r = float("nan")
        cells = []
        for name, fn in METHODS.items():
            try:
                t = float(fn(vol))
            except Exception as e:                       # a method can fail on a tile
                cells.append(f"{'FAILED':>9} {'':>7} {'':>2}")
                print(f"  note: {name} failed on setup {s}: {e}", file=sys.stderr)
                continue
            n_fg = int((a > t).sum())
            frac = n_fg / a.size
            empty = "E" if _would_be_empty(n_fg, a.size) else ""
            rows[name].append((t, frac, empty == "E"))
            cells.append(f"{t:>9.0f} {frac:>6.2%} {empty:>2}")
            if name == "li" and n_fg and n_fg < a.size:
                m0, m1 = a[a <= t].mean(), a[a > t].mean()
                r = m1 / m0 if m0 > 0 else float("inf")
        print(line + f"{r:>7.1f} " + " ".join(cells))

    print("\n(E = this threshold would classify the tile `empty`: dropped from the gain "
          "solve and passed through uncorrected)")
    print("\n" + "-" * 60)
    for name in METHODS:
        vals = rows[name]
        if not vals:
            continue
        ts = [v[0] for v in vals]
        spread = max(ts) / max(min(ts), 1e-9)
        n_empty = sum(v[2] for v in vals)
        print(f"{name:>9}: thresholds {min(ts):.0f}-{max(ts):.0f} ({spread:.1f}x spread), "
              f"median fg {np.median([v[1] for v in vals]):.2%}, "
              f"{n_empty}/{len(vals)} tiles would be called empty")
    print("\nRead `r` first: it is the ratio of the class means, and li/otsu ~ "
          "2(r-1)/((r+1)ln r).\nr below ~5 means Otsu is not being dragged into a tail "
          "on this data and Li will\nchange little; r in the tens or hundreds means it "
          "is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
