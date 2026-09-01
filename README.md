# spotlight

Shading and intensity correction for large microscopy datasets.

Shading correction: flat-field/dark-field estimation via the BaSiC algotithm (heavily inspired by [this repo](https://github.com/JaneliaSciComp/BigFlatFieldIlluminator.jl)) \
Intensity correction: Per-camera or per-tile intensity matching based upon overlapping regions

Supported formats: OME-Zarr v2/v3, N5, OME-TIFF (output only)

## Install

```bash
pixi install                 # linux-64, osx-arm64 and win-64 all resolve from one lock
```

The cluster workflow below is Linux-only (it shells out to `bsub`). Windows runs the
**local workflow** only. Two Windows-specific things to know:

* **Write paths in the toml with forward slashes**, or single-quoted: `"C:\data\exp"` is
  not valid TOML — `\d` is an unrecognised escape and `tomllib` rejects the whole file.
  `"C:/data/exp"` and `'C:\data\exp'` both work, and `set_config()` escapes correctly on
  your behalf. Backslashes are normalised to `/` internally either way.
* **Enable long paths** if a store lives more than a few directories deep. Zarr chunk keys
  are long, and the 260-character default limit surfaces as a confusing tensorstore write
  error. `Computer Configuration > Administrative Templates > System > Filesystem > Enable
  Win32 long paths`, or the `LongPathsEnabled` registry DWORD.

## Configure

Settings live in `LocalPreferences.toml` in each experiment's own directory, under a
`[spotlight]` table (and `[spotlight.basic]` for the BaSiC parameters). Every stage reads
it from its own working directory, so an experiment is selected by `cd`-ing into it — and
the LSF jobs must run from the same directory the scripts were generated in.

Config params can either be set programmatically (see below for an example), or by hand-editing.

```python
import spotlight as sp

sp.set_config(
    input_basic_path  = "/path/to/data/dataset.ome.zarr",   # the store BaSiC reads
    output_basic_path = "/path/to/basic_corrected.ome.zarr",# the store `correct` writes
    results_root      = "/path/to/results",
    dataset_xml       = "/path/to/data/dataset.xml",  # tile overlaps, for the gain solve
    input_format      = "zarr3",  # zarr2 | zarr3 | zarr3_unsharded | zarr3_zyx | zarr3_raw | n5
    output_format     = "zarr3",  # same list, plus tiff (output only)
    last_setup        = 8,
    setups_per_camera = 9,
    chunk_size        = [128, 128, 64],
    shard_size        = [512, 512, 256],
    lsf_project       = "myproject",
    output_stem       = "$HOME/output/output",
    error_stem        = "$HOME/output/error",
)
sp.set_basic_config(estimate_darkfield=True)
```

See later in this README for a discussion regarding parameter choices.

## Workflow

One command runs a whole pipeline in this process. `--cluster` writes a single `bsub`
script for the same stages instead, chained on LSF job ids — nothing is submitted until
you run that script, so it is its own dry run.

```bash
python -m spotlight run basic        # emptiness, stats, qstack, basic, correct
python -m spotlight run intensity    # emptiness, int-stats, int-aggregate, correct
python -m spotlight run both         # both corrections, applied in ONE pass

python -m spotlight run both --cluster              # -> bsub_pipeline_both.sh (then run it)
python -m spotlight run both --dry-run              # list the stages and units first
python -m spotlight run basic --stop-after qstack   # stop and look at the qstack
python -m spotlight run basic --start-at basic      # resume after inspecting it
python -m spotlight run intensity 200-395           # narrow the correct stage to these tiles
```

Two pipelines take the tiles to act on, as ids or inclusive ranges:

```bash
python -m spotlight run spotfix 126 158   # repair local dimming in a corrected tile (local only)
python -m spotlight run copy 200-395      # rewrite tiles into the corrected layout, uncorrected
```

Individual stages, for a hand-driven run or a custom bsub script. `emptiness` must precede
`int-aggregate` — the tile classification needs its `empty_area`:

```bash
python -m spotlight emptiness                 # background level, threshold, empty fractions
python -m spotlight stats <camera> [start] [stop]
python -m spotlight qstack                    # -> qstacks/camera{N}.tiff  (inspect these)
python -m spotlight basic [cameras...]        # -> results_root/camera{N}/{Flat,Dark}-field.tif
python -m spotlight int-stats <setup>
python -m spotlight int-aggregate             # -> tile_gains.json, intensity_target.json
python -m spotlight correct <setup> [--mode auto|basic|intensity|both|copy|copy-basic]
python -m spotlight submit stats|correct|intensity   # the older per-stage bsub scripts
```

The same entry points exist in Python (`sp.create_quartile_histograms()`, `sp.save_qstack()`,
`sp.run_basic()`, `sp.create_intensity_correction_script()`, …) if you would rather drive it
from a notebook.

## Modes

See config.py for a complete list of params. Below is a compilation of some that are dataset dependent.

### BaSiC Parameters
```"basic_unmix_empty": bool (default false)``` \
&emsp;Turn this on if you have camera(s) with asymmetrical darkfield profiles (i.e. a lobe or something similar that is empty on only one part of the tile xy plane).

```"estimate_darkfield": bool (default true)``` \
&emsp;Turn this off if there is visible no empty space (i.e. there is no darkfield in the sample)

```"override_darkfield": bool (default false) or number``` \
&emsp;Turn this on if you would like to supplant the estimated darkfield with a flat darkfield.

### Intensity Correction
```"gain_grouping": "camera" (default) or "tile"``` \
&emsp;Use "camera" if you are trying to match intensities across cameras. "Tile" is typically used to deal with per-tile bleaching.

```"gain_estimator": "intersection" (default) or "independent"``` \
&emsp;If your dataset is already reasonably well-stitched, use "intersection." "Independent" is used for unstitched or poorly stitched data.

```"gain_lambda": Float between 0 and 1``` \
&emsp;This is the intensity matching regularization term. If you would like your gain mappings to be almost unregularized (for ex, highly determined systems like camera-based workflows), we recommend 1e-6. 0.01 or 0.1 is typical for tile-based workflows.

```"gain_floor": "tile" (default), "otsu", "li", "pooled", or a number``` \
&emsp;This is the minimum intensity cutoff used to calculate the gain mappings. It defaults to the current value of tile_threshold.

```"tile_threshold": "li" (default), "otsu", "pooled", or a number``` \
&emsp;This is the minimum intensity cutoff used during the intensity correction step; i.e. all voxels with an intensity below this threshold will be considered to be background and not adjusted. Typically matches gain_floor, but it can be helpful to lower this is some foreground voxels are not being adjusted. Li is the default and works well on datasets with unequal background and foreground distributions. Otsu is more efficient.