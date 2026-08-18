# spotlight

Shading and intensity correction for large microscopy datasets.

Shading correction: flat-field/dark-field estimation via the BaSiC algotithm (heavily inspired by [this repo](https://github.com/JaneliaSciComp/BigFlatFieldIlluminator.jl)) \
Intensity correction: Per-camera or per-tile intensity matching based upon overlapping regions

Supported formats: OME-Zarr v2/v3, N5, OME-TIFF (output only)

## Install

```bash
pixi install                 # both linux-64 and osx-arm64 resolve from one lock
```

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
    input_format      = "zarr3",       # zarr2 | zarr3 | zarr3_unsharded | n5
                                       # output_format defaults to input_format; add "tiff" for OME-TIFF out
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

### Cluster-driven Workflow

#### Full pipeline:
```python
sp.create_quartile_histograms()               # -> bsub_command.sh      (submit it, wait)
sp.save_qstack()                              # -> qstacks/camera{N}.tiff  (inspect these)
sp.run_basic()                                # -> results_root/camera{N}/{Flat,Dark}-field.tif
sp.create_intensity_correction_script()       # -> bsub_int_stats.sh, bsub_int_aggregate.sh, bsub_int_correct.sh   (submit in order, wait for each before submitting the next)
```

#### BaSiC-only:
```python
sp.create_quartile_histograms()   # -> bsub_command.sh      (submit it, wait)
sp.save_qstack()                  # -> qstacks/camera{N}.tiff  (inspect these)
sp.run_basic()                    # -> results_root/camera{N}/{Flat,Dark}-field.tif
sp.write_correction_script()      # -> bsub_correction.sh   (submit it)
```

#### Intensity-only:
```python
sp.create_intensity_correction_script()       # -> bsub_int_stats.sh, bsub_int_aggregate.sh, bsub_int_correct.sh   (submit in order, wait for each before submitting the next)
```

### Local Workflow

The `bsub` scripts above exist to spread the work across a cluster. On a workstation, or
for a dataset small enough to sit on one machine, run the whole thing in one process:

```bash
python -m spotlight run basic        # emptiness, stats, qstack, basic, correct
python -m spotlight run intensity    # emptiness, int-stats, int-aggregate, correct
python -m spotlight run both         # both corrections, applied in ONE pass
python -m spotlight run both --dry-run          # list the stages and units first
python -m spotlight run basic --stop-after qstack   # stop and look at the qstack
python -m spotlight run basic --start-at basic      # resume after inspecting it
```

## Modes

See config.py for a complete list of params. Below is a compilation of some that are dataset dependent.

### BaSiC Parameters
```"autotune": bool (default true)``` \
&emsp;Chooses `lambda` (the flat-field smoothness weight) by grid search instead of taking the `l1/800` default, following [BaSiCPy](https://www.biorxiv.org/content/10.64898/2026.04.28.721386v1) — lambda has the largest single effect on the fit and is genuinely dataset-dependent. Candidates are multiples of that default, scored by the entropy of the corrected stack plus a penalty on high-frequency structure left in the flat field. Costs ~11 extra fits per camera, all at 128x128, so it is cheap regardless of tile size. Setting `lambda` to a nonzero value turns it off — an explicit lambda is honoured as chosen. Turn it off to reproduce a fit made before this existed.

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
&emsp;If your dataset is already reasonably well-stitched, use "intersection." "Independent" is use for unstitched or poorly stitched data.

```"gain_lambda": Float between 0 and 1``` \
&emsp;This is the intensity matching regularization term. If you would like your gain mappings to be almost unregularized (for ex, highly determined systems like camera-based workflows), we recommend 1e-6. 0.01 or 0.1 is typical for tile-based workflows.

```"gain_floor": "otsu" (default), "li", "pooled", or a number``` \
&emsp;This is the minimum intensity cutoff used to calculate the gain mappings. Otsu is the most efficient and is ok for most workflows. However, with very sparse data (i.e. a small specimen that does not fill the whole tile), "li" works better.

```"tile_threshold": "otsu" (default), "li", "pooled", or a number``` \
&emsp;This is the minimum intensity cutoff used during the intensity correction step; i.e. all voxels with an intensity below this threshold will be considered to be background and not adjusted. Typically matches gain_floor, but it can be helpful to lower this if some foreground voxels are not being adjusted.
