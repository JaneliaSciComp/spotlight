"""spotlight -- illumination correction for large microscopy datasets.

Workflow, from a dataset's own working directory (the one holding its
`LocalPreferences.toml`):

1. `set_config(...)` / `set_basic_config(...)`
2. `create_quartile_histograms()` -> `bsub_command.sh`, then submit it
3. `save_qstack()` -> `qstacks/camera{N}.tiff` (inspect these)
4. `run_basic()` -> `{results_root}/camera{N}/{Flat,Dark}-field.tif`
5. either `write_correction_script()` -> `bsub_correction.sh`, or -- if you also want the
   per-tile intensity correction -- `create_intensity_correction_script()` alone, which
   applies both in one pass.
"""

from .config import (
    basic_params, camera_setups, load_config, num_cameras, set_basic_config, set_config,
)

__all__ = [
    "load_config", "set_config", "set_basic_config", "basic_params",
    "camera_setups", "num_cameras",
    "create_quartile_histograms", "save_qstack", "run_basic",
    "write_correction_script", "create_intensity_correction_script",
    "calculate_camera_stats", "apply_correction_chunked", "basic_estimate",
]


# Imported lazily: `import spotlight` should not pay for tensorstore/scipy/tifffile just
# to read a config.
def __getattr__(name):
    if name in ("create_quartile_histograms", "write_correction_script",
                "create_intensity_correction_script", "measure_emptiness"):
        from . import scripts
        return getattr(scripts, name)
    if name in ("save_qstack", "save_qstack_camera"):
        from . import qstack
        return getattr(qstack, name)
    if name in ("run_basic", "run_basic_camera", "basic_estimate"):
        from . import basic
        return getattr(basic, name)
    if name == "calculate_camera_stats":
        from . import quantiles
        return quantiles.calculate_camera_stats
    if name == "apply_correction_chunked":
        from . import correct
        return correct.apply_correction_chunked
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
