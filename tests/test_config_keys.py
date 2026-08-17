"""Every key in `DEFAULTS` must actually be read somewhere.

A key that appears in the config list but reaches no code is worse than an absent one:
it is discoverable, plausible, and silently inert -- you set it, nothing changes, and
there is no error to notice. So this file checks the two directions separately:

  * every DEFAULTS key is read by the package (or is a documented pure-input key)
  * the classification/emptiness/apply gates actually change behaviour when overridden
"""

import ast
import pathlib

import numpy as np
import pytest

from spotlight import config, tilestats

PKG = pathlib.Path(tilestats.__file__).parent


def _keys_read_in_source():
    """Every string used as cfg["k"] / cfg.get("k") anywhere in the package."""
    found = set()
    for f in PKG.glob("*.py"):
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                    and isinstance(n.slice, ast.Constant)
                    and isinstance(n.slice.value, str)):
                found.add(n.slice.value)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("get", "setdefault") and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)):
                found.add(n.args[0].value)
    return found


# Keys consumed by the bsub generator's f-strings or by `_LIMIT_KEYS` indirection
# rather than by a literal cfg["..."], so the AST scan cannot see them.
INDIRECT = set(tilestats._LIMIT_KEYS) | {"setups_per_row"}


def test_every_config_key_is_read_somewhere():
    read = _keys_read_in_source() | INDIRECT
    dead = sorted(k for k in config.DEFAULTS if k not in read)
    assert not dead, f"in DEFAULTS but never read (silently inert): {dead}"


# ─── the overrides must actually bite ────────────────────────────────────────


def _stats(n_fg=1000, n_vox=1_000_000, all_std=50.0, empty_area=0.5):
    return {"n_foreground": n_fg, "n_voxels": n_vox, "mean": 100.0, "std": 20.0,
            "all_std": all_std, "empty_area": empty_area, "setup": 0}


def test_min_tile_fraction_moves_the_empty_boundary():
    """The gate that dropped a real tile from the gain solve and left it uncorrected."""
    st = _stats(n_fg=1000, n_vox=1_000_000)          # 0.1%, exactly the default
    assert tilestats._classify(st, tilestats.limits({"min_tile_fraction": 0.0005})) != "empty"
    assert tilestats._classify(st, tilestats.limits({"min_tile_fraction": 0.05})) == "empty"


def test_min_background_area_switches_bimodal_and_uniform():
    """`uniform` rescales every pixel; `bimodal` only above the threshold."""
    st = _stats(empty_area=0.10)
    assert tilestats._classify(st, tilestats.limits({"min_background_area": 0.02})) == "bimodal"
    assert tilestats._classify(st, tilestats.limits({"min_background_area": 0.50})) == "uniform"


def test_min_uniform_std_sends_an_all_noise_tile_to_empty():
    st = _stats(all_std=12.0)
    assert tilestats._classify(st, tilestats.limits({"min_uniform_std": 5.0})) != "empty"
    assert tilestats._classify(st, tilestats.limits({"min_uniform_std": 20.0})) == "empty"


def test_limits_falls_back_to_the_module_constants():
    """Callable with no config at all -- tests and bench scripts have none."""
    lim = tilestats.limits()
    assert lim["min_tile_fraction"] == tilestats.MIN_FG_FRACTION
    assert lim["max_gain_scale"] == tilestats.MAX_SCALE
    assert tilestats.limits({}) == lim
    assert tilestats.limits(None) == lim


def test_the_defaults_match_the_module_constants():
    """The config list states the default; if the constant moves and this does not, the
    documented default becomes a lie."""
    for key, const in tilestats._LIMIT_KEYS.items():
        assert config.DEFAULTS[key] == getattr(tilestats, const), key


def test_an_override_of_none_is_ignored_rather_than_crashing():
    """tomli cannot express None, but a cfg built in Python can carry it."""
    assert tilestats.limits({"min_tile_fraction": None}) == tilestats.limits()
