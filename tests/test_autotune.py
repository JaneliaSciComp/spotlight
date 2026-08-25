"""Tests for the lambda autotune.

Golden-free on purpose -- the autotune is not part of the Julia port, so there is nothing
to be in parity with. The cost terms are checked against synthetic fields where the right
answer is known by construction, then the search is checked for the properties that make
it usable at all: it stays on the grid, and a multiplier of 1.0 reproduces `_auto_lambda`.
"""

import numpy as np
import pytest

from spotlight import basic


def _smooth_flat(n=128, amp=0.3):
    """A vignette: mean ~1, low-frequency only."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1.0)
    f = 1.0 + amp * np.cos(np.pi * (x - 0.5)) * np.cos(np.pi * (y - 0.5))
    return (f / f.mean()).astype(np.float32)


def _stack(n=64, frames=12, seed=0):
    """A vignetted stack with uncorrelated foreground blobs and a flat pedestal."""
    rng = np.random.default_rng(seed)
    flat = _smooth_flat(n)
    signal = rng.random((n, n, frames), dtype=np.float32) ** 8 * 400.0
    return (signal + 100.0) * flat[:, :, None] + 50.0, flat


# ─── the cost terms ───────────────────────────────────────────────────────────


def test_fourier_l0_separates_smooth_from_noisy():
    """The term the hinge is built on: small for a vignette, near 1 once high-frequency
    structure is in the field.

    Deliberately a RELATIVE bound. A real vignette scores ~0.015, an order of magnitude
    above `_FOURIER_HINGE`: a smooth hump is not band-limited in the DCT basis, its
    coefficients decay only as 1/u^2, and the unnormalised threshold this constant is
    calibrated against is reached by that tail. So the term does not separate "smooth"
    from "not smooth" on an absolute scale, it separates candidates from each other. That
    is the published calibration, verified against BaSiCPy's own `fourier_L0_norm`.
    """
    smooth = _smooth_flat()
    noisy = smooth + np.random.default_rng(1).normal(0, 0.2, smooth.shape).astype(np.float32)
    assert basic.fourier_l0(smooth) < 0.05 < 0.5 < basic.fourier_l0(noisy)
    assert basic.fourier_l0(np.ones((128, 128), np.float32)) == 0.0


def test_fourier_l0_is_size_independent():
    """Scored at the calibration size, so the same field resized must score the same --
    otherwise the constants would mean different things per camera."""
    smooth = _smooth_flat(128)
    assert basic.fourier_l0(basic.imresize(smooth, (512, 512))) == pytest.approx(
        basic.fourier_l0(smooth), abs=2e-3)


def test_entropy_prefers_the_compact_distribution():
    rng = np.random.default_rng(2)
    tight = rng.normal(100.0, 2.0, 200_000)
    broad = rng.normal(100.0, 20.0, 200_000)
    assert basic.histogram_entropy(tight, 0.0, 200.0) < basic.histogram_entropy(
        broad, 0.0, 200.0)


def test_entropy_of_an_empty_window_is_infinite():
    """A candidate whose corrected stack misses the window entirely must lose, not crash
    or silently score zero -- zero would make it the winner."""
    assert basic.histogram_entropy(np.zeros(10), 5.0, 10.0) == np.inf


def test_cost_penalises_a_noisy_flatfield():
    """Same corrected stack, two flat fields: the hinge has to dominate."""
    corrected = _stack()[0]
    smooth = _smooth_flat()
    noisy = smooth + np.random.default_rng(3).normal(0, 0.2, smooth.shape).astype(np.float32)
    v_range = float(np.ptp(np.quantile(corrected, [0.01, 0.99])))
    assert (basic.autotune_cost(corrected, noisy, v_range)
            > basic.autotune_cost(corrected, smooth, v_range))


# ─── the search ───────────────────────────────────────────────────────────────


def test_autotune_returns_a_grid_multiple_of_the_default():
    """The multiplier must land on the published grid, and `lam` must be that multiple of
    the l1/800 default -- so `autotune=false` and a 1.0 result mean the same fit."""
    images, _ = _stack()
    lam, mult = autotune = basic.autotune_lambda(
        images, max_iterations=20, max_reweighting_iterations=2)
    assert mult in basic.AUTOTUNE_FINE or mult in basic.AUTOTUNE_COARSE
    small = np.stack([basic.imresize(images[:, :, k], (128, 128))
                      for k in range(images.shape[2])], axis=2)
    mean_img = small.mean(axis=2) / np.float32(small.mean())
    lam_default, _ = basic._auto_lambda(mean_img, 128, 128, 0.0, 0.0)
    assert lam == pytest.approx(mult * float(lam_default), rel=1e-5)
    assert autotune[0] > 0.0


def test_autotune_rejects_a_lambda_that_overfits_the_foreground():
    """The point of the exercise: the smallest multiplier on the grid leaves foreground
    texture in the flat field, so the search must not choose it."""
    images, _ = _stack(seed=4)
    _, mult = basic.autotune_lambda(images, max_iterations=20,
                                    max_reweighting_iterations=2)
    assert mult > min(basic.AUTOTUNE_COARSE)


def test_autotune_bounds_the_number_of_fits():
    """~11 fits per camera, not the whole fine grid. Counted, because the cost of this
    search is the only reason it could be a bad default."""
    calls = []
    real = basic.basic_estimate

    def counting(images, **kwargs):
        calls.append(kwargs.get("lam"))
        return real(images, **kwargs)

    images, _ = _stack(seed=5)
    basic.basic_estimate = counting
    try:
        basic.autotune_lambda(images, max_iterations=20, max_reweighting_iterations=2)
    finally:
        basic.basic_estimate = real
    assert len(calls) <= 13, f"{len(calls)} candidate fits"
    assert len(calls) == len(set(calls)), "a candidate was fitted twice"


def test_driver_autotunes_by_default(tmp_path, monkeypatch, capsys):
    """Default-on is the whole request, so it is worth pinning: the driver must reach
    `basic_estimate` with the tuned lambda, not with 0.0."""
    from spotlight.config import BASIC_DEFAULTS

    images, _ = _stack(seed=6)
    monkeypatch.setattr("spotlight.qstack.load_qstack", lambda c, cam: images)
    monkeypatch.setattr("spotlight.qstack.qstack_frame_size", lambda c, s: images.shape[:2])
    seen = {}
    real = basic.basic_estimate

    def record(imgs, **kwargs):
        seen.setdefault("lam", kwargs["lam"])
        return real(imgs, **kwargs)

    monkeypatch.setattr(basic, "basic_estimate", record)
    params = dict(BASIC_DEFAULTS)
    params.update(working_size=0, max_iterations=20, max_reweighting_iterations=2)
    assert params["autotune"] is True
    basic.run_basic_camera({"results_root": str(tmp_path), "basic_stats_level": 0,
                            "input_format": "zarr2", "last_setup": 0,
                            "setups_per_camera": 1, "setup_ids": []}, 0, params)
    assert seen["lam"] > 0.0
    assert "autotuned lambda" in capsys.readouterr().out
