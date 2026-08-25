"""Parity tests for the BaSiC port.

Split into primitives, then norms, then the converged fit, so a failure says WHICH step
diverged instead of just "the fields differ". The primitives are where a line-by-line
port most easily goes wrong -- the DCT scaling and the resize half-pixel convention are
both places where transcribing Julia faithfully gives the WRONG answer, because scipy and
skimage already apply what Julia applies by hand.
"""

import numpy as np
import pytest

from spotlight import basic

from golden_io import have_golden, load_bin, load_json

pytestmark = pytest.mark.skipif(
    not have_golden(),
    reason="run tests/gen_golden.jl (needs Julia) to produce the reference values",
)


# ─── primitives ───────────────────────────────────────────────────────────────


def test_dct_matches_fftw():
    """Julia reaches the orthonormal DCT from FFTW's UNNORMALISED REDFT10 times its own
    weights; scipy's norm="ortho" is already that, so transcribing the weights would scale
    the whole transform wrong.

    `atol` is set against the transform's own scale (coefficients reach ~2.8e3), not
    against zero: FFTW and pocketfft accumulate in a different order, so the handful of
    near-zero coefficients disagree in their last float32 bits. A relative bound alone
    would fail on those while saying nothing about the coefficients that matter.
    """
    x = load_bin("dct_input")
    ref = load_bin("dct_forward")
    np.testing.assert_allclose(basic.dct2_ortho(x), ref,
                               rtol=1e-5, atol=1e-6 * np.abs(ref).max())


def test_dct_round_trips():
    x = load_bin("dct_input")
    ref = load_bin("dct_roundtrip")
    np.testing.assert_allclose(basic.idct2_ortho(basic.dct2_ortho(x)), ref,
                               rtol=1e-5, atol=1e-6 * np.abs(ref).max())
    np.testing.assert_allclose(basic.idct2_ortho(basic.dct2_ortho(x)), x,
                               rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("name,out", [
    ("resize_ramp_down", (8, 3)),
    ("resize_ramp_up", (77, 3)),
])
def test_resize_1d_ramp(name, out):
    """A linear ramp, so a half-pixel misalignment changes every value -- a smooth blob
    would hide it."""
    got = basic.imresize(load_bin("resize_ramp_in"), out)
    np.testing.assert_allclose(got, load_bin(name), rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize("name,out", [
    ("resize_2d_down", (7, 5)),
    ("resize_2d_up", (40, 33)),
])
def test_resize_2d(name, out):
    got = basic.imresize(load_bin("resize_2d_in"), out)
    np.testing.assert_allclose(got, load_bin(name), rtol=1e-6, atol=1e-4)


def test_resize_is_identity_at_the_same_size():
    a = load_bin("resize_2d_in")
    np.testing.assert_allclose(basic.imresize(a, a.shape), a, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["shrink_scalar", "shrink_array"])
def test_shrink(name):
    g = load_json(name + ".json")
    x = np.asarray(g["input"], dtype=np.float32)
    t = g["t"]
    thr = np.float32(t) if np.isscalar(t) else np.asarray(t, dtype=np.float32)
    np.testing.assert_allclose(basic.shrink(x, thr),
                               np.asarray(g["output"], dtype=np.float32),
                               rtol=1e-6, atol=1e-6)


# ─── initialisation ───────────────────────────────────────────────────────────


def test_norms_match():
    """The spectral norm sets the ALM's initial penalty, so a LAPACK disagreement here
    would shift every subsequent iteration. Checked separately from the fit so the two
    causes stay distinguishable."""
    g = load_json("basic_norms.json")
    stack = load_bin("basic_stack")
    d = stack / np.float32(stack.mean())
    d.sort(axis=2)
    d_flat = d.reshape(-1, d.shape[2])
    assert float(stack.mean()) == pytest.approx(g["global_mean"], rel=1e-6)
    assert float(np.linalg.svd(d_flat, compute_uv=False)[0]) == pytest.approx(
        g["norm_two"], rel=1e-5)
    assert float(np.linalg.norm(d_flat.reshape(-1))) == pytest.approx(g["norm_D"], rel=1e-6)
    assert float(d[:, :, 0].mean()) == pytest.approx(g["darkfield_limit"], rel=1e-6)


def test_mean_image_matches():
    """`mean_img` warm-starts W_hat, so it is the fit's actual starting point."""
    stack = load_bin("basic_stack")
    d = stack / np.float32(stack.mean())
    d.sort(axis=2)
    np.testing.assert_allclose(d.mean(axis=2), load_bin("basic_mean_img"),
                               rtol=1e-5, atol=1e-6)


# ─── the fit ──────────────────────────────────────────────────────────────────


def test_converged_fit_matches_julia():
    """The full ALM, float32 and iterative, against the Julia fields.

    Looser than the primitives on purpose: the convergence tests are data-dependent, so a
    1e-7 drift can change the last iteration taken. The flat field is what every consumer
    divides by, so it gets the tight bound; the darkfield is an additive pedestal in
    counts, so it gets an absolute one.
    """
    stack = load_bin("basic_stack")
    flat, dark = basic.basic_estimate(stack, estimate_darkfield=True, working_size=0)
    np.testing.assert_allclose(flat, load_bin("basic_flat"), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(dark, load_bin("basic_dark"), rtol=2e-2, atol=1.0)


def test_flatfield_is_mean_one():
    """Independent of the golden: the flat field is a mean-1 multiplier by construction,
    so this catches a normalisation dropped in the port even if the goldens are stale."""
    flat, _ = basic.basic_estimate(load_bin("basic_stack"), working_size=0)
    assert float(flat.mean()) == pytest.approx(1.0, rel=1e-5)
    assert (flat >= 0).all()


def test_fixed_darkfield_is_held():
    """With an override the darkfield must come back at the supplied level, and the flat
    field must have been fitted GIVEN it -- not fitted freely and then overwritten."""
    stack = load_bin("basic_stack")
    flat, dark = basic.basic_estimate(stack, estimate_darkfield=True, working_size=0,
                                      darkfield_override=120.0)
    np.testing.assert_allclose(flat, load_bin("basic_flat_fixed_dark"), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(dark, load_bin("basic_dark_fixed_dark"), rtol=1e-3, atol=1e-2)
    assert float(dark.mean()) == pytest.approx(120.0, rel=1e-3)


def test_output_size_retargets_the_single_resize():
    """Asking for a larger frame must not cost a second interpolation -- the fit happens
    on the working grid and the one resize out of it goes straight to `output_size`."""
    stack = load_bin("basic_stack")
    h, w, _ = stack.shape
    flat, dark = basic.basic_estimate(stack, working_size=0, output_size=(2 * h, 3 * w))
    assert flat.shape == (2 * h, 3 * w)
    assert dark.shape == (2 * h, 3 * w)


def test_all_zero_stack_is_rejected():
    with pytest.raises(ValueError, match="all-zero"):
        basic.basic_estimate(np.zeros((8, 6, 21), np.float32))


# ─── the darkfield-collapse warning ───────────────────────────────────────────


def _run_camera(tmp_path, monkeypatch, measured, **overrides):
    """Fit one synthetic camera through `run_basic_camera`, bypassing the store.

    Goes through the driver rather than `basic_estimate` directly, because the guard under
    test lives there -- calling the estimator would skip it entirely.
    """
    from spotlight import basic
    from spotlight.config import BASIC_DEFAULTS

    monkeypatch.setattr(basic, "measured_background_level", lambda cfg, cam: measured)
    stack = load_bin("basic_stack")
    monkeypatch.setattr("spotlight.qstack.load_qstack", lambda c, cam: stack)
    monkeypatch.setattr("spotlight.qstack.qstack_frame_size", lambda c, s: stack.shape[:2])

    params = dict(BASIC_DEFAULTS)
    # autotune off: these tests are about the collapse warning, and searching lambda would
    # cost ~11 extra fits to reach the same driver code path.
    params.update(working_size=0, max_iterations=20, max_reweighting_iterations=2,
                  autotune=False)
    params.update(overrides)
    cfg = {"results_root": str(tmp_path), "qstacks_dir": str(tmp_path),
           "basic_stats_level": 0, "input_format": "zarr2",
           "last_setup": 0, "setups_per_camera": 1, "setup_ids": []}
    basic.run_basic_camera(cfg, 0, params)
    # The run must actually have completed, or an empty capsys would pass vacuously.
    assert (tmp_path / "camera1" / "Flat-field.tif").is_file()


def test_collapse_warning_is_silent_when_darkfield_is_not_estimated(tmp_path, capsys,
                                                                    monkeypatch):
    """With `estimate_darkfield` off the field is zero BY REQUEST, so it has not collapsed
    -- warning about it sends the reader to change a setting they chose. Measured
    background is set absurdly high so the check would certainly fire if it ran.
    """
    _run_camera(tmp_path, monkeypatch, measured=1e6, estimate_darkfield=False)
    assert "collapsed" not in capsys.readouterr().out


def test_collapse_warning_fires_and_names_both_remedies(tmp_path, capsys, monkeypatch):
    """When BaSiC did fit one and it collapsed, say so -- and name the two settings."""
    _run_camera(tmp_path, monkeypatch, measured=1e6, estimate_darkfield=True)
    out = capsys.readouterr().out
    assert "darkfield collapsed" in out
    assert "estimate_darkfield" in out and "override_darkfield" in out
    # The old message explained vignetting at length; it should be gone.
    assert "VIGNETTING" not in out.upper()
