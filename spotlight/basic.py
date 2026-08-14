"""BaSiC flat-field / dark-field estimation.

Implements the BaSiC algorithm (Peng et al., Nature Communications 2017):
  "A BaSiC tool for background and shading correction of optical microscopy images"

A line-by-line port of `src/basic.jl`, deliberately: the goal is that this produces the
same fields as the Julia version on the same qstack, so the two can be diffed. Where a
numpy idiom would reorder the arithmetic, the Julia order wins.

Given a per-camera quantile stack (written by `save_qstack`), estimates a flat-field and
a dark-field and writes them where the apply stages expect:
  {results_root}/camera{N}/Flat-field.tif
  {results_root}/camera{N}/Dark-field.tif
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.fft import dctn, idctn

from . import config as _config
from . import qstack as _qstack

__all__ = ["basic_estimate", "run_basic", "run_basic_camera", "imresize"]


# ─── Orthonormal 2D DCT-II ────────────────────────────────────────────────────
#
# Julia builds this out of FFTW's UNNORMALISED r2r/REDFT10 and applies its own
# `DCTWeights ./ 4` to reach the orthonormal scaling. scipy's `norm="ortho"` is already
# that scaling, so the weights must NOT be transcribed -- porting those two lines
# faithfully is precisely how this goes wrong.


def dct2_ortho(x):
    return dctn(x, type=2, norm="ortho").astype(np.float32)


def idct2_ortho(y):
    return idctn(y, type=2, norm="ortho").astype(np.float32)


def shrink(x, t):
    """Soft threshold (the L1 proximal operator), in place: sign(x) * max(|x| - t, 0)."""
    np.multiply(np.sign(x), np.maximum(np.abs(x) - t, 0.0), out=x)
    return x


# ─── Resize ───────────────────────────────────────────────────────────────────


def _axis_weights(n_in, n_out):
    """`ImageTransformations.imresize`'s coordinate mapping for one axis.

    That function maps the OUTER CORNERS of the two images to each other -- a pixel is
    treated as a sensor integrating over `i ± 0.5` -- giving `sf = n_in / n_out` and a
    source coordinate of `sf * (i - 0.5) + 0.5` for 1-based `i`, i.e. `sf * (i + 0.5) -
    0.5` 0-based. Coordinates are clamped into range, which only bites when upsampling
    (`sf < 1`), where the first and last output pixels fall outside the source grid.

    No anti-aliasing, matching Julia: BaSiC's downsample to `working_size` relies on it.
    Weights are computed in float64 because Julia's `sf`/`offset` are, then applied to
    float32 data.
    """
    sf = n_in / n_out
    c = np.clip(sf * (np.arange(n_out) + 0.5) - 0.5, 0.0, n_in - 1.0)
    lo = np.floor(c).astype(np.intp)
    if n_in > 1:
        lo = np.minimum(lo, n_in - 2)
        hi = lo + 1
    else:
        hi = lo
    return lo, hi, (c - lo)


def _interp_axis(a, axis, n_out):
    if a.shape[axis] == n_out:
        return a
    lo, hi, g = _axis_weights(a.shape[axis], n_out)
    shape = [1] * a.ndim
    shape[axis] = n_out
    g = g.reshape(shape)
    # (1 - g) * a_lo + g * a_hi, matching Interpolations.jl's BSpline(Linear()) weight
    # form rather than the algebraically-equal `a + g * (b - a)`.
    return (1.0 - g) * np.take(a, lo, axis=axis) + g * np.take(a, hi, axis=axis)


def imresize(a, out_shape):
    """Bilinear resize of the leading two axes, matching `ImageTransformations.imresize`."""
    out = np.asarray(a, dtype=np.float64)
    for axis, n_out in enumerate(out_shape):
        out = _interp_axis(out, axis, n_out)
    return np.ascontiguousarray(out, dtype=np.float32)


# ─── Core algorithm ───────────────────────────────────────────────────────────


def _auto_lambda(mean_img, h, w, lam, lam_dark):
    """Lambdas from the L1 of the mean image's DCT, when not supplied.

    The 800/2000 constants are calibrated for a 128x128 working image, so `l1_dct` is
    always computed at 128x128 regardless of `working_size` -- the threshold/DC ratio
    then depends only on N, not on H*W.
    """
    if lam == 0.0 or lam_dark == 0.0:
        m = mean_img if (h == 128 and w == 128) else imresize(mean_img, (128, 128))
        l1_dct = float(np.abs(dct2_ortho(m)).sum())
        lam = l1_dct / 800.0 if lam == 0.0 else lam
        lam_dark = l1_dct / 2000.0 if lam_dark == 0.0 else lam_dark
    return np.float32(lam), np.float32(lam_dark)


def _update_darkfield(a_offset, b_offset, a1_offset, b1_coeff, f, r1, a1_coeff,
                      darkfield_limit, lambda_darkfield, ent2, mu):
    """One darkfield update. Mutates a_offset/b_offset/a1_offset/b1_coeff; returns B1_offset."""
    # Frames with below-average illumination carry the most darkfield signal.
    valid = a1_coeff < 1.0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.float32(0.0)

    f_mean = np.float32(f.mean())
    f_high = f > (f_mean - 1e-6)
    f_low = f < (f_mean + 1e-6)
    g = np.float32(r1.mean())
    safe_gr1 = np.float32(1.0) if abs(g) < np.finfo(np.float32).eps else g

    # Guard: if f is spatially constant, both masks cover every pixel and the
    # high/low contrast is meaningless (float32 mean error can exceed 1e-6 on
    # large arrays, so this is reachable in practice, not just in theory).
    n_high, n_low = int(f_high.sum()), int(f_low.sum())
    if n_high == 0 or n_low == 0:
        b1_coeff[:] = 0.0
    else:
        for k in range(a1_coeff.size):
            if valid[k]:
                frame = r1[:, :, k]
                b1_coeff[k] = (frame[f_high].mean() - frame[f_low].mean()) / safe_gr1
            else:
                b1_coeff[k] = 0.0

    # Least-squares scalar darkfield offset (MATLAB B1_offset), clamped to
    # [0, darkfield_limit / f_mean] to keep it physical.
    a1_valid = a1_coeff[valid]
    b1_valid = b1_coeff[valid]
    kn = np.float32(n_valid)
    t1, t2 = np.float32((a1_valid ** 2).sum()), np.float32(a1_valid.sum())
    t3, t4 = np.float32(b1_valid.sum()), np.float32((a1_valid * b1_valid).sum())
    t5 = t2 * t3 - kn * t4
    b1_off = np.float32(0.0) if abs(t5) < np.finfo(np.float32).eps else (t1 * t3 - t2 * t4) / t5
    b1_off = np.float32(min(max(b1_off, 0.0),
                            darkfield_limit / max(f_mean, np.finfo(np.float32).eps)))

    # The part anti-correlated with the flatfield shape.
    b_offset[...] = b1_off * f_mean - b1_off * f
    a1_offset[...] = r1[:, :, valid].mean(axis=2) - a1_valid.mean() * f
    a1_offset -= a1_offset.mean()
    a_offset[...] = a1_offset - b_offset

    # Smooth and sparsify via DCT, then again in the image domain.
    w_off = dct2_ortho(a_offset)
    shrink(w_off, lambda_darkfield / (ent2 * mu))
    a_offset[...] = idct2_ortho(w_off)
    shrink(a_offset, lambda_darkfield / (ent2 * mu))
    a_offset += b_offset
    return b1_off


def _alm_loop(w_hat, e, a1_coeff, a_offset, y1, d, w_coeff, f, a1_hat, r_w, r1, z,
              b1_coeff, b_offset, a1_offset, *, mu_init, mu_bar, rho, ent1, ent2,
              lam, lambda_darkfield, darkfield_limit, estimate_darkfield,
              max_iterations, optimization_tol, norm_d):
    """The inner ALM loop. Every array argument is modified in place."""
    h, w, n = d.shape
    mu = mu_init
    b1_off = np.float32(0.0)
    coeff_view = a1_coeff.reshape(1, 1, n)
    offset_view = a_offset.reshape(h, w, 1)

    for _ in range(max_iterations):
        # Low-rank model: a rank-1 flatfield scaled per frame, plus the additive darkfield.
        np.maximum(idct2_ortho(w_hat), 0.0, out=f)
        np.multiply(f[:, :, None], coeff_view, out=a1_hat)
        a1_hat += offset_view

        # W_hat update. `e` doubles as scratch for the per-frame residual before averaging.
        e *= -1.0
        e += d
        e -= a1_hat
        e += y1 / mu
        np.mean(e, axis=2, out=r_w)
        w_hat += dct2_ortho(r_w / ent1)
        shrink(w_hat, lam / (ent1 * mu))

        np.maximum(idct2_ortho(w_hat), 0.0, out=f)
        np.multiply(f[:, :, None], coeff_view, out=a1_hat)
        a1_hat += offset_view

        # E update: proximal step on ‖W_coeff ⊙ E‖₁ subject to D = A1_hat + E.
        np.subtract(d, a1_hat, out=e)
        e += y1 / mu
        shrink(e, w_coeff / (ent1 * mu))

        # Per-frame illumination scale.
        np.subtract(d, e, out=r1)
        global_r1 = np.float32(r1.mean())
        for k in range(n):
            a1_coeff[k] = max(np.float32(r1[:, :, k].mean()) / global_r1, 0.0)
        np.multiply(f[:, :, None], coeff_view, out=a1_hat)
        a1_hat += offset_view

        if estimate_darkfield:
            b1_off = _update_darkfield(a_offset, b_offset, a1_offset, b1_coeff, f, r1,
                                       a1_coeff, darkfield_limit, lambda_darkfield, ent2, mu)
            np.multiply(f[:, :, None], coeff_view, out=a1_hat)
            a1_hat += offset_view

        # Dual ascent and penalty growth.
        np.subtract(d, a1_hat, out=z)
        z -= e
        y1 += mu * z
        mu = min(mu * rho, mu_bar)

        if np.linalg.norm(z.reshape(-1)) / norm_d < optimization_tol:
            break

    return b1_off


def _update_weights(w_coeff, e, f, a1_coeff, a_offset, epsilon):
    """MATLAB: weight[h,w,k] = 1 / (|E / mean_hw(A1_hat[:,:,k])| + eps), mean-normalised."""
    h, w, n = e.shape
    frame_means = (np.float32(f.mean()) * a1_coeff + np.float32(a_offset.mean())).reshape(1, 1, n)
    w_new = 1.0 / (np.abs(e / (frame_means + 1e-6)) + epsilon)
    np.multiply(w_new, np.float32(h * w * n) / np.float32(w_new.sum()), out=w_coeff)


def basic_estimate(images, *, lam=0.0, lambda_darkfield=0.0, estimate_darkfield=True,
                   max_iterations=500, optimization_tol=1e-6, reweight_tol=1e-3,
                   max_reweighting_iterations=10, epsilon=0.1, working_size=0,
                   darkfield_override=None, output_size=None, _norm_two=None):
    """Estimate flat-field and dark-field from an (H, W, N) float32 stack.

    Returns `(flatfield, darkfield)` as float32, flatfield normalised to mean ~1.

    `output_size` is the (rows, cols) the fields come back at; `None` means the input
    stack's own frame. The fit always happens on the `working_size` grid and the single
    resize out of it goes straight to `output_size`, so asking for a frame LARGER than
    the input stack costs no extra interpolation -- it just retargets the resize that
    already happens. That is what lets a `basic_stats_level > 0` fit publish full-size
    fields without a second pass through the coarse frame.

    `_norm_two` overrides the spectral norm. It exists for parity testing: injecting
    Julia's value tells a LAPACK disagreement apart from a port bug without a bisect.
    """
    images = np.ascontiguousarray(images, dtype=np.float32)
    h, w, n = images.shape
    h_orig, w_orig = h, w

    # A supplied darkfield is HELD FIXED and the flat field fitted given it: the additive
    # term enters `a1_hat` exactly as a fitted one would, so every other update sees the
    # pedestal accounted for. Strictly better than fitting both and replacing afterwards,
    # which leaves the flat field solved under the wrong additive assumption.
    fixed_darkfield = darkfield_override is not None
    estimate_darkfield = bool(estimate_darkfield) and not fixed_darkfield

    if working_size > 0 and (h != working_size or w != working_size):
        images = np.stack([imresize(images[:, :, k], (working_size, working_size))
                           for k in range(n)], axis=2)
        h = w = working_size

    global_mean = np.float32(images.mean())
    if global_mean < np.finfo(np.float32).eps:
        raise ValueError("BaSiC: image stack is all-zero")
    d = images / global_mean
    d.sort(axis=2)                       # matches MATLAB `sort(D,3)`
    mean_img = d.mean(axis=2)

    lam, lambda_darkfield = _auto_lambda(mean_img, h, w, lam, lambda_darkfield)

    d_flat = d.reshape(h * w, n)
    norm_two = (np.float32(np.linalg.svd(d_flat, compute_uv=False)[0])
                if _norm_two is None else np.float32(_norm_two))
    norm_d = np.float32(np.linalg.norm(d_flat.reshape(-1)))
    mu_init = np.float32(12.5) / norm_two
    mu_bar = mu_init * np.float32(1e7)
    rho = np.float32(1.5)
    ent1 = np.float32(1.0)               # step-size denominator for the flatfield
    ent2 = np.float32(10.0)              # step-size denominator for the darkfield

    # After the sort, d[:, :, 0] holds the per-pixel minimum across frames.
    darkfield_limit = np.float32(d[:, :, 0].mean())

    w_hat = dct2_ortho(mean_img)         # warm-started from the mean image
    e = np.zeros((h, w, n), np.float32)
    a1_coeff = np.ones(n, np.float32)
    a_offset = np.zeros((h, w), np.float32)
    if fixed_darkfield:
        # `darkfield_override` arrives in the raw counts of `images`; d is normalised.
        a_offset[:] = np.float32(darkfield_override) / global_mean
    w_coeff = np.ones((h, w, n), np.float32)

    f = np.zeros((h, w), np.float32)
    a1_hat = np.zeros((h, w, n), np.float32)
    r_w = np.zeros((h, w), np.float32)
    r1 = np.zeros((h, w, n), np.float32)
    z = np.zeros((h, w, n), np.float32)
    y1 = np.zeros((h, w, n), np.float32)
    ff_curr = np.zeros((h, w), np.float32)
    b1_coeff = np.zeros(n if estimate_darkfield else 0, np.float32)
    b_offset = np.zeros((h, w) if estimate_darkfield else (0, 0), np.float32)
    a1_offset = np.zeros((h, w) if estimate_darkfield else (0, 0), np.float32)

    flatfield_prev = np.ones((h, w), np.float32)
    darkfield_prev = np.zeros((h, w), np.float32)

    for _ in range(max_reweighting_iterations):
        # Reset together -- MATLAB resets both on each call to inexact_alm_rspca_l1.
        y1[...] = 0.0
        e[...] = 0.0

        b1_off = _alm_loop(w_hat, e, a1_coeff, a_offset, y1, d, w_coeff, f, a1_hat,
                           r_w, r1, z, b1_coeff, b_offset, a1_offset,
                           mu_init=mu_init, mu_bar=mu_bar, rho=rho, ent1=ent1, ent2=ent2,
                           lam=lam, lambda_darkfield=lambda_darkfield,
                           darkfield_limit=darkfield_limit,
                           estimate_darkfield=estimate_darkfield,
                           max_iterations=max_iterations,
                           optimization_tol=np.float32(optimization_tol), norm_d=norm_d)

        if estimate_darkfield:
            a_offset += b1_off * f

        # Reweighting convergence: normalised L1 change, matching MATLAB.
        np.multiply(f, np.float32(a1_coeff.mean()), out=ff_curr)
        ff_curr /= max(np.float32(ff_curr.mean()), np.finfo(np.float32).eps)
        mad_ff = (np.abs(ff_curr - flatfield_prev).sum()
                  / max(np.abs(flatfield_prev).sum(), np.finfo(np.float32).eps))
        if estimate_darkfield:
            td = np.abs(a_offset - darkfield_prev).sum()
            mad_df = 0.0 if td < 1e-7 else td / max(np.abs(darkfield_prev).sum(), 1e-6)
        else:
            mad_df = 0.0

        if max(mad_ff, mad_df) <= reweight_tol:
            break

        flatfield_prev[...] = ff_curr
        if estimate_darkfield:
            darkfield_prev[...] = a_offset
        _update_weights(w_coeff, e, f, a1_coeff, a_offset, np.float32(epsilon))

    flatfield = np.maximum(idct2_ortho(w_hat), 0.0)
    flatfield_mean = np.float32(flatfield.mean())
    if flatfield_mean < np.finfo(np.float32).eps:
        flatfield_mean = np.float32(1.0)
    flatfield = flatfield / flatfield_mean

    out_size = (h_orig, w_orig) if output_size is None else tuple(output_size)
    if (h, w) != out_size:
        flatfield = imresize(flatfield, out_size)
        a_offset = imresize(a_offset, out_size)

    # Report the darkfield in the SAME units as `images`, undoing the internal
    # `d = images / global_mean`. The flatfield is a mean-1 multiplier and so scale-free;
    # the darkfield is ADDITIVE, and left normalised it comes back ~O(1) instead of the
    # true dark level, whereupon the apply's `(raw - dark)/flat` subtracts nearly nothing.
    return flatfield.astype(np.float32), (a_offset * global_mean).astype(np.float32)


# ─── Driver ───────────────────────────────────────────────────────────────────

# A fitted darkfield below this fraction of the MEASURED background is treated as
# degenerate. 0.5 is deliberately loose: a real darkfield can legitimately sit well under
# the background it was measured from, since some of that background is scattered light,
# which is scene-dependent and not a detector offset. The case this exists for measured
# 1.6 counts against a 122-count background.
DARKFIELD_DEGENERATE_FRACTION = 0.5


def measured_background_level(cfg, camera):
    """The background level in counts that the `emptiness` stage measured, or None.

    That stage merges `background_level` into every tile's own stats file, so any tile of
    this camera carries it; the first readable one answers.
    """
    for setup in _config.camera_setups(cfg)[camera]:
        path = Path(cfg["results_root"]) / "intensity_stats" / f"setup{setup}.json"
        if not path.is_file():
            continue
        try:
            lvl = json.loads(path.read_text()).get("background_level")
        except (OSError, ValueError):
            continue
        if isinstance(lvl, (int, float)):
            return float(lvl)
    return None


def warn_if_darkfield_collapsed(darkfield, measured, label):
    """Report -- but do NOT repair -- a darkfield that collapsed against the background.

    Only meaningful when BaSiC actually FITTED a darkfield. With `estimate_darkfield`
    off the field is zero by request, and with an override it is whatever was supplied;
    neither is a collapse, and the caller does not call this for either.

    Reports only. Substituting the measured value here would be worse than useless: the
    flat field has already been solved against the collapsed darkfield, and swapping the
    darkfield afterwards leaves a pair of files that each look reasonable alone and are
    wrong together. The fit has to be redone with the darkfield known.
    """
    if measured is None:
        return
    fitted = float(darkfield.mean())
    if fitted >= DARKFIELD_DEGENERATE_FRACTION * measured:
        return
    print(f"warning: BaSiC {label}: darkfield collapsed -- {fitted:.3g} counts against a "
          f"measured background of {measured:.3g}. Either turn OFF estimate_darkfield "
          "(for all-tissue samples with no discernable darkfield) or turn ON "
          "override_darkfield (for datasets with asymmetrical darkfields). "
          "`override_darkfield` also takes a number of counts, if the measured "
          "background is higher than the detector's actual pedestal.")


def resolve_darkfield_override(cfg, camera, params):
    """The darkfield to hold fixed for this camera, or None to fit one.

    `override_darkfield` is False (fit it), True (use the emptiness stage's measured
    background level), or a number of raw counts. True with no measurement available is
    an error rather than a silent fallback -- the whole point of asking for the override
    is not to get a fitted darkfield.
    """
    ovr = params["override_darkfield"]
    if ovr is False:
        return None
    if ovr is True:
        measured = measured_background_level(cfg, camera)
        if measured is None:
            raise RuntimeError(
                f"override_darkfield = true but no background level was measured for "
                f"camera {camera + 1}. Run the emptiness stage first (it merges "
                "background_level into each tile's intensity_stats JSON), or set "
                "override_darkfield to an explicit count.")
        return measured
    if not ovr > 0:
        raise ValueError(
            f"override_darkfield must be true, false, or a positive count; got {ovr}")
    return float(ovr)


def save_basic_field(data, path):
    """Write raw float32 pixels, so the readers round-trip them unchanged."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.ascontiguousarray(data, dtype=np.float32))


def run_basic_camera(cfg, camera, params=None):
    params = _config.basic_params(cfg) if params is None else params
    images = _qstack.load_qstack(cfg, camera)
    override = resolve_darkfield_override(cfg, camera, params)
    lvl = cfg["basic_stats_level"]
    # The fields are published at the LEVEL-0 frame whatever level the qstack was built
    # at, because that is the frame every consumer indexes them in.
    out_size = _qstack.qstack_frame_size(cfg, 0)
    expected = _qstack.qstack_frame_size(cfg, lvl)
    if images.shape[:2] != tuple(expected):
        print(f"warning: qstack frame {images.shape[:2]} does not match "
              f"basic_stats_level={lvl} (expected {tuple(expected)}); it may be stale -- "
              "rerun save_qstack()")
    print(f"BaSiC: estimating camera {camera + 1}, stack {images.shape}, "
          f"darkfield_override={override}, output_size={out_size}")
    flatfield, darkfield = basic_estimate(
        images,
        darkfield_override=override,
        output_size=out_size,
        lam=params["lambda"],
        lambda_darkfield=params["lambda_darkfield"],
        estimate_darkfield=params["estimate_darkfield"],
        max_iterations=params["max_iterations"],
        optimization_tol=params["optimization_tol"],
        reweight_tol=params["reweight_tol"],
        max_reweighting_iterations=params["max_reweighting_iterations"],
        epsilon=params["epsilon"],
        working_size=params["working_size"],
    )
    # Only when BaSiC actually fitted the darkfield: an override supplied it, and
    # `estimate_darkfield = false` asked for none, so a zero field there is the requested
    # answer rather than a collapse. Reports only -- it must not alter what gets written.
    if override is None and params["estimate_darkfield"]:
        warn_if_darkfield_collapsed(darkfield, measured_background_level(cfg, camera),
                                    f"camera {camera + 1}")
    flat, dark = (Path(cfg["results_root"]) / f"camera{camera + 1}" / name
                  for name in ("Flat-field.tif", "Dark-field.tif"))
    save_basic_field(flatfield, flat)
    save_basic_field(darkfield, dark)
    print(f"BaSiC: saved {flat} {dark}")


def run_basic(cfg=None, cameras=None):
    cfg = _config.load_config() if cfg is None else cfg
    params = _config.basic_params(cfg)
    if cameras is None:
        cameras = range(_config.num_cameras(cfg))
    for camera in cameras:
        run_basic_camera(cfg, camera, params)
