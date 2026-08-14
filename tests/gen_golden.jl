# Dump Julia reference values for the Python port's parity tests.
#
#   julia --project=<scratch> tests/gen_golden.jl [path/to/BigFlatFieldIlluminator.jl]
#
# Writes tests/golden/*.json (small arrays inline) and tests/golden/*.bin (larger arrays,
# raw little-endian, with a .json sidecar giving shape/dtype/order).
#
# `src/basic.jl` is `include`d directly rather than loaded through the package, behind a
# handful of stubs for the config types its DRIVER functions mention. That skips
# PythonCall/PyTensorStore and the CondaPkg environment they pull in, none of which the
# numerics need -- the functions under test here (`basic_estimate`, `dct2_ortho`,
# `idct2_ortho`, `shrink!`) touch no I/O at all.
#
# Everything below is deterministic -- no RNG, no clock -- so a rerun must reproduce
# byte-identical files.

using OnlineStats: OrderStats, fit!, nobs, value
using Statistics: quantile, mean, mean!, median
using LinearAlgebra: norm, svdvals
using JSON3: JSON3
using ImageTransformations: imresize
using Colors: Gray
using FixedPointNumbers: N0f16
using FileIO: load, save
using FFTW: r2r, REDFT10, REDFT01
using Printf: @sprintf
using TiffImages: TiffImages

const GOLDEN = joinpath(@__DIR__, "golden")
mkpath(GOLDEN)

# ─── helpers ──────────────────────────────────────────────────────────────────

function write_json(name, obj)
    path = joinpath(GOLDEN, name)
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
    println("wrote $path")
end

# Raw dump + sidecar. Julia is column-major, so the sidecar says order="F" and numpy
# reads it back with reshape(..., order="F") -- no transpose anywhere, which is the
# point: a transposed golden would hide exactly the bug these exist to catch.
function write_bin(name, A::Array{T}) where {T}
    path = joinpath(GOLDEN, name * ".bin")
    open(path, "w") do io
        write(io, A)
    end
    write_json(name * ".json", (shape=collect(size(A)), dtype=string(T), order="F"))
    println("wrote $path")
end

# One accumulator's full observable state, exactly what the Python port must reproduce.
snap(o) = (value=collect(value(o)), n=nobs(o), min=minimum(o), max=maximum(o))

function os_fit(p::Int, ys)
    o = OrderStats(p)
    fit!(o, collect(Float64.(ys)))
    return o
end

# ─── OrderStats cases ─────────────────────────────────────────────────────────

cases = Dict{String,Any}()

# 1. Clean multiple of p: 3 full blocks, exercises the 1/k running mean.
cases["clean_multiple"] = merge(snap(os_fit(4, 1:12)), (p=4, ys=collect(1.0:12.0)))

# 2. Trailing partial block is DISCARDED, but nobs still counts it (14, not 12), which
#    is what every merge is weighted by. `value` must equal case 1's.
cases["trailing_partial"] = merge(snap(os_fit(4, 1:14)), (p=4, ys=collect(1.0:14.0)))

# 3. p larger than the data: no block ever completes, so `value` stays all-zero while
#    the extrema are real. This is the all-zero-qstack trap -- assert the zeros.
cases["p_exceeds_data"] = merge(snap(os_fit(42, 1:30)), (p=42, ys=collect(1.0:30.0)))

# 5. True extrema live in the DISCARDED tail, so minima/maxima cannot be shortcut to
#    value[1]/value[end].
let ys = Float64[5, 6, 7, 8, 1, 99]
    cases["extrema_in_tail"] = merge(snap(os_fit(4, ys)), (p=4, ys=ys))
end

# 6. Merge with equal nobs -> gamma = 1/2, a plain average.
let a = os_fit(4, 1:8), b = os_fit(4, 101:108)
    merge!(a, b)
    cases["merge_equal"] = merge(snap(a), (p=4, ys=collect(1.0:8.0), ys_b=collect(101.0:108.0)))
end

# 7. Unequal nobs -> gamma = 20/28, not exact in binary, so operation order matters.
let a = os_fit(4, 1:8), b = os_fit(4, 101:120)
    merge!(a, b)
    cases["merge_unequal"] = merge(snap(a), (p=4, ys=collect(1.0:8.0), ys_b=collect(101.0:120.0)))
end

# 8. THE case. `a` has 10 observations (2 blocks + 2 discarded), `b` has 8 (2 blocks).
#    gamma is 8/18 -- weighted by OBSERVATIONS -- not 2/4 by blocks.
let a = os_fit(4, 1:10), b = os_fit(4, 101:108)
    merge!(a, b)
    cases["merge_partial_block"] = merge(snap(a), (p=4, ys=collect(1.0:10.0), ys_b=collect(101.0:108.0)))
end

# 9. Merge into a fresh accumulator -> gamma = nb/nb = 1, i.e. b wins outright.
let a = OrderStats(4), b = os_fit(4, 101:108)
    merge!(a, b)
    cases["merge_into_empty"] = merge(snap(a), (p=4, ys=Float64[], ys_b=collect(101.0:108.0)))
end

# 10. Merge is NOT associative in float. The stats pass folds strictly left-to-right over
#     setups[2:end]; both orders are dumped so the Python test can assert they differ.
let ysa = collect(1.0:10.0), ysb = collect(101.0:120.0), ysc = collect(1001.0:1013.0)
    left = merge!(merge!(os_fit(4, ysa), os_fit(4, ysb)), os_fit(4, ysc))
    right = merge!(os_fit(4, ysa), merge!(os_fit(4, ysb), os_fit(4, ysc)))
    cases["fold_left"] = merge(snap(left), (p=4, ys=ysa, ys_b=ysb, ys_c=ysc))
    cases["fold_right"] = merge(snap(right), (p=4, ys=ysa, ys_b=ysb, ys_c=ysc))
end

# 14. Non-integral input, pinning the float64 accumulation rather than an integer path.
let ys = [1.5, 2.25, 3.125, 4.0625, 5.5, 6.75, 7.875, 8.9375]
    cases["non_integral"] = merge(snap(os_fit(4, ys)), (p=4, ys=ys))
end

write_json("orderstats.json", cases)

# ─── 11/12. R-7 quantiles and uint16 rounding ─────────────────────────────────
#
# The 21 levels the stats pass writes, over a length-63 `value` -- the shape a 64-deep
# chunk produces. Non-uniform spacing, so q005 (h = 62*0.05 = 3.1) is a genuine
# interpolation rather than a lattice hit.

    # Sorted, because that is what an OrderStats `value` is; `quantile_r7` on the Python
    # side takes sorted input and does not sort for itself. Still unevenly spaced, which
    # is what makes the interpolation cases meaningful.
let v63 = sort(Float64[100 + 3i + 17 * sin(i / 5) for i in 0:62])
    qs = [quantile(v63, q / 100) for q in 0:5:100]
    write_json("quantiles_r7.json", (
        value=v63,
        levels=collect(0:5:100),
        quantiles=qs,
        rounded=[Int(round(UInt16, q)) for q in qs],   # RoundNearest = ties-to-even
    ))
end

let xs = [0.5, 1.5, 2.5, 2047.5, 2048.5, 65534.5]
    write_json("round_ties.json", (inputs=xs, rounded=[Int(round(UInt16, x)) for x in xs]))
end

# ─── 13. Full tile end-to-end ─────────────────────────────────────────────────
#
# A 5x3 tile -- never square, so a transposed port fails on shape -- of 63-deep uint16
# columns, through the exact reduction the stats pass runs.

let X = 5, Y = 3, Z = 63
    A = Array{UInt16}(undef, X, Y, Z)
    for x in 1:X, y in 1:Y, z in 1:Z
        A[x, y, z] = UInt16((1009 * (x - 1) + 101 * (y - 1) + 7 * (z - 1)) % 4096)
    end
    p = min(Z, div(Z - 1, 21) * 21)
    os = [OrderStats(p) for _ in 1:X, _ in 1:Y]
    for x in 1:X, y in 1:Y
        fit!(os[x, y], Float64.(@view(A[x, y, :])))
    end
    write_bin("tile_input", A)
    write_json("tile_stats.json", (
        p=p,
        shape=[X, Y, Z],
        minima=[Int(UInt16(minimum(o))) for o in os],
        maxima=[Int(UInt16(maximum(o))) for o in os],
        levels=collect(0:5:100),
        # Column-major flattening, matching write_bin's order="F".
        quantiles=[[Int(round(UInt16, quantile(o, q / 100))) for o in os] for q in 0:5:100],
    ))
end

# ─── BaSiC ────────────────────────────────────────────────────────────────────

# Stubs for what src/basic.jl's driver functions mention. None are reached by the
# numerics dumped below; they exist so the file's method signatures resolve.
struct Config end
load_config() = Config()
_read_section() = Dict{String,Any}()
num_cameras(::Config) = 0
camera_setups(::Config) = Vector{Vector{Int}}()
basic_stats_level(::Config) = 0
qstack_frame_size(::Config, ::Int) = (0, 0)

# Defaults to a sibling checkout, so this works from wherever the two repos live
# (the Mac's SMB mount and the cluster's /groups path are different strings).
const JULIA_PKG_ROOT = length(ARGS) >= 1 ? ARGS[1] :
                 normpath(joinpath(@__DIR__, "..", "..", "BigFlatFieldIlluminator.jl"))
include(joinpath(JULIA_PKG_ROOT, "src", "basic.jl"))

# Primitives first, then the norms, then a converged fit -- so a failure says WHICH step
# diverged rather than just "the fields differ".

let H = 24, W = 16
    img = Float32[100 + 40 * sin(i / 4) * cos(j / 3) + 2 * (i + j) for i in 1:H, j in 1:W]
    write_bin("dct_input", img)
    write_bin("dct_forward", dct2_ortho(img))
    write_bin("dct_roundtrip", idct2_ortho(dct2_ortho(img)))
end

# imresize on a linear ramp, so a half-pixel misalignment changes every value instead of
# hiding in a smooth blob. Both directions: BaSiC downsamples to working_size and
# upsamples the fields back out.
let ramp = Float32[i for i in 1:32, j in 1:3]   # constant across columns; 3 wide because
    write_bin("resize_ramp_in", ramp)           # linear interpolation needs >1 sample per axis
    write_bin("resize_ramp_down", Float32.(imresize(ramp, (8, 3))))
    write_bin("resize_ramp_up", Float32.(imresize(ramp, (77, 3))))
end
let img2 = Float32[i * 10 + j for i in 1:24, j in 1:16]
    write_bin("resize_2d_in", img2)
    write_bin("resize_2d_down", Float32.(imresize(img2, (7, 5))))
    write_bin("resize_2d_up", Float32.(imresize(img2, (40, 33))))
end

let x = Float32[-3, -1, -0.5, 0, 0.5, 1, 3]
    y = copy(x); shrink!(y, 1.0f0)
    write_json("shrink_scalar.json", (input=x, t=1.0, output=y))
    z = copy(x); t = Float32[0.1, 0.5, 1, 2, 1, 0.5, 0.1]
    shrink!(z, t)
    write_json("shrink_array.json", (input=x, t=t, output=z))
end

# A full converged fit on a synthetic stack built the way BaSiC assumes: a smooth
# multiplicative flat field, a constant dark pedestal, per-frame scaling, and a sparse
# bright object. No RNG -- the "texture" is a fixed deterministic pattern.
let H2 = 32, W2 = 24, N = 21
    flat_true = Float32[1 + 0.3 * cos(pi * (i - 1) / (H2 - 1)) * cos(pi * (j - 1) / (W2 - 1))
                        for i in 1:H2, j in 1:W2]
    flat_true ./= mean(flat_true)
    dark_true = 120.0f0
    stack = Array{Float32}(undef, H2, W2, N)
    for k in 1:N
        scale = 0.8f0 + 0.4f0 * (k - 1) / (N - 1)
        for i in 1:H2, j in 1:W2
            obj = ((i * 7 + j * 13 + k * 3) % 97 < 4) ? 900.0f0 : 0.0f0
            base = 600.0f0 + 30.0f0 * sin((i + 2j + 3k) / 6)
            stack[i, j, k] = flat_true[i, j] * scale * (base + obj) + dark_true
        end
    end
    write_bin("basic_stack", stack)
    write_bin("basic_flat_true", flat_true)

    D = stack ./ mean(stack)
    sort!(D, dims=3)
    Dflat = reshape(D, H2 * W2, N)
    write_json("basic_norms.json", (
        global_mean=Float64(mean(stack)),
        norm_two=Float64(svdvals(Dflat)[1]),
        norm_D=Float64(norm(Dflat)),
        darkfield_limit=Float64(mean(@view D[:, :, 1])),
    ))
    write_bin("basic_mean_img", Float32.(mean(D, dims=3)[:, :, 1]))

    ff, df = basic_estimate(copy(stack); estimate_darkfield=true, working_size=0)
    write_bin("basic_flat", Float32.(ff))
    write_bin("basic_dark", Float32.(df))

    ff2, df2 = basic_estimate(copy(stack); estimate_darkfield=true, working_size=0,
                              darkfield_override=dark_true)
    write_bin("basic_flat_fixed_dark", Float32.(ff2))
    write_bin("basic_dark_fixed_dark", Float32.(df2))
end

println("golden files written to $GOLDEN")

# ─── 15. Divergence dataset: where Python and Julia CANNOT agree ───────────────
#
# Everything above uses small integers or dyadic rationals, so both implementations get
# exact arithmetic and agree bit for bit. That hides two real differences, which this
# section measures instead of assuming:
#
#   * `_fit!` folds each completed block into a RUNNING mean (`value += (block - value)/k`),
#     accumulating ~1e-13 per value. The Python port sums the blocks instead, which for
#     uint16 input is EXACT (every partial sum is an integer well under 2^53) and so is
#     correctly rounded. Where they differ, Python is the correct one.
#   * `Statistics._quantile` computes its interpolation position with `fma`, which
#     Python cannot reproduce.
#
# A tile with many blocks and non-degenerate values, so both effects are live. The Python
# side re-runs the same reduction on this input and compares the WRITTEN uint16.
let X = 40, Y = 30, NBLK = 15, P = 63
    Z = NBLK * P
    A = Array{UInt16}(undef, X, Y, Z)
    s = UInt64(0x2545F4914F6CDD1D)          # deterministic; no RNG, so reruns match
    for i in eachindex(A)
        s = s * 0x5851F42D4C957F2D + 0x14057B7EF767814F   # UInt64 wraps natively
        A[i] = UInt16((s >> 33) % 4096)
    end
    os = [OrderStats(P) for _ in 1:X, _ in 1:Y]
    for x in 1:X, y in 1:Y
        fit!(os[x, y], Float64.(@view(A[x, y, :])))
    end
    write_bin("divergence_input", A)
    write_json("divergence_stats.json", (
        p=P, nblk=NBLK, shape=[X, Y, Z],
        minima=[Int(UInt16(minimum(o))) for o in os],
        maxima=[Int(UInt16(maximum(o))) for o in os],
        levels=collect(0:5:100),
        quantiles=[[Int(round(UInt16, quantile(o, q / 100))) for o in os] for q in 0:5:100],
    ))
    # The float64 `value` for the first few pixels, so the Python side can show WHICH
    # implementation is closer to the exact mean rather than just that they differ.
    write_json("divergence_value.json", (
        value=[collect(value(os[x, y])) for x in 1:3, y in 1:2][:],
    ))
end
