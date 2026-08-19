"""Store access for the BaSiC side of the pipeline.

Thin adapters over `intensity`'s format helpers, pointed at `input_basic_path` /
`output_basic_path` via `config.basic_view`. There is deliberately no second format
abstraction here: `formats._SPEC` already covers n5, zarr2, zarr3 (sharded and
unsharded), and the two non-standard zarr3 axis orders, and it resolves the on-disk
layout from OME-NGFF metadata rather than assuming a directory convention.
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorstore as ts

from . import config as _config
from .formats import (
    _AXES, _AXES_ZYX, _in_order, _input_location, _ngff_datasets, _output_path, _SPEC,
    canonical_shape, canonical_view,
)

__all__ = [
    "context", "open_source", "source_size_xyz", "camera_source_size_xyz",
    "open_stats_array", "stats_array_path", "open_target", "xy_chunks",
    "ensure_group_json", "slots",
]


def context(cfg=None):
    """One shared `ts.Context` for every array this process opens.

    It must be a Context OBJECT passed as the `context=` argument, not a context dict
    embedded in each spec. A dict is a context *spec*: every `ts.open` builds its own
    Context from it, and therefore its own cache pool. The stats pass opens one array per
    setup plus 23 statistic arrays, so with an embedded dict a `cache_pool` of 512 MiB
    becomes 512 MiB EACH -- measured 11.4 GiB peak RSS on an 18-setup camera, against
    well under 1 GiB once shared.

    Sharing also makes the pool do its job: on a sharded store every X/Y chunk in a job
    lands in the same shard, so one pool across all of them keeps the shard index and
    neighbouring inner chunks resident instead of re-reading them per chunk.

    One definition for the whole package -- memoised here, so this
    and every stage in that module hand `ts.open` the same object. Size the pool with
    SPOTLIGHT_CACHE_BYTES.
    """
    return _context()


# Memory an LSF slot carries, in bytes. Janelia hands out cores and memory in a fixed
# ratio -- a `-n 20` job reported "Total Requested Memory: 307200.00 MB", i.e. 15 GiB per
# slot -- which is what makes the core count predict anything at all here.
GB_PER_SLOT = float(os.getenv("SPOTLIGHT_GB_PER_SLOT", "15"))

# Fraction of the allocation the in-flight data may occupy. The rest is the interpreter,
# numpy's own temporaries, the tensorstore cache pool and its decode arenas. LSF KILLS a
# job that exceeds its reservation, so this errs low: the cost of being conservative is
# some wasted parallelism, the cost of being wrong is a dead array element.
MEMORY_FRACTION = float(os.getenv("SPOTLIGHT_MEMORY_FRACTION", "0.5"))


def memory_budget():
    """Bytes this job may hold in flight, derived from what LSF actually gave it.

    Preference order: an explicit override, the cgroup limit LSF enforces, then
    cores x GB_PER_SLOT. Falls back to 4 GiB off-cluster.

    Note what this is NOT: a core count. Concurrency here is bounded by MEMORY, because
    the reads are latency-bound rather than CPU-bound -- sizing in-flight reads to the
    core count starves the I/O and, when the limit falls below the number of units,
    stalls on head-of-line blocking as well. Measured on an 18-setup camera at 8 cores:
    concurrency 8 took 61 s, concurrency 18 took 26 s, at 1.4 GiB peak either way. The
    thread pool that runs the numpy IS sized by cores -- that part is CPU-bound.
    """
    env = os.getenv("SPOTLIGHT_MEMORY_BYTES")
    if env:
        return int(env)
    cg = os.getenv("LSB_CG_MEMLIMIT")          # LSF cgroup limit, bytes, when enforced
    if cg:
        try:
            return int(int(cg) * MEMORY_FRACTION)
        except ValueError:
            pass
    cores = os.getenv("LSB_DJOB_NUMPROC")
    if cores:
        return int(float(cores) * GB_PER_SLOT * 2**30 * MEMORY_FRACTION)
    return int(_machine_memory() * MEMORY_FRACTION)


def _machine_memory():
    """Total RAM of this machine, for runs outside LSF.

    Off-cluster there is no allocation to derive from, so use what the box has. A flat
    default would be wrong in both directions -- far too small on a 128 GiB workstation,
    far too large on a laptop -- and getting it too large is the one that hurts, since
    the process is then competing with everything else the user is running.

    `os.sysconf` covers Linux and macOS with no dependency; anything else falls back.
    """
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 8 * 2**30


def _open(spec, ctx, **kwargs):
    """`ts.open`, with the context passed as an object rather than embedded in the spec."""
    if ctx is None:
        return ts.open(spec, **kwargs).result()
    return ts.open(spec, context=ctx, **kwargs).result()


def open_source(cfg, setup, scale=0, ctx=None):
    """A canonical (Z, Y, X) view of one setup's input array at `scale`."""
    bcfg = _config.basic_view(cfg)
    path, order = _input_location(bcfg, setup, scale)
    spec = {"driver": _SPEC[bcfg["input_format"]]["driver"],
            "kvstore": {"driver": "file", "path": path}}
    return canonical_view(_open(spec, ctx), order)


def source_size_xyz(cfg, setup=None, scale=0):
    """(X, Y, Z) of one setup at `scale`, matching Julia's `get_source_size`.

    Reads the array metadata rather than opening the array: the stats pass calls this
    once per setup, and on a 1600-tile mosaic opening each one just to learn its shape is
    1600 array opens for three integers.
    """
    bcfg = _config.basic_view(cfg)
    if setup is None:
        setup = _config.camera_setups(cfg)[0][0]
    path, order = _input_location(bcfg, setup, scale)
    meta_name = _SPEC[bcfg["input_format"]]["meta"]
    with open(f"{path}/{meta_name}") as f:
        meta = json.load(f)
    shape = meta["dimensions"] if "dimensions" in meta else meta["shape"]
    z, y, x = canonical_shape(shape, order)
    return [x, y, z]


def camera_source_size_xyz(cfg, setups, scale=0):
    """Element-wise minimum (X, Y, Z) over every setup a camera reads.

    Tiles in one dataset are not guaranteed to share a shape -- a later acquisition block
    can be shallower in Z than setup 0 -- so anything that slices ALL of a camera's
    setups with the same bounds has to be sized off the smallest, not off the reference
    setup. Reading past a shorter tile's end makes tensorstore throw OUT_OF_RANGE.
    """
    sizes = [source_size_xyz(cfg, setup=s, scale=scale) for s in setups]
    smallest = [min(v) for v in zip(*sizes)]
    largest = [max(v) for v in zip(*sizes)]
    if smallest != largest:
        print(f"warning: setups in this camera differ in size (min {smallest}, "
              f"max {largest}); using the smallest")
    return smallest


def xy_chunks(size_xy, tile_xy):
    """The X/Y tiling of a frame, as (x_slice, y_slice) pairs.

    A flat list in the same order Julia's `get_pychunks` produces (column-major over the
    chunk grid, X fastest), because the LSF array's chunk indices address it positionally
    -- job N owns `chunks[start:stop]`, and a different ordering silently gives job N
    somebody else's part of the frame.
    """
    nx = -(-size_xy[0] // tile_xy[0])
    ny = -(-size_xy[1] // tile_xy[1])
    out = []
    for iy in range(ny):
        for ix in range(nx):
            out.append((slice(ix * tile_xy[0], min((ix + 1) * tile_xy[0], size_xy[0])),
                        slice(iy * tile_xy[1], min((iy + 1) * tile_xy[1], size_xy[1]))))
    return out


def stats_array_path(cfg, camera, stat, scale=0):
    """Where one per-camera statistic array lives, for callers that need the path itself.

    One definition, so a caller inspecting the array on disk cannot drift from the one
    `open_stats_array` hands tensorstore.
    """
    return Path(cfg["results_root"]) / f"camera{camera + 1}" / stat / f"s{scale}"


def open_stats_array(cfg, camera, stat, xy_size, scale=0, ctx=None, rebuild=False):
    """Create/open one per-camera statistic array: `{results_root}/camera{N}/{stat}/s{scale}`.

    Always n5 and always (X, Y), whatever the input format is -- these are this package's
    own intermediates, and `save_qstack` transposes on read for the zarr formats. Camera
    is 1-based on disk, matching what the Julia package wrote.

    `rebuild` throws the array away and makes a fresh one, for metadata that cannot be
    opened at all (see `create_quartile_histograms`). tensorstore rejects `delete_existing`
    together with `open`, which is why it is a separate mode rather than an extra flag --
    and why it must stay opt-in: it discards every chunk already written.
    """
    spec = {
        "driver": "n5",
        "kvstore": {"driver": "file",
                    "path": str(stats_array_path(cfg, camera, stat, scale))},
        "metadata": {
            "dimensions": list(xy_size),
            "blockSize": list(cfg["chunk_size"][:2]),
            "dataType": "uint16",
            "compression": {"type": "zstd", "level": 3},
        },
    }
    mode = ({"create": True, "delete_existing": True} if rebuild
            else {"create": True, "open": True})
    return _open(spec, ctx, **mode)


def open_target(cfg, setup, size_xyz, ctx=None):
    """Create/open the corrected output array for one setup, as a canonical (Z, Y, X) view.

    Mirrors Julia's four `load_target` methods, but routed through
    `_output_metadata` so the two pipelines cannot drift on codec or chunk
    layout. `size_xyz` is this setup's own size -- tiles differ in shape, and a target
    sized off the reference setup leaves a shorter tile's tail never written.
    """
    bcfg = _config.basic_view(cfg)
    fmt = bcfg["output_format"]
    order = _SPEC[fmt]["order"]
    x, y, z = size_xyz
    shape = _in_order((z, y, x), order)
    chunk = _in_order(cfg["chunk_size"][::-1], order)
    shard = _in_order(cfg["shard_size"][::-1], order)
    driver, meta = _output_metadata(fmt, list(shape), chunk, shard, "uint16")
    spec = {"driver": driver,
            "kvstore": {"driver": "file",
                        "path": _output_path(fmt, bcfg["output_intensity_path"],
                                                       setup, 0)},
            "metadata": meta}
    arr = _open(spec, ctx, create=True, open=True)
    return canonical_view(arr, order)


# ─── moved from py ──────────────────────────────────────────────────
#
# The tensorstore context, and everything that creates or describes an output array:
# NGFF/n5 group metadata, the per-format output spec, and the pyramid shape/factor
# discovery both the writers and the readers size themselves from.


# Thread pool sizes, chosen by measurement rather than arithmetic. Scicomp's contract is
# that a host's 1-minute load average stays under 2x the slots LSF reserved, and load counts
# threads that are RUNNABLE *or* in uninterruptible sleep -- so a thread parked in a network
# filesystem call is charged exactly like one burning a core. There is no free I/O
# concurrency, which is what makes this a budget rather than a preference.
#
# `bench/sweep_threads.py correct 0` at 64 slots -- peak threads runnable-or-blocked (which
# IS the load) against wall clock:
#
#     file_io   data_copy   peak busy   / slots   wall
#     slots*64     slots         1157     18.1x   44.1 s   the behaviour that drew the email
#     slots*16   slots/2         1045     16.3x   53.7 s
#     slots*4    slots/2          295      4.6x   50.4 s
#     slots/2    slots/2           71      1.1x   49.4 s
#     slots      slots/2          102      1.6x   48.6 s
#     slots/2      slots          104      1.6x   44.6 s   <-- what this uses
#     slots        slots          138      2.2x   44.3 s   over the contract
#
# The two pools are not interchangeable, and only measuring separated them:
#
# * `file_io_concurrency` buys NO throughput at all. 32 -> 64 -> 256 -> 1024 went 49.4 ->
#   48.6 -> 50.4 -> 53.7 s, flat then worse, while costing load the whole way. The `slots*64`
#   multiplier this replaced was not paying for itself even before anyone mentioned load: at
#   30 slots it was 1920 threads, and one correction element peaked at 2165 -- 72x its
#   allocation. Half the slots is enough to keep an unsharded n5 source's opens overlapping.
# * `data_copy_concurrency` is where the time was. It is zstd decode and encode -- real CPU
#   work on cores the job reserved -- and halving it cost 11% wall clock (49.4 vs 44.6 s) for
#   33 threads of load. Full slots recovers all of it and still lands at 1.6x, because a
#   read, its decode and the kernel's pass over the result do not peak together: the
#   worst-case sum is 2.5x slots and the measured peak is 1.6x.
#
# The honest consequence: under a load-vs-slots contract, concurrency is bought with SLOTS.
# A stage that wants more reads in flight raises its `bsub -n` (`n_cores_stats` and friends),
# not a multiplier. The stats stage drops from 192 in-flight opens to 2 at 3 slots, and it is
# the stage the old multiplier was tuned for, holding a twentieth of the slots -- so it will
# cost more than the numbers above. That one is still UNMEASURED:
# `bench/sweep_threads.py stats <camera> <start> <stop>`. `SPOTLIGHT_IO_CONCURRENCY` and
# `SPOTLIGHT_COPY_CONCURRENCY` override either pool for a one-off.


def slots(default=8):
    """Cores LSF reserved for this job: what every thread pool here is sized against.

    One definition, because the pools only add up to something meaningful if they are all
    measured against the same number.

    `default` applies only OFF the cluster and should come from `config.stage_cores` -- the
    `bsub -n` the stage would have been submitted with. It is CLAMPED to the machine's core
    count there, and only there: off the cluster there is no reservation to honour, so the
    box is the constraint, and a 20-core default would otherwise oversubscribe a laptop
    fourfold. LSF's own number is never clamped -- a 30-slot job on a 128-core host means 30.
    """
    reserved = os.getenv("LSB_DJOB_NUMPROC")
    if reserved:
        return int(reserved)
    return max(1, min(int(default), os.cpu_count() or int(default)))


def _context_spec():
    # Not `config.stage_cores`: `_context()` is memoised process-wide and called from every
    # stage without a cfg in hand, so there is no stage to attribute it to. Off-cluster the
    # machine is the only honest bound anyway, which is what `slots` clamps to.
    n = slots(os.cpu_count() or 8)
    # Floored at 2 so a 1-slot job still overlaps one read with one decode rather than
    # serialising against itself. See the table above for why these two differ.
    io = int(os.getenv("SPOTLIGHT_IO_CONCURRENCY", str(max(2, n // 2))))
    copy = int(os.getenv("SPOTLIGHT_COPY_CONCURRENCY", str(max(2, n))))
    return {
        "data_copy_concurrency": {"limit": copy},
        "file_io_concurrency": {"limit": io},
        "cache_pool": {"total_bytes_limit":
                       int(os.getenv("SPOTLIGHT_CACHE_BYTES", 512 * 2**20))},
    }


_SHARED_CONTEXT = None


def _context():
    """The ONE `ts.Context` this process uses, built once.

    It must be a Context object shared by every `ts.open`, not a spec dict embedded in
    each one. A dict is a context SPEC: tensorstore builds a fresh Context from it per
    open, and therefore a fresh `cache_pool` and a fresh set of concurrency limits. The
    limits are the subtler half -- `file_io_concurrency` is meant to CAP in-flight opens,
    and a per-array copy of it caps nothing.

    Measured: 18 arrays opened with a spec dict held 0.68 GiB against 0.11 GiB shared, and
    the stats pass (18 setups + 23 statistic arrays) peaked at 11.4 GiB before this.
    `_read_tile` is the other place it bit -- it is `pool.map`ped over every setup, so a
    1600-tile mosaic built 1600 pools.
    """
    global _SHARED_CONTEXT
    if _SHARED_CONTEXT is None:
        _SHARED_CONTEXT = ts.Context(_context_spec())
    return _SHARED_CONTEXT


def _atomic_write_json(path, obj):
    """Write JSON atomically (temp + os.replace) so concurrent array jobs writing
    the shared top-level group file can't corrupt it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, p)


def write_group_metadata(cfg, setup, factors):
    """Write multiscale group metadata for the output dataset/setup.

    zarr3 / zarr3_unsharded / zarr3_zyx -> OME-NGFF 0.5 (multiscales under
    attributes.ome in the group zarr.json); zarr2 -> OME-NGFF 0.4 (multiscales in
    .zattrs); n5 -> n5 `downsamplingFactors` group attributes. `factors` is the
    per-level list of cumulative (fz, fy, fx) downsample factors (level 0 ==
    (1, 1, 1)). zarr3_zyx writes 3-D (z,y,x) axes/scale/translation arrays - the
    standard NGFF spatial order - instead of the 5-D (t,c,z,y,x) the other zarr
    formats use, so downstream readers that assume that standard order (rather
    than reading axes by name) work without any changes.
    """
    fmt = cfg["output_format"]
    order = _SPEC[fmt]["order"]
    axes = _AXES_ZYX if order == "zyx" else _AXES
    datasets = _ngff_datasets(factors, order)

    if fmt in ("zarr3", "zarr3_unsharded", "zarr3_zyx"):
        ds_dir = cfg["output_intensity_path"]
        setup_dir = f"{ds_dir}/s{setup}-t0.zarr"
        ome = {
            "version": "0.5",
            "multiscales": [{
                "name": "/",
                "axes": axes,
                "datasets": datasets,
                "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(axes)}],
            }],
        }
        _atomic_write_json(f"{ds_dir}/zarr.json", {"zarr_format": 3, "node_type": "group"})
        _atomic_write_json(f"{setup_dir}/zarr.json",
                           {"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}})
    elif fmt == "zarr2":
        ds_dir = cfg["output_intensity_path"]
        setup_dir = f"{ds_dir}/s{setup}-t0.zarr"
        multiscales = [{
            "version": "0.4",
            "name": "/",
            "axes": axes,
            "datasets": datasets,
            "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(axes)}],
        }]
        _atomic_write_json(f"{ds_dir}/.zgroup", {"zarr_format": 2})
        _atomic_write_json(f"{setup_dir}/.zgroup", {"zarr_format": 2})
        _atomic_write_json(f"{setup_dir}/.zattrs", {"multiscales": multiscales})
    else:  # n5: group-level downsamplingFactors (x, y, z), the n5 multiscale convention
        ds_dir = cfg["output_intensity_path"]
        grp = f"{ds_dir}/setup{setup}/timepoint0"
        xyz = [[float(fx), float(fy), float(fz)] for (fz, fy, fx) in factors]
        _atomic_write_json(f"{ds_dir}/attributes.json", {"n5": "2.0.0"})
        _atomic_write_json(f"{grp}/attributes.json",
                           {"downsamplingFactors": xyz, "scales": xyz})


def _output_metadata(fmt, shape, chunk, shard, dtype_name):
    """(driver, metadata) for one output array in the given format."""
    if fmt in ("zarr3", "zarr3_zyx"):   # same sharded array layout, different axis order
        return "zarr3", {
            "data_type": dtype_name,
            "shape": shape,
            "fill_value": 0,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shard}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": chunk,
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "zstd", "configuration": {"level": 3}},
                    ],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }],
        }
    if fmt == "zarr3_unsharded":
        return "zarr3", {
            "data_type": dtype_name,
            "shape": shape,
            "fill_value": 0,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 3}},
            ],
        }
    if fmt == "zarr2":
        return "zarr2", {
            "dtype": ">u2",
            "shape": shape,
            "chunks": chunk,
            "dimension_separator": "/",
            "compressor": {"id": "zstd", "level": 3},
        }
    return "n5", {                       # n5
        "dimensions": shape,
        "blockSize": chunk,
        "dataType": dtype_name,
        "compression": {"type": "gzip"},
    }


def ensure_group_json(cfg, setup):
    """Make the output a valid group AS SOON AS the first array is created.

    `write_group_metadata` runs at the END of a setup's correction, once the pyramid
    factors are known. Until then the arrays exist under a directory with no group node,
    so anything that opens the store mid-run -- a viewer, a sanity check, a downstream
    job started early -- sees an invalid zarr. On a long array run that window is hours.

    This writes only the part that needs nothing: the bare group node. The multiscale
    `attributes` still arrive at the end. Written if ABSENT, so it never clobbers the
    fuller version, and the content is identical across concurrent array elements, so
    two of them racing write the same bytes.
    """
    fmt = cfg["output_format"]
    root = cfg["output_intensity_path"]
    if fmt in ("zarr3", "zarr3_unsharded", "zarr3_zyx"):
        node = {"zarr_format": 3, "node_type": "group"}
        for path in (f"{root}/zarr.json", f"{root}/s{setup}-t0.zarr/zarr.json"):
            if not Path(path).exists():
                _atomic_write_json(path, node)
    elif fmt == "zarr2":
        for path in (f"{root}/.zgroup", f"{root}/s{setup}-t0.zarr/.zgroup"):
            if not Path(path).exists():
                _atomic_write_json(path, {"zarr_format": 2})
    elif fmt == "n5":
        if not Path(f"{root}/attributes.json").exists():
            _atomic_write_json(f"{root}/attributes.json", {"n5": "2.0.0"})


def open_output_array(cfg, setup, level, shape, dtype_name, context):
    """Create one output level array (shape given in the output's stored order)."""
    fmt = cfg["output_format"]
    order = _SPEC[fmt]["order"]
    path = _output_path(fmt, cfg["output_intensity_path"], setup, level)
    chunk = _in_order(cfg["chunk_size"][::-1], order)   # inner chunk
    shard = _in_order(cfg["shard_size"][::-1], order)   # outer chunk / file
    driver, meta = _output_metadata(fmt, list(shape), chunk, shard, dtype_name)
    arr = ts.open({
        "driver": driver,
        "kvstore": {"driver": "file", "path": path},
        "metadata": meta,
    }, context=context, create=True, open=True).result()
    ensure_group_json(cfg, setup)
    return arr, shard, path


_SHAPE_CACHE = {}


def source_pyramid_shapes(cfg, setup):
    """Canonical (Z, Y, X) shape of every input pyramid level present on disk.

    Cached per (store, format, setup): walking the pyramid means one array open
    per level, and both `source_pyramid_factors` and `basic_model` need it (the
    latter once per setup per level, which would otherwise re-open the whole
    pyramid for every shard).
    """
    key = (cfg["input_intensity_path"], cfg["input_format"], setup)
    hit = _SHAPE_CACHE.get(key)
    if hit is not None:
        return hit

    spec = _SPEC[cfg["input_format"]]
    context = _context()
    shapes = []
    level = 0
    while True:
        path, order = _input_location(cfg, setup, level)
        if not (Path(path) / spec["meta"]).exists():
            break
        arr = ts.open({
            "driver": spec["driver"],
            "kvstore": {"driver": "file", "path": path},
        }, context=context, open=True, read=True).result()
        shapes.append(canonical_shape(arr.domain.shape, order))   # (Z, Y, X)
        level += 1

    _SHAPE_CACHE[key] = shapes
    return shapes


def source_pyramid_factors(cfg, setup):
    """Cumulative (fz, fy, fx) downsample factors for each input pyramid level.

    Derived generically from the on-disk level shapes (factor = round(shape0 /
    shapeL) per axis), so it works for n5 / zarr2 / zarr3 alike. Level 0 is
    always (1, 1, 1). Falls back to a single level if no pyramid is present.
    """
    shapes = source_pyramid_shapes(cfg, setup)
    if not shapes:
        return [(1, 1, 1)]
    z0, y0, x0 = shapes[0]
    return [(max(round(z0 / z), 1), max(round(y0 / y), 1), max(round(x0 / x), 1))
            for (z, y, x) in shapes]
