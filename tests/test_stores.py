"""End-to-end I/O over synthetic stores, in all four formats.

These do not need Julia: the stores are built here, and the expected values are computed
from the same definition the writer used. What they check is that the port reads and
writes the right voxels in the right order -- the part that pure-math parity cannot see.
"""

import numpy as np
import pytest

from make_store import DEPTHS, FORMATS, X, Y, volume, write_store

from spotlight.stores import _context
from spotlight import config, correct, quantiles, stores
from spotlight.orderstats import LEVELS, OrderStats, block_size, quantile_r7, to_uint16


@pytest.fixture(params=FORMATS)
def store(request, tmp_path):
    return write_store(tmp_path / "in", request.param)


def test_source_size_is_xyz(store):
    """`source_size_xyz` reports (X, Y, Z) whatever the on-disk axis order is."""
    for setup, z in DEPTHS.items():
        assert stores.source_size_xyz(store, setup=setup) == [X, Y, z]


def test_read_returns_canonical_zyx_with_the_right_voxels(store):
    src = stores.open_source(store, 1)
    got = np.asarray(src[0:5, 0:Y, 0:X].read().result())
    assert got.shape == (5, Y, X)
    np.testing.assert_array_equal(got, volume(1, 63)[:5])


def test_camera_source_size_takes_the_smallest(store, capsys):
    """Slicing every setup with the same bounds has to be sized off the shallowest, or
    tensorstore throws OUT_OF_RANGE on the short one."""
    assert stores.camera_source_size_xyz(store, [0, 1, 2, 3]) == [X, Y, 63]
    assert "differ in size" in capsys.readouterr().out


def test_xy_chunks_tile_the_frame_exactly():
    chunks = stores.xy_chunks((X, Y), (32, 32))
    assert len(chunks) == 2 * 2                       # 64/32 x ceil(48/32)
    covered = np.zeros((X, Y), dtype=int)
    for xs, ys in chunks:
        covered[xs, ys] += 1
    assert (covered == 1).all()                       # a partition, no gaps or overlaps
    assert chunks[-1][1] == slice(32, 48)             # the ragged Y edge is not padded


def test_stats_pass_writes_what_the_reduction_says(store, tmp_path):
    """The whole pass over one chunk, against the reduction computed directly here."""
    setups = [0, 1, 2]
    store = {**store, "last_setup": 2, "setups_per_camera": 3}
    quantiles.calculate_camera_stats(store, 0, 1, 1)

    xs, ys = stores.xy_chunks((X, Y), (32, 32))[0]
    # Derived the way the pass derives it, because the block depth is format-dependent:
    # sharded zarr3 sizes it off shard_size[2] * z_batch, everything else off chunk_size[2].
    z_blocks = quantiles._z_blocks(store, 63)
    p = block_size(quantiles._z_depth(store, z_blocks[0][1] - z_blocks[0][0]))
    expected = None
    for s in setups:
        a = np.ascontiguousarray(volume(s, 63)[:, ys, xs].T)      # (X, Y, Z)
        st = OrderStats.fit(a, p)
        expected = st if expected is None else expected.merge(st)

    ctx = stores.context()
    for name, want in (("minima", to_uint16(expected.vmin)),
                       ("maxima", to_uint16(expected.vmax))):
        arr = stores.open_stats_array(store, 0, name, (X, Y), ctx=ctx)
        np.testing.assert_array_equal(np.asarray(arr[xs, ys].read().result()), want)

    v = np.sort(expected.value, axis=-1)
    for q in LEVELS:
        arr = stores.open_stats_array(store, 0, f"q{q:03d}", (X, Y), ctx=ctx)
        got = np.asarray(arr[xs, ys].read().result())
        np.testing.assert_array_equal(got, to_uint16(quantile_r7(v, q / 100.0)),
                                      err_msg=f"q{q:03d}")


def test_stats_pass_refuses_a_camera_too_shallow_in_z(store, tmp_path):
    """Fewer than 21 slices per setup means there is no distribution to summarise; the
    raw-stack path handles those, and this one must say so rather than write zeros."""
    shallow = write_store(tmp_path / "shallow", store["input_format"],
                          setups=(0,), depths={0: 8})
    with pytest.raises(RuntimeError, match="N_QUARTILES"):
        quantiles.calculate_camera_stats({**shallow, "n_cores_stats": 2}, 0, 1, 1)


def test_stats_pass_refuses_a_block_depth_that_yields_no_blocks(store, tmp_path):
    """A chunk depth of exactly 21 gives a block size of 0 -- every quantile would read
    0. Julia dies with a DivideError from inside OnlineStats; say what is wrong instead."""
    from spotlight.orderstats import block_size
    with pytest.raises(ValueError, match="block size of 0"):
        block_size(21)


def test_correction_round_trips_through_every_output_format(store, tmp_path):
    """Write fields, correct a setup, read it back, compare against the math done here.

    Within one gray level, not exact. The shared kernel folds `(raw - dark)/flat` into the
    affine `raw*(1/flat) + (-dark/flat)` so the flat/dark and intensity corrections compose
    into a single multiply-add per voxel. That differs from the unfolded order by an ULP,
    which lands on the far side of a rounding boundary for a small fraction of voxels --
    measured 0.011% here, always by exactly 1. That is far less error than the
    intermediate uint16 rounding a two-pass correction would introduce, which is the
    trade this buys.
    """
    from spotlight import basic

    setup = 0
    z = DEPTHS[setup]
    results = tmp_path / "results"
    flat = (1.0 + 0.2 * np.cos(np.linspace(0, np.pi, Y))[:, None]
            * np.cos(np.linspace(0, np.pi, X))[None, :]).astype(np.float32)
    dark = np.full((Y, X), 37.0, np.float32)
    basic.save_basic_field(flat, results / "camera1" / "Flat-field.tif")
    basic.save_basic_field(dark, results / "camera1" / "Dark-field.tif")

    cfg = {**store, "results_root": str(results), "last_setup": 3,
           "setups_per_camera": 4, "apply_basic": True}
    correct.apply_correction_chunked(cfg, setup)

    out = stores.open_source({**cfg, "input_basic_path": cfg["output_basic_path"]},
                             setup, 0)
    got = np.asarray(out[:, :, :].read().result())
    raw = volume(setup, z).astype(np.float32)
    from spotlight.correct import FLAT_FLOOR
    want = np.round(np.clip((raw - dark) / np.maximum(flat, FLAT_FLOOR),
                            0, 65535)).astype(np.uint16)
    diff = np.abs(got.astype(np.int32) - want.astype(np.int32))
    assert diff.max() <= 1, f"max difference {diff.max()} counts"
    assert (diff != 0).mean() < 0.001, f"{(diff != 0).mean():.4%} of voxels differ"


def test_qstack_orientation(store, tmp_path):
    """n5 keeps the stats arrays' (X, Y); everything else is transposed to (Y, X). The
    fields are read back with the same rule, so the two cannot disagree."""
    from spotlight import qstack

    expect_yx = store["input_format"] != "n5"
    assert qstack.in_plane_order(store) == ("yx" if expect_yx else "xy")
    assert qstack.qstack_frame_size(store, 0) == ((Y, X) if expect_yx else (X, Y))


def test_context_is_a_shared_object_not_a_spec_dict():
    """A context DICT in a spec makes `ts.open` build its own Context, and so its own
    cache pool, per array. The stats pass opens one array per setup plus 23 statistic
    arrays, so that turned a 512 MiB pool into 11.4 GiB of peak RSS on an 18-setup
    camera. Sharing one Context object is the fix; this pins it.
    """
    import tensorstore as ts
    assert isinstance(stores.context(), ts.Context)


def test_no_module_embeds_a_context_in_a_spec():
    """Same bug, caught wherever it reappears -- every module, no exemptions.

    `intensity.py` was exempt while it was a verbatim copy of the original script. It
    stopped deserving that (`_read_tile` embedded a fresh spec per call and is
    `pool.map`ped over every setup, so the emptiness stage built one cache pool PER TILE),
    and it no longer exists -- its contents are the eight modules this globs.
    """
    from pathlib import Path
    pkg = Path(stores.__file__).parent
    offenders = [p.name for p in pkg.glob("*.py") if '"context":' in p.read_text()]
    assert not offenders, f"{offenders} embed a context dict in a tensorstore spec"


def test_context_is_memoised_across_the_package():
    """Every stage must hand `ts.open` the SAME object, or the limits cap nothing and the
    pools multiply."""
    assert stores.context() is stores.context()
    assert stores.context() is _context()


# ─── deriving the knobs from the allocation ───────────────────────────────────


def test_memory_budget_prefers_the_cgroup_limit(monkeypatch):
    """`LSB_CG_MEMLIMIT` is what LSF actually enforces, so it beats any inference."""
    monkeypatch.delenv("SPOTLIGHT_MEMORY_BYTES", raising=False)
    monkeypatch.setenv("LSB_CG_MEMLIMIT", str(100 * 2**30))
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "2")        # would infer far less; ignored
    assert stores.memory_budget() == int(100 * 2**30 * stores.MEMORY_FRACTION)


def test_memory_budget_scales_with_cores(monkeypatch):
    """Cores predict MEMORY, not concurrency: LSF allocates them in a fixed ratio."""
    monkeypatch.delenv("SPOTLIGHT_MEMORY_BYTES", raising=False)
    monkeypatch.delenv("LSB_CG_MEMLIMIT", raising=False)
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "8")
    eight = stores.memory_budget()
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "16")
    assert stores.memory_budget() == 2 * eight
    assert eight == int(8 * stores.GB_PER_SLOT * 2**30 * stores.MEMORY_FRACTION)


def test_memory_budget_off_cluster_reads_the_machine(monkeypatch):
    """No LSF allocation to derive from, so use what the box actually has -- see
    test_sweep.py for the fallback when even that is unavailable."""
    for k in ("SPOTLIGHT_MEMORY_BYTES", "LSB_CG_MEMLIMIT", "LSB_DJOB_NUMPROC"):
        monkeypatch.delenv(k, raising=False)
    assert stores.memory_budget() == int(stores._machine_memory() * stores.MEMORY_FRACTION)


def test_budget_stays_under_the_allocation(monkeypatch):
    """The whole point: exceeding the reservation gets the array element KILLED, so the
    budget must be a fraction of what was granted, never all of it."""
    monkeypatch.delenv("SPOTLIGHT_MEMORY_BYTES", raising=False)
    monkeypatch.delenv("LSB_CG_MEMLIMIT", raising=False)
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "20")
    granted = 20 * stores.GB_PER_SLOT * 2**30
    assert stores.memory_budget() < granted
    assert stores.MEMORY_FRACTION <= 0.75, "too little headroom for numpy + tensorstore"


def test_stats_concurrency_rises_with_the_allocation(monkeypatch):
    """More cores -> more memory -> more setups in flight, capped at the setup count."""
    from spotlight import quantiles
    monkeypatch.delenv("SPOTLIGHT_STATS_CONCURRENCY", raising=False)
    monkeypatch.delenv("SPOTLIGHT_STATS_MEMORY_BYTES", raising=False)
    monkeypatch.delenv("LSB_CG_MEMLIMIT", raising=False)
    # 512 MiB per unit, so memory is what binds. At the ~7 MiB a 64x64 tile actually
    # costs, even a 1-core slot affords ~1000 -- in practice the setup count is the
    # binding limit and memory only matters for large tiles or deep Z blocks.
    per_setup = 512 * 2**20
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "1")
    small = quantiles._concurrency({}, 100, per_setup)
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "8")
    big = quantiles._concurrency({}, 100, per_setup)
    assert small < big, f"{small} !< {big}: more cores must buy more in flight"
    # ...but never past the number of units, since idle slots buy nothing.
    assert quantiles._concurrency({}, 4, 7 * 2**20) == 4


def test_the_two_pools_are_sized_from_the_reservation_and_differ(monkeypatch):
    """Pinned because the pair is a MEASUREMENT, and the two halves are not interchangeable.

    Sized against slots, since load is charged per reserved slot and a thread parked in an
    NFS call counts like one burning a core. But `bench/sweep_threads.py correct 0` at 64
    slots separated them: file_io went 32 -> 64 -> 256 -> 1024 for 49.4 -> 48.6 -> 50.4 ->
    53.7 s (flat then worse) while data_copy at half cost 11% wall clock outright. So file_io
    stays at half and data_copy gets the lot -- 1.6x measured load, against 18.1x before.
    Symmetrical-looking limits here would be a regression that reads like tidying.
    """
    monkeypatch.delenv("SPOTLIGHT_IO_CONCURRENCY", raising=False)
    monkeypatch.delenv("SPOTLIGHT_COPY_CONCURRENCY", raising=False)

    for n in (8, 30, 64):
        monkeypatch.setenv("LSB_DJOB_NUMPROC", str(n))
        spec = stores._context_spec()
        assert spec["file_io_concurrency"]["limit"] == n // 2
        assert spec["data_copy_concurrency"]["limit"] == n
        assert stores.slots() == n, "the kernel pool must size from the same number"

    # A 1-slot job would otherwise get a limit of 0 and serialise against itself.
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "1")
    spec = stores._context_spec()
    assert spec["file_io_concurrency"]["limit"] == 2
    assert spec["data_copy_concurrency"]["limit"] == 2


def test_both_tensorstore_limits_stay_overridable(monkeypatch):
    """Scicomp may agree to more load for a one-off; that must not need a code change."""
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "30")
    monkeypatch.setenv("SPOTLIGHT_IO_CONCURRENCY", "48")
    monkeypatch.setenv("SPOTLIGHT_COPY_CONCURRENCY", "7")
    spec = stores._context_spec()
    assert spec["file_io_concurrency"]["limit"] == 48
    assert spec["data_copy_concurrency"]["limit"] == 7


def test_one_definition_of_the_slot_count(monkeypatch):
    """The total only means something if every pool sizes itself from the same number, so
    nothing may read LSB_DJOB_NUMPROC on its own. The differing defaults are deliberate --
    they apply only off-cluster, where there is no reservation to honour."""
    import pathlib
    src = pathlib.Path(stores.__file__).parent
    # The variable NAME is fine anywhere -- prose mentions it, and `scripts.runner()` puts
    # it in a shell string for LSF to expand. Reading it in Python is what fragments the
    # budget.
    culprits = [f.name for f in src.glob("*.py")
                if 'getenv("LSB_DJOB_NUMPROC' in f.read_text()
                or 'environ["LSB_DJOB_NUMPROC' in f.read_text()]
    assert culprits == ["stores.py"], (
        f"{culprits} size a pool without going through stores.slots()")

    # LSF's number wins outright and is NEVER clamped: 30 slots on a 128-core host means 30,
    # and a job that reserved more than the box has is the scheduler's problem, not ours.
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "11")
    assert stores.slots() == stores.slots(999) == 11

    # Off-cluster there is no reservation, so the machine is the constraint -- otherwise
    # `n_cores_int_correct = 20` builds a 20-thread pool on a 4-core laptop.
    monkeypatch.delenv("LSB_DJOB_NUMPROC")
    import os
    cores = os.cpu_count() or 8
    assert stores.slots(cores + 100) == cores            # clamped down to the box
    assert stores.slots(1) == 1                          # never clamped UP
    assert stores.slots(0) == 1                          # and never returns 0
