"""OME-TIFF output.

The interesting properties are the ones a downstream reader depends on and that a
plausible-looking implementation gets wrong: the axis order, the voxel calibration, and
that a tile larger than memory is streamed rather than materialised.
"""

import numpy as np
import pytest
import tifffile

from spotlight import config, tiffout
from spotlight.formats import FORMATS, OUTPUT_FORMATS


XML = """<?xml version="1.0" encoding="UTF-8"?>
<SpimData version="0.2"><SequenceDescription><ViewSetups>
  <ViewSetup><id>0</id><size>1984 1984 667</size>
    <voxelSize><unit>um</unit><size>0.1 0.2 0.4</size></voxelSize></ViewSetup>
  <ViewSetup><id>1</id><size>10 10 10</size></ViewSetup>
  <ViewSetup><id>2</id><size>10 10 10</size>
    <voxelSize><unit>furlong</unit><size>1 1 1</size></voxelSize></ViewSetup>
</ViewSetups></SequenceDescription></SpimData>
"""


@pytest.fixture
def xml(tmp_path):
    p = tmp_path / "dataset.xml"
    p.write_text(XML)
    return {"dataset_xml": str(p)}


# ─── voxel size ──────────────────────────────────────────────────────────────


def test_voxel_size_is_read_zyx_from_the_xml_which_stores_it_xyz():
    """The transposition that would silently mis-scale z: SpimData lists voxelSize as
    X Y Z (matching its <size>), and everything internal here is (Z, Y, X)."""
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    (d / "dataset.xml").write_text(XML)
    got = tiffout.voxel_size_um({"dataset_xml": str(d / "dataset.xml")}, 0)
    assert got == (0.4, 0.2, 0.1), "expected (z, y, x) from an x y z record"


def test_missing_calibration_is_none_not_one(xml):
    """A TIFF claiming 1 um/px is worse than one claiming nothing, because the first is
    believed. Setup 1 has no voxelSize at all."""
    assert tiffout.voxel_size_um(xml, 1) is None


def test_an_unrecognised_unit_is_refused_rather_than_assumed(xml):
    assert tiffout.voxel_size_um(xml, 2) is None


def test_no_xml_configured_is_none(tmp_path):
    assert tiffout.voxel_size_um({}, 0) is None
    assert tiffout.voxel_size_um({"dataset_xml": str(tmp_path / "nope.xml")}, 0) is None


def test_an_unknown_setup_is_none(xml):
    assert tiffout.voxel_size_um(xml, 99) is None


# ─── the written file ────────────────────────────────────────────────────────


def _write(tmp_path, shape=(20, 8, 6), voxel=(0.4, 0.2, 0.1), planes=7, **kw):
    ref = (np.arange(int(np.prod(shape)), dtype=np.uint16) % 4096).reshape(shape)
    calls = []

    def read_slab(z0, z1):
        calls.append((z0, z1))
        return ref[z0:z1]

    path = tiffout.write_tile_ome_tiff(tmp_path / "t.ome.tif", read_slab, shape,
                                       np.uint16, voxel=voxel, planes=planes, **kw)
    return path, ref, calls


def test_round_trips_exactly(tmp_path):
    path, ref, _ = _write(tmp_path)
    np.testing.assert_array_equal(tifffile.imread(path), ref)


def test_axes_are_zyx_so_a_reader_sees_a_z_stack(tmp_path):
    """Without this OME-XML a stack of IFDs is ambiguous, and readers commonly default to
    time -- which loads as 20 timepoints of one plane rather than one 20-plane volume."""
    path, ref, _ = _write(tmp_path)
    with tifffile.TiffFile(path) as tf:
        assert tf.is_ome
        assert tf.series[0].axes == "ZYX"
        assert tf.series[0].shape == ref.shape


def test_the_voxel_size_survives_into_the_ome_xml(tmp_path):
    path, _, _ = _write(tmp_path)
    with tifffile.TiffFile(path) as tf:
        x = tf.ome_metadata
    assert 'PhysicalSizeX="0.1"' in x and 'PhysicalSizeY="0.2"' in x
    assert 'PhysicalSizeZ="0.4"' in x
    assert x.count("µm") >= 3 or x.count("&#181;m") >= 3


def test_no_voxel_size_writes_no_calibration_rather_than_a_wrong_one(tmp_path):
    path, _, _ = _write(tmp_path, voxel=None)
    with tifffile.TiffFile(path) as tf:
        assert "PhysicalSizeX" not in (tf.ome_metadata or "")


def test_it_is_bigtiff_regardless_of_size(tmp_path):
    """Set unconditionally: level-0 tiles here are 4.9-9.5 GiB and classic TIFF caps at
    4 GiB of offsets, so deciding per file would only add a way to get it wrong."""
    path, _, _ = _write(tmp_path)
    with tifffile.TiffFile(path) as tf:
        assert tf.is_bigtiff


def test_it_streams_in_slabs_rather_than_materialising_the_tile(tmp_path):
    """The property that makes a 9.5 GiB tile fit a 15 GiB slot. 20 planes at 7 per slab
    is 3 reads, and none of them is the whole stack."""
    _, _, calls = _write(tmp_path, shape=(20, 8, 6), planes=7)
    assert calls == [(0, 7), (7, 14), (14, 20)]
    assert all(z1 - z0 <= 7 for z0, z1 in calls)


def test_every_plane_is_counted_including_the_last_partial_slab(tmp_path):
    """The generator bug this had: ticking once per slab AFTER yielding its planes loses
    the final slab, because the consumer stops calling next() on the last plane. It
    reported 640/667 on a real tile."""
    class Bar:
        def __init__(self):
            self.n = 0

        def advance(self, k=1):
            self.n += k

    bar = Bar()
    _write(tmp_path, shape=(20, 8, 6), planes=7, progress=bar)
    assert bar.n == 20


def test_zstd_is_actually_applied(tmp_path):
    path, ref, _ = _write(tmp_path)
    with tifffile.TiffFile(path) as tf:
        assert tf.pages[0].compression == 50000, "expected zstd"


def test_uncompressed_is_still_readable(tmp_path):
    """The escape hatch for a site without imagecodecs."""
    path, ref, _ = _write(tmp_path, compression=None)
    np.testing.assert_array_equal(tifffile.imread(path), ref)


def test_the_path_is_one_file_per_tile(tmp_path):
    cfg = {"output_intensity_path": str(tmp_path / "out")}
    assert tiffout.tiff_tile_path(cfg, 0).name == "tile0.ome.tif"
    assert tiffout.tiff_tile_path(cfg, 12).name == "tile12.ome.tif"
    assert tiffout.tiff_tile_path(cfg, 0).parent == tmp_path / "out"


# ─── format registration ─────────────────────────────────────────────────────


def test_tiff_is_output_only():
    """There is no TIFF reader here, so offering it as an input would promise one."""
    assert "tiff" in OUTPUT_FORMATS
    assert "tiff" not in FORMATS


def test_tiff_is_not_in_the_driver_table():
    """`_SPEC` maps formats to tensorstore drivers and tensorstore has no TIFF writer; an
    entry there would be a driver name that does not exist."""
    from spotlight.formats import _SPEC
    assert "tiff" not in _SPEC


def test_output_format_tiff_validates_and_input_format_tiff_does_not(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    common = ('results_root = "r"\ninput_intensity_path = "i"\n'
              'output_intensity_path = "o"\n')
    (tmp_path / "LocalPreferences.toml").write_text(
        f'[spotlight]\n{common}input_format = "zarr2"\noutput_format = "tiff"\n')
    assert config.load_config()["output_format"] == "tiff"

    (tmp_path / "LocalPreferences.toml").write_text(
        f'[spotlight]\n{common}input_format = "tiff"\n')
    with pytest.raises(ValueError, match="cannot be read"):
        config.load_config()


def test_the_pre_rename_format_key_is_rejected_not_ignored(tmp_path, monkeypatch):
    """Ignoring it would read an n5 store with the zarr2 default -- the wrong driver."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "LocalPreferences.toml").write_text(
        '[spotlight]\nresults_root = "r"\ninput_intensity_path = "i"\n'
        'output_intensity_path = "o"\nformat = "n5"\n')
    with pytest.raises(ValueError, match="input_format"):
        config.load_config()
