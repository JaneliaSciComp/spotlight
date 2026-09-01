"""`crop_margin`: trim a tile's deconvolution-artifact border in the copy pass.

Two halves that have to agree, or the dataset is quietly mispositioned rather than broken:
the voxels (`correct._cropped`) and the geometry (`spotlight.crop_xml`). The claim is that
a cropped tile still lands where its uncropped self did -- checked through
`aggregate._view_registration_transforms`, the code that actually consumes the xml.

Mutation-checked: dropping the `translate_to[0]` rebase, cropping only the far faces,
letting a correcting mode crop, prepending the crop offset instead of appending it, and
dropping crop_xml's in-voxelSize guard or its count check all fail at least one of these.
"""

import difflib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from make_store import DEPTHS, X, Y, volume, write_store

from spotlight import aggregate, config, correct
from spotlight.crop_xml import crop_xml
from spotlight.formats import _SPEC, canonical_view
from spotlight.stores import _context
from spotlight.__main__ import main

import tensorstore as ts


MARGIN = 2

# One setup, deliberately awkward: an INTEGER voxel size (so a `<size>` regex that ignores
# the voxelSize block corrupts it) and an ANISOTROPIC calibration (so an offset composed in
# the wrong place lands 4x off in z).
MINI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<SpimData version="0.2">
  <SequenceDescription>
    <ImageLoader format="bdv.multimg.zarr" version="3.1">
      <zarr type="relative">dataset.ome.zarr</zarr>
    </ImageLoader>
    <ViewSetups>
      <ViewSetup>
        <id>0</id>
        <name>0</name>
        <size>64 48 63</size>
        <voxelSize>
          <unit>um</unit>
          <size>1 1 4</size>
        </voxelSize>
      </ViewSetup>
    </ViewSetups>
  </SequenceDescription>
  <ViewRegistrations>
    <ViewRegistration timepoint="0" setup="0">
      <ViewTransform type="affine">
        <Name>Stitching Transform</Name>
        <affine>1.0 0.0 0.0 5.0 0.0 1.0 0.0 -3.0 0.0 0.0 1.0 2.0</affine>
      </ViewTransform>
      <ViewTransform type="affine">
        <Name>calibration</Name>
        <affine>1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 4.0 0.0</affine>
      </ViewTransform>
    </ViewRegistration>
  </ViewRegistrations>
</SpimData>
"""


def _world(text, setup, voxel_xyz):
    """Where one (x, y, z) voxel index of `setup` lands, per the xml's own consumer."""
    root = ET.fromstring(text)
    m = aggregate._view_registration_transforms(root)[setup]
    return m @ np.array([*voxel_xyz, 1.0])


# ─── the xml ──────────────────────────────────────────────────────────────────

def test_a_cropped_tile_lands_where_its_uncropped_self_did():
    """The whole point. Voxel (0,0,0) of the cropped tile IS voxel (m,m,m) of the original,
    so the two must map to the same world point -- otherwise every tile shifts by the margin
    and the stitching solve is silently wrong rather than visibly broken.
    """
    cropped, _, _ = crop_xml(MINI_XML, MARGIN)
    for corner in ((0, 0, 0), (60, 44, 59)):
        shifted = tuple(c + MARGIN for c in corner)
        np.testing.assert_allclose(_world(cropped, 0, corner),
                                   _world(MINI_XML, 0, shifted))


def test_the_view_size_shrinks_by_twice_the_margin_and_the_voxel_size_does_not():
    """A crop changes how many voxels a tile has, not how big one is. Both are `<size>`."""
    cropped, n_size, n_reg = crop_xml(MINI_XML, MARGIN)
    assert (n_size, n_reg) == (1, 1)
    root = ET.fromstring(cropped)
    assert aggregate._view_setup_sizes(root)[0] == (64 - 4, 48 - 4, 63 - 4)
    assert root.find(".//voxelSize/size").text == "1 1 4"


def test_nothing_but_the_offset_is_added():
    """Patched line by line, not re-serialised: a BigStitcher xml is hundreds of thousands
    of lines and a diff of the whole file hides the crop.
    """
    cropped, _, n_reg = crop_xml(MINI_XML, MARGIN)
    before, after = MINI_XML.splitlines(), cropped.splitlines()
    assert len(after) == len(before) + 4 * n_reg
    diff = [l for l in difflib.ndiff(before, after) if l[0] in "+-"]
    # 4 lines of transform per registration, plus the one rewritten <size> line.
    assert sum(l.startswith("+") for l in diff) == 4 * n_reg + 1, diff
    assert sum(l.startswith("-") for l in diff) == 1, diff


def test_a_size_the_line_matcher_missed_is_refused_rather_than_half_cropped():
    """The count check. A one-line ViewSetup does not match, and a tile left 2*margin too
    large reads past the end of the cropped store -- worth a hard stop, not a warning.
    """
    one_line = MINI_XML.replace("""      <ViewSetup>
        <id>0</id>
        <name>0</name>
        <size>64 48 63</size>""", "      <ViewSetup><id>0</id><size>64 48 63</size>")
    with pytest.raises(SystemExit, match="partly cropped"):
        crop_xml(one_line, MARGIN)


def test_a_margin_that_consumes_a_view_is_refused():
    with pytest.raises(SystemExit, match="leaves nothing"):
        crop_xml(MINI_XML, 32)      # 63 - 64 in z


# ─── the voxels ───────────────────────────────────────────────────────────────

@pytest.fixture
def experiment(tmp_path, monkeypatch):
    store = write_store(tmp_path / "in", "zarr2", setups=(0, 1))
    monkeypatch.chdir(tmp_path)
    xml = tmp_path / "dataset.xml"
    xml.write_text('<?xml version="1.0"?><SpimData version="0.2">'
                   "<SequenceDescription><ViewSetups>"
                   + "".join(f"<ViewSetup><id>{i}</id><size>{X} {Y} {DEPTHS[i]}</size>"
                             "</ViewSetup>" for i in (0, 1))
                   + "</ViewSetups></SequenceDescription></SpimData>")
    config.set_config(
        input_basic_path=store["input_basic_path"],
        output_basic_path=store["output_basic_path"],
        input_intensity_path=store["input_intensity_path"],
        output_intensity_path=store["output_intensity_path"],
        results_root=str(tmp_path / "results"), qstacks_dir=str(tmp_path / "qstacks"),
        input_format="zarr2", output_format="zarr2", last_setup=1, setups_per_camera=2,
        chunk_size=[32, 32, 32], shard_size=[64, 64, 64], dataset_xml=str(xml),
        lsf_project="p", output_stem=str(tmp_path / "o"), error_stem=str(tmp_path / "e"),
        n_cores_int_correct=2, crop_margin=MARGIN,
    )
    return tmp_path


def test_a_crop_writes_the_interior_voxels_and_only_those(experiment):
    """All six faces, and the interior rebased to origin 0. Values encode their own
    coordinates, so an off-by-one origin fails on value and not merely on shape.
    """
    main(["run", "copy", "1"])
    arr = ts.open({"driver": _SPEC["zarr2"]["driver"],
                   "kvstore": {"driver": "file",
                               "path": f"{experiment}/in_out/s1-t0.zarr/0"}},
                  context=_context(), open=True, read=True).result()
    out = np.asarray(canonical_view(arr, _SPEC["zarr2"]["order"])[:, :, :].read().result())
    m, z = MARGIN, DEPTHS[1]
    assert out.shape == (z - 2 * m, Y - 2 * m, X - 2 * m)
    np.testing.assert_array_equal(out, volume(1, z)[m:z - m, m:Y - m, m:X - m])


def test_a_correcting_mode_refuses_to_crop(experiment):
    """The flat/dark field is sliced with the OUTPUT's (y, x), so a cropped correction
    would divide by a field offset by the margin -- silently, and by a plausible amount.
    """
    src = ts.open({"driver": "array", "array": np.zeros((16, 16, 16), np.uint16),
                   "dtype": "uint16"}).result()
    for mode in ("basic", "intensity", "both"):
        with pytest.raises(RuntimeError, match="crop_margin"):
            correct._cropped(src, {"crop_margin": MARGIN}, mode, 0)
    for mode in correct.COPY_MODES:
        assert correct._cropped(src, {"crop_margin": MARGIN}, mode, 0).domain.shape == (12,) * 3


def test_no_margin_is_the_default_and_leaves_the_view_untouched():
    """`crop_margin` unset must not so much as re-wrap the source view: every correction
    run goes through this call."""
    src = ts.open({"driver": "array", "array": np.zeros((4, 4, 4), np.uint16),
                   "dtype": "uint16"}).result()
    assert config.DEFAULTS["crop_margin"] == 0
    assert correct._cropped(src, {}, "both", 0) is src
