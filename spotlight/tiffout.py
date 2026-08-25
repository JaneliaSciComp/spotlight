"""OME-TIFF output: one BigTIFF per corrected tile.

    output_format = "tiff"

Every other output format here is a tensorstore driver dispatched through `formats._SPEC`.
TensorStore has no TIFF WRITER, so this is a separate path -- and it has to be, because
TIFF differs from the zarr formats in the one way that matters to `correct.py`:

  A zarr store is a directory of independent chunk files, so shards can be written in ANY
  order and in parallel. A TIFF is ONE file whose IFDs are laid out in sequence, so the
  planes must be produced in z order, by a single writer.

So the correction's shard-parallel write loop cannot be reused. What is reused is the read
and the kernel: this streams z-slabs, corrects each with the same `ShardCorrection`, and
hands the planes to `tifffile` in order.

STREAMING IS NOT OPTIONAL. A level-0 tile is 4.89 GiB on the worm and 9.50 GiB on RID19
s7, against an LSF slot of ~15 GiB, so `imwrite(whole_array)` would not fit even once, let
alone beside the reads. `TiffWriter.write()` takes an iterator and consumes it lazily,
holding one z-slab plus one being prefetched whatever the tile size. Those sizes also
force `bigtiff=True`: classic TIFF caps at 4 GiB of offsets.

WHY tifffile AND NOT pylibtiff. Rejected on evidence, not taste: 0.6.1 against libtiff
4.6.0 on arm64 macOS writes CORRUPT FILES WITHOUT RAISING -- an 8x8 uint16 test image came
back with compression tag 43856, an invalid PHOTOMETRIC, dimensions of 81505104 and
zero-length pixel data, while `write_image` reported success. The cause is its ctypes
varargs marshalling of `TIFFSetField`: Apple's arm64 ABI passes variadic arguments
differently from x86-64, so the tag values arrive as garbage. It may well work on
linux-64, where the ABI matches its assumption, but this package targets both -- and
silent corruption discovered after writing 171 GiB is the worst failure mode available.
libtiff itself is fine and does carry COMPRESSION_ZSTD; the wrapper is the problem.

VOXEL SIZE COMES FROM dataset.xml, NOT FROM THE ZARR. The input OME-NGFF here declares
`scale: [1,1,1,1,1]` -- a placeholder -- while the real 0.05345 um sits in the SpimData2
`<ViewSetup><voxelSize>`. Writing the NGFF value would produce a TIFF that loads at 1
um/px and mis-scales anything measured from it. Note the XML lists voxel size as X Y Z,
matching its `<size>`, while OME-TIFF metadata is named per axis.
"""

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

__all__ = ["voxel_size_um", "write_tile_ome_tiff", "tiff_tile_path"]

# zstd via imagecodecs: ~2-4x on sparse 16-bit microscopy and fast to decode. Named here
# rather than inline so a site that lacks imagecodecs has one place to change.
COMPRESSION = "zstd"

# How many z planes to read/correct at a time. The buffer is Y*X*PLANES*2 bytes, so at
# 2304x2304 a 64-plane slab is ~680 MiB -- big enough that the read amortises tensorstore
# overhead, small enough that two of them (one in flight, one being written) fit a slot.
DEFAULT_SLAB_PLANES = 64


def tiff_tile_path(cfg, setup):
    """`<output_intensity_path>/tile<setup>.ome.tif`.

    The output path names a DIRECTORY here, not a store: one file per tile is the whole
    point, so the configured path is where they go.
    """
    return Path(cfg["output_intensity_path"]) / f"tile{setup}.ome.tif"


def voxel_size_um(cfg, setup):
    """(z, y, x) voxel size in micrometres from dataset.xml, or None if unavailable.

    None rather than a 1.0 default: a TIFF that silently claims 1 um/px is worse than one
    carrying no calibration at all, because the first is believed.
    """
    path = cfg.get("dataset_xml")
    if not path or not Path(path).is_file():
        return None
    root = ET.parse(path).getroot()
    for vs in root.findall(".//ViewSetups/ViewSetup"):
        if int(vs.findtext("id")) != int(setup):
            continue
        node = vs.find("voxelSize")
        if node is None or node.findtext("size") is None:
            return None
        vx, vy, vz = (float(v) for v in node.findtext("size").split())
        unit = (node.findtext("unit") or "um").strip().lower()
        # SpimData writes "um"/"micrometer"; anything else we do not silently rescale.
        if unit not in ("um", "µm", "micrometer", "micrometre", "micron"):
            return None
        return (vz, vy, vx)
    return None


def _ome_metadata(voxel):
    """OME-XML fields tifffile understands. Axes first -- that is what makes a stack of
    IFDs a z stack rather than a time series.
    """
    meta = {"axes": "ZYX"}
    if voxel is not None:
        vz, vy, vx = voxel
        meta.update(PhysicalSizeX=vx, PhysicalSizeXUnit="µm",
                    PhysicalSizeY=vy, PhysicalSizeYUnit="µm",
                    PhysicalSizeZ=vz, PhysicalSizeZUnit="µm")
    return meta


def _slabs(z_total, planes):
    for z0 in range(0, z_total, planes):
        yield z0, min(z0 + planes, z_total)


def write_tile_ome_tiff(path, read_slab, shape, dtype, voxel=None,
                        planes=DEFAULT_SLAB_PLANES, compression=COMPRESSION,
                        progress=None):
    """Stream one corrected tile to a BigTIFF OME-TIFF.

    `read_slab(z0, z1)` returns the corrected (z1-z0, Y, X) block. It is called from a
    worker thread, so its read overlaps the previous slab's compression and write.

    One slab is prefetched, not many: the writer is strictly sequential, so a deeper queue
    would only buy latency hiding that one slab ahead already provides -- while
    multiplying the largest buffer in the stage.
    """
    Z, Y, X = shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bounds = list(_slabs(Z, planes))

    def _planes():
        # A one-deep prefetch: slab n+1 is read while slab n is compressed and written.
        # Without it the stage strictly alternates read and write and takes twice as long
        # on data this size.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(read_slab, *bounds[0])
            for i in range(len(bounds)):
                block = pending.result()
                if i + 1 < len(bounds):
                    pending = pool.submit(read_slab, *bounds[i + 1])
                for plane in block:
                    # Per plane, not per slab: the consumer stops calling `next()` once
                    # it has the last plane, so anything after the final `yield` never
                    # runs and a per-slab tick would silently lose the last slab
                    # (observed: "done: 640/667").
                    if progress is not None:
                        progress.advance()
                    yield plane

    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as tw:
        tw.write(_planes(), shape=(Z, Y, X), dtype=np.dtype(dtype),
                 compression=compression, metadata=_ome_metadata(voxel))
    return path
