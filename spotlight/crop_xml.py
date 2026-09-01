#!/usr/bin/env python3
"""The dataset.xml for a `crop_margin` copy of a SpimData2 / BigStitcher dataset.

    python -m spotlight.crop_xml <in.xml> <out.xml> <margin>

Two edits per view and nothing else. The file is patched LINE BY LINE rather than
re-serialised: a BigStitcher xml runs to hundreds of thousands of lines, and ElementTree
would rewrite every one of them, so a diff would show the whole file instead of the crop.

* `<ViewSetup><size>` shrinks by `2 * margin` on each axis. `<voxelSize><size>` is left
  alone -- the crop changes how many voxels a tile has, not how big one is.
* each `<ViewRegistration>` gains a `crop offset` translation of `+margin` voxels,
  appended LAST. SpimData composes a registration as `M_0 @ M_1 @ ... @ M_last`, so the
  last-listed transform applies FIRST to a raw voxel coordinate -- and that is what the
  crop is: `world = M_old @ (v_cropped + margin)`. Last also puts it ahead of
  `calibration`, so the offset stays in voxels on an anisotropic dataset.

So no tile moves: each one keeps its solved position and its box gets `2 * margin`
smaller. `StitchingResults` shifts are relative and left untouched, as is
`IntensityAdjustments`.

Write the result beside the CROPPED store, since the `<zarr>` path is normally relative --
this refuses to overwrite an existing file, because the input is usually the only copy of
a stitching solve.
"""

import re
import sys
from pathlib import Path

__all__ = ["crop_xml"]

# 6-space indent for a <ViewTransform>, matching what BigStitcher writes. Cosmetic; the
# parsers do not care.
_XFORM = ("      <ViewTransform type=\"affine\">\n"
          "        <Name>crop offset</Name>\n"
          "        <affine>1.0 0.0 0.0 {m} 0.0 1.0 0.0 {m} 0.0 0.0 1.0 {m}</affine>\n"
          "      </ViewTransform>")

# Integer triplet only, so a float `<voxelSize><size>` cannot match even if the
# in-voxelSize guard below is removed. Both checks are wanted: a voxel size of `1 1 4` is
# all integers.
_SIZE = re.compile(r"^(\s*<size>)(\d+) (\d+) (\d+)(</size>\s*)$")


def crop_xml(text, margin):
    """`(patched text, n_setups, n_registrations)` for a `margin`-voxel crop."""
    out, in_voxel, n_size, n_reg = [], False, 0, 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<voxelSize"):
            in_voxel = True
        elif stripped.startswith("</voxelSize"):
            in_voxel = False
        hit = None if in_voxel else _SIZE.match(line)
        if hit:
            sizes = [int(v) - 2 * margin for v in hit.group(2, 3, 4)]
            if min(sizes) <= 0:
                raise SystemExit(f"margin {margin} leaves nothing of a "
                                 f"{' '.join(hit.group(2, 3, 4))} view: {stripped}")
            line = f"{hit.group(1)}{' '.join(str(v) for v in sizes)}{hit.group(5)}"
            n_size += 1
        elif stripped == "</ViewRegistration>":
            out.append(_XFORM.format(m=float(margin)))
            n_reg += 1
        out.append(line)

    # The counts are the check that the line matching found the right lines: one size per
    # ViewSetup, one offset per ViewRegistration. A `<size>` the regex missed (different
    # indentation, attributes, one line) would otherwise leave a tile 2*margin too large.
    want_size = text.count("<ViewSetup>")
    want_reg = text.count("<ViewRegistration ")
    if (n_size, n_reg) != (want_size, want_reg):
        raise SystemExit(f"patched {n_size} <size> of {want_size} <ViewSetup> and {n_reg} "
                         f"of {want_reg} <ViewRegistration>; refusing to write a partly "
                         f"cropped xml")
    return "\n".join(out) + "\n", n_size, n_reg


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        raise SystemExit(f"usage: python -m spotlight.crop_xml <in.xml> <out.xml> <margin>"
                         f"\n{__doc__}")
    src, dst, margin = Path(argv[0]), Path(argv[1]), int(argv[2])
    if margin <= 0:
        raise SystemExit(f"margin must be positive, got {margin}")
    if dst.exists():
        raise SystemExit(f"{dst} exists; move it aside first (the input xml is usually the "
                         "only copy of a stitching solve)")

    text = src.read_text(encoding="utf-8")
    patched, n_size, n_reg = crop_xml(text, margin)
    dst.write_text(patched, encoding="utf-8")
    print(f"{dst}: {n_size} view size(s) shrunk by {2 * margin} per axis, {n_reg} "
          f"registration(s) offset by +{margin} voxel(s)")

    # The store the new xml resolves to, which is the mistake worth catching: a cropped xml
    # beside the UNCROPPED store describes every tile 2*margin too small.
    zarr = re.search(r'<zarr type="(relative|absolute)">([^<]+)</zarr>', text)
    if zarr:
        store = Path(zarr.group(2))
        store = store if zarr.group(1) == "absolute" else dst.parent / store
        print(f"  reads {store}" + ("" if store.exists() else "  <-- NOT PRESENT YET"))
        if not store.exists():
            print("  that store is where the cropped tiles have to land "
                  "(output_intensity_path), not the uncropped input")


if __name__ == "__main__":
    main()
