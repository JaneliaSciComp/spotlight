#!/usr/bin/env python3
"""
Sync tile (ViewRegistration) locations in a BigStitcher/BDV dataset.xml that
only contains a subset of setups (identified by their "s{N}-t0.zarr" zarr
path) to match the tile locations recorded in a "full" dataset.xml that has
one ViewSetup/ViewRegistration per s{N}.

Usage:
    python3 sync_tile_locations.py <main_dataset.xml> <subset_dataset.xml>

The subset file is patched in place. Matching is done via the zgroup
"path" attribute (e.g. path="s1009-t0.zarr") in the subset file's
ImageLoader, which gives the setup id N in the main file for each local
setup id in the subset file.

Each subset ViewRegistration is expected to have a "Translation"
ViewTransform (identity) plus a separate "calibration" ViewTransform that
only carries the Z anisotropy scale factor (no translation). This script
only rewrites the X/Y/Z translation entries of the "Translation" transform,
taken from the main file's "calibration" ViewTransform affine (indices
3, 7, 11 of the row-major 3x4 matrix). The subset file's own "calibration"
transform (z-scale) is left untouched.
"""
import re
import sys


def build_local_to_main_setup_map(subset_xml):
    zgroup_re = re.compile(r'<zgroup setup="(\d+)" tp="0" path="s(\d+)-t0\.zarr"')
    return {int(m.group(1)): int(m.group(2)) for m in zgroup_re.finditer(subset_xml)}


def extract_main_calibration_affines(main_xml):
    reg_re = re.compile(
        r'<ViewRegistration timepoint="0" setup="(\d+)">\s*'
        r'<ViewTransform type="affine">\s*<Name>calibration</Name>\s*'
        r'<affine>([^<]+)</affine>\s*</ViewTransform>\s*</ViewRegistration>'
    )
    return {int(m.group(1)): m.group(2).strip() for m in reg_re.finditer(main_xml)}


def get_translation(affine_str):
    vals = [float(v) for v in affine_str.split()]
    # row-major 3x4 affine: indices 3, 7, 11 are the X, Y, Z translation
    return vals[3], vals[7], vals[11]


def replace_translation(subset_xml, setup_id, tx, ty, tz):
    pattern = re.compile(
        r'(<ViewRegistration timepoint="0" setup="' + str(setup_id) + r'">\s*'
        r'<ViewTransform type="affine">\s*<Name>Translation</Name>\s*<affine>)'
        r'[^<]+'
        r'(</affine>)'
    )
    new_affine = f"1.0 0.0 0.0 {tx} 0.0 1.0 0.0 {ty} 0.0 0.0 1.0 {tz}"
    new_xml, n = pattern.subn(r"\g<1>" + new_affine + r"\g<2>", subset_xml, count=1)
    if n != 1:
        raise RuntimeError(f"failed to patch setup {setup_id}, matched {n} times")
    return new_xml


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <main_dataset.xml> <subset_dataset.xml>")
        sys.exit(1)

    main_path, subset_path = sys.argv[1], sys.argv[2]

    with open(main_path, encoding="utf-8") as f:
        main_xml = f.read()
    with open(subset_path, encoding="utf-8") as f:
        subset_xml = f.read()

    local_to_main = build_local_to_main_setup_map(subset_xml)
    main_affine = extract_main_calibration_affines(main_xml)

    missing = [n for n in local_to_main.values() if n not in main_affine]
    if missing:
        raise RuntimeError(f"setups missing from main file: {missing}")

    for local_id, main_id in local_to_main.items():
        tx, ty, tz = get_translation(main_affine[main_id])
        subset_xml = replace_translation(subset_xml, local_id, tx, ty, tz)

    with open(subset_path, "w", encoding="utf-8") as f:
        f.write(subset_xml)

    print(f"patched {len(local_to_main)} setups in {subset_path}")


if __name__ == "__main__":
    main()
