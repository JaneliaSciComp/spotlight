"""A dataset xml that applies the BaSiC fields while BigStitcher reads, instead of a copy.

    python -m spotlight run basic --xml              # ... -> basic -> xml
    python -m spotlight xml [XML ...]                # default: dataset_xml

The `correct` stage rewrites every voxel of the dataset. This writes one xml instead: the
original xml with its `<ImageLoader>` wrapped in multiview-reconstruction's flat-field
loader (`spimreconstruction.wrapped.flatfield.viewer`), and one `<FlatField>` per setup
pointing at its camera's fields. Every BigStitcher / BigStitcher-Spark stage that reads
images through the xml -- intensity matching, fusion, the viewer -- then sees corrected
voxels, with nothing rewritten.

The java loader computes `(raw - dark) * mean(bright - dark) / (bright - dark)`, clamped
to the type's range. `Bright-field.tif = max(flat, flat_floor) + dark` makes that exactly
`correct`'s `max((raw - dark) / flat, 0)`, because the flat field is normalised to mean 1.
Checked against a `correct --mode basic` store: 99.98% of level-0 voxels identical, the
rest one gray level apart (rounding).

Given a split xml (`split.viewerimgloader`), this wraps the loader NESTED inside it, whose
setup ids for the fields are the original ones: `python -m spotlight xml dataset.split.xml`. Re-splitting the flat-field xml would work
too, but would throw away whatever was registered on the split views since.

Both files are written beside their input, so the wrapped loader's relative paths still
resolve, and never over an existing file: an xml is where registrations get saved.
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .config import basic_field_paths, basic_view, camera_groups
from .correct import FLAT_FLOOR
from .fields import _oriented_field, _read_field_tiff
from .formats import write_plane_tiff
from .stores import source_pyramid_shapes

__all__ = ["WRAPPER", "bright_field_path", "write_bright_fields", "wrap_xml",
           "write_flatfield_xml"]

WRAPPER = "spimreconstruction.wrapped.flatfield.viewer"

_OPEN = re.compile(r'^(\s*)<ImageLoader\b[^>]*\bformat="([^"]+)"')
_STORE = re.compile(r'<(zarr|n5|hdf5)\b[^>]*\btype="(relative|absolute)"[^>]*>([^<]+)<')
_BASE = re.compile(r'<BasePath\b[^>]*\btype="(relative|absolute)"[^>]*>([^<]+)<')


def bright_field_path(cfg, camera):
    """`camera{N}/Bright-field.tif`, beside the camera's flat and dark fields."""
    return basic_field_paths(cfg, camera)[0].with_name("Bright-field.tif")


def write_bright_fields(cfg):
    """Write each camera's `max(flat, flat_floor) + dark`; `{setup: (bright, dark)}`.

    The dark field is referenced where `basic` wrote it, so it must already be canonical
    (Y, X) -- a transposed one from an old run would pair with a canonical bright field.
    """
    floor = np.float32(cfg.get("flat_floor") or FLAT_FLOOR)
    raw = basic_view(cfg)   # the store BaSiC was fitted on: the one the xml must read
    out = {}
    for cam, group in enumerate(camera_groups(cfg)):
        flat_p, dark_p = basic_field_paths(cfg, cam)
        for p in (flat_p, dark_p):
            if not p.exists():
                raise SystemExit(f"{p} is missing; run `python -m spotlight basic {cam}` "
                                 f"first")
        full_yx = tuple(source_pyramid_shapes(raw, group[0])[0][1:])
        dark = _read_field_tiff(dark_p)
        if dark.shape != full_yx:
            raise SystemExit(f"{dark_p} is {dark.shape}, not the tile's (Y, X) {full_yx}; "
                             f"re-run `python -m spotlight basic {cam}` to rewrite it")
        flat = _oriented_field(_read_field_tiff(flat_p), full_yx, flat_p)
        bright_p = bright_field_path(cfg, cam)
        write_plane_tiff(bright_p, np.maximum(flat, floor) + dark)
        print(f"xml: camera {cam + 1} -> {bright_p} (flat mean={float(flat.mean()):.4f}, "
              f"dark mean={float(dark.mean()):.2f})")
        out.update({s: (bright_p, dark_p) for s in group})
    return out


def _loader_span(lines):
    """`(first, last)` line of the image loader to wrap: the first one that is not a split
    loader, i.e. the dataset's own loader or the one nested inside a split xml."""
    for i, line in enumerate(lines):
        m = _OPEN.match(line)
        if m and not m.group(2).startswith("split."):
            depth = 0
            for j in range(i, len(lines)):
                opened = lines[j].count("<ImageLoader")
                depth += opened - lines[j].count("</ImageLoader>")
                if opened and lines[j].rstrip().endswith("/>"):
                    depth -= 1
                if depth == 0:
                    return i, j
            break
    raise SystemExit("no <ImageLoader> to wrap (or it is never closed)")


def wrap_xml(text, fields, timepoint=0):
    """`text` with its image loader wrapped in the flat-field loader; `fields` is
    `{setup: (bright, dark)}`."""
    if WRAPPER in text:
        raise SystemExit("this xml already applies flat-field correction; wrapping it "
                         "again would apply the fields twice")
    lines = text.splitlines()
    i, j = _loader_span(lines)
    pad = _OPEN.match(lines[i]).group(1)
    # ponytail: timepoint 0 only, like every other stage; add a FlatField per timepoint if
    # spotlight ever handles more than one
    entries = [f'{pad}    <FlatField Timepoint="{timepoint}" ViewSetup="{s}">\n'
               f'{pad}      <BrightImg format="tif" type="absolute">{escape(str(b))}</BrightImg>\n'
               f'{pad}      <DarkImg format="tif" type="absolute">{escape(str(d))}</DarkImg>\n'
               f'{pad}    </FlatField>'
               for s, (b, d) in sorted(fields.items())]
    wrapped = ([f'{pad}<ImageLoader format="{WRAPPER}" Active="true" Cached="false">',
                f'{pad}  <WrappedImgLoader>']
               + ["    " + line for line in lines[i:j + 1]]
               + [f'{pad}  </WrappedImgLoader>', f'{pad}  <FlatFields>'] + entries
               + [f'{pad}  </FlatFields>', f'{pad}</ImageLoader>'])
    return "\n".join(lines[:i] + wrapped + lines[j + 1:]) + "\n"


def _check_reads(text, src, store):
    """Refuse an xml whose loader reads another store than the one BaSiC was fitted on:
    pointed at an already-corrected copy, it would apply the fields twice."""
    lines = text.splitlines()
    i, j = _loader_span(lines)
    hit = _STORE.search("\n".join(lines[i:j + 1]))
    if hit is None or "://" in str(store):
        print(f"xml: WARNING could not check the store {src} reads; make sure it is the "
              f"raw data ({store}), or the fields get applied twice")
        return
    path = Path(hit.group(3).strip())
    if hit.group(2) == "relative":
        root, base = src.parent, _BASE.search(text)
        if base:
            kind, value = base.group(1), Path(base.group(2).strip())
            root = value if kind == "absolute" else root / value
        path = root / path
    if path.resolve() != Path(store).resolve():
        raise SystemExit(f"{src} reads {path}, but the fields were fitted on {store}. "
                         f"Applying them to any other store is wrong -- to an already "
                         f"corrected one, twice.")


def _target(src, cfg):
    """`(src, text, dst)`, checked before anything is written: the destination must not
    exist, and the source must read the store the fields were fitted on."""
    src = Path(src)
    if src.suffix != ".xml":
        raise SystemExit(f"{src} is not an .xml file")
    dst = src.with_name(f"{src.stem}.flatfield.xml")
    if dst.exists():
        raise SystemExit(f"{dst} exists; move it aside first (it may hold registrations "
                         f"saved since it was written)")
    text = src.read_text(encoding="utf-8")
    _check_reads(text, src, cfg["input_basic_path"])
    return src, text, dst


def write_flatfield_xml(cfg, xmls=None):
    """The `xml` stage: bright fields, then `<name>.flatfield.xml` beside each of `xmls`
    (default: `dataset_xml`). Returns the xml(s) written."""
    xmls = xmls or [cfg.get("dataset_xml")]
    for xml in xmls:
        if not xml or "://" in str(xml):
            raise SystemExit(f"the xml stage writes beside its input xml, which must be a "
                             f"local path; got {xml!r}")
    targets = [_target(x, cfg) for x in xmls]
    fields = write_bright_fields(cfg)
    for src, text, dst in targets:
        dst.write_text(wrap_xml(text, fields), encoding="utf-8")
        print(f"xml: {dst} -- {len(fields)} setup(s) flat/dark-corrected on read")
    return [dst for _, _, dst in targets]
