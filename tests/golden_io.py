"""Read the reference values `gen_golden.jl` dumped from the Julia implementation."""

import json
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).resolve().parent / "golden"


def have_golden():
    return (GOLDEN / "orderstats.json").is_file()


def load_json(name):
    with open(GOLDEN / name) as f:
        return json.load(f)


def load_bin(name):
    """A raw dump plus its sidecar, back as the array Julia held.

    Julia is column-major, so the sidecar says order="F" and the reshape honours it --
    reading these C-ordered would transpose every golden, which is exactly the class of
    bug they exist to catch.
    """
    meta = load_json(name + ".json")
    dtype = {"Float32": "<f4", "Float64": "<f8", "UInt16": "<u2"}[meta["dtype"]]
    flat = np.fromfile(GOLDEN / (name + ".bin"), dtype=dtype)
    return flat.reshape(meta["shape"], order=meta["order"])
