import sys
from pathlib import Path

# Kept so the suite runs from a bare checkout -- before `pixi install`, or in CI -- even
# though the package is normally installed editable into the environment. When it IS
# installed this shadows it with the working tree, which is what you want while developing.
# `tests/` is on the path for the helper modules (make_store, golden_io).
ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
