"""A `(` in the experiment path made the generated script a bash syntax error.

`.../mouse_brain_18x_0.025BIS(2026_06_13)/...` went into `-o`/`-e` unquoted, and bash
rejected the whole line at parse time -- nothing was submitted. Mutation-checked:
dropping either `shlex.quote` fails this.
"""

import os
import subprocess

import pytest

from spotlight import config, scripts

UGLY = "/g/x (2026_06_13)/logs/out"


@pytest.mark.skipif(os.name == "nt", reason="drives bash; the cluster is always Linux")
def test_a_log_path_with_shell_metacharacters_parses():
    cfg = dict(config.DEFAULTS, lsf_project="p", output_stem=UGLY, error_stem=UGLY)
    for array in (None, 148):
        line = scripts._bsub(cfg, "j", 1, "is", "cmd", array=array)
        # `bash -n` parses without executing -- the failure this reproduces is at parse time.
        assert subprocess.run(["bash", "-n"], input=line, text=True).returncode == 0, line
        assert "%I" in line or array is None  # LSF, not the shell, expands it
