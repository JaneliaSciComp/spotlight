"""`-W` bounds an element wedged on a bad host; `-Q 140` turns that kill into a retry.

Each is useless without the other, so pin them together. Mutation-checked: dropping
`{_watchdog(cfg)}` from `_bsub`, defaulting `lsf_runlimit_minutes` to 0, and dropping the
`-Q` half of the suffix all fail these.
"""

from spotlight import config, scripts


def _cfg(**over):
    return dict(config.DEFAULTS, lsf_project="p", output_stem="o", error_stem="e", **over)


def test_every_generated_bsub_line_kills_and_requeues_a_wedged_element():
    assert config.DEFAULTS["lsf_runlimit_minutes"] > 0
    cfg = _cfg()
    for array in (None, 560):
        line = scripts._bsub(cfg, "j", 30, "s", "cmd", array=array)
        assert f" -W {cfg['lsf_runlimit_minutes']}" in line, line
        assert ' -Q "140"' in line, line


def test_requeue_is_the_run_limit_code_only():
    """Requeueing 1 would retry the TOML-BOM and output-shape failures forever."""
    assert scripts.REQUEUE_EXIT_CODES.split() == ["140"]


def test_a_zero_run_limit_drops_both_halves():
    """`-Q 140` can never fire without a run limit to produce a 140."""
    line = scripts._bsub(_cfg(lsf_runlimit_minutes=0), "j", 30, "s", "cmd", array=560)
    assert " -W " not in line and " -Q " not in line, line
