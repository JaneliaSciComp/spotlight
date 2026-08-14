"""The progress bar, in both of its output modes.

The mode is the whole point: these stages run interactively AND under `bsub` with stdout
in a file, and a carriage-return bar in a log file is unreadable (`\\r` erases nothing, so
the whole run becomes one line). So the tests are mostly about what does NOT appear.
"""

import io
import time

import pytest

from spotlight.progress import Progress, _duration


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_a_terminal_gets_a_redrawn_bar():
    out = Tty()
    with Progress(4, "stats", stream=out, min_interval=0) as bar:
        for _ in range(4):
            bar.advance()
    s = out.getvalue()
    assert "\r" in s and "#" in s
    assert "4/4" in s and "100%" in s
    assert s.endswith("\n"), "the bar must end its line, or the next print lands on it"


def test_a_log_file_gets_lines_and_never_a_carriage_return():
    """The failure this design exists to prevent: one unreadable line in an LSF log."""
    out = io.StringIO()
    with Progress(100, "correct setup 0", stream=out) as bar:
        for _ in range(100):
            bar.advance()
    s = out.getvalue()
    assert "\r" not in s, "carriage returns do not erase in a file"
    assert "#" not in s, "no bar glyphs in a log"
    assert "correct setup 0 done: 100/100" in s


def test_a_log_file_is_not_one_line_per_unit():
    """The old behaviour printed every chunk; a 1000-chunk job made 1000 lines that never
    said how much was left."""
    out = io.StringIO()
    with Progress(1000, "stats", stream=out) as bar:
        for _ in range(1000):
            bar.advance()
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    assert len(lines) <= 12, f"{len(lines)} lines for 1000 units"
    assert any("50%" in ln for ln in lines), "the steps it does emit should be readable"
    assert any("eta" in ln for ln in lines)


def test_the_final_line_is_always_emitted_whatever_the_rate_limit_decided():
    """A job that finishes inside one rate-limit window must still report."""
    out = io.StringIO()
    bar = Progress(3, "x", stream=out)
    bar.advance(3)
    bar.close()
    assert "x done: 3/3" in out.getvalue()


def test_close_is_idempotent():
    out = io.StringIO()
    bar = Progress(2, "x", stream=out)
    bar.advance(2)
    bar.close()
    bar.close()
    assert out.getvalue().count("done:") == 1


def test_advance_after_close_does_not_reopen_the_line():
    out = Tty()
    bar = Progress(2, "x", stream=out, min_interval=0)
    bar.advance()
    bar.close()
    before = out.getvalue()
    bar.advance()
    assert out.getvalue() == before


def test_zero_total_does_not_divide_by_zero():
    """A setup with no shards is a real case (an empty tile), not an error."""
    out = io.StringIO()
    with Progress(0, "empty", stream=out) as bar:
        pass
    assert "0/0" in out.getvalue()


def test_a_stream_without_isatty_is_treated_as_a_file():
    class Bare:
        def __init__(self):
            self.buf = []

        def write(self, s):
            self.buf.append(s)

        def flush(self):
            pass
    out = Bare()
    with Progress(2, "x", stream=out) as bar:
        bar.advance(2)
    assert "\r" not in "".join(out.buf)


def test_advance_is_thread_safe():
    """Called from the event loop today, but a garbled count would be a silent wrong
    number rather than a crash, so the lock is worth asserting."""
    import threading
    out = io.StringIO()
    bar = Progress(400, "x", stream=out)
    threads = [threading.Thread(target=lambda: [bar.advance() for _ in range(100)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bar.close()
    assert bar.n == 400


@pytest.mark.parametrize("seconds,text", [
    (0, "0s"), (45, "45s"), (60, "1:00"), (125, "2:05"), (3600, "1:00:00"),
    (3725, "1:02:05"), (None, "--"), (float("inf"), "--"), (float("nan"), "--"),
])
def test_duration_formatting(seconds, text):
    assert _duration(seconds) == text
