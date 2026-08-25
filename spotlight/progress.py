"""A progress bar that is also readable in an LSF log.

    with Progress(len(chunks), "stats camera 1") as bar:
        for chunk in chunks:
            ...
            bar.advance()

The only interesting design point: these stages run BOTH interactively (`python -m
spotlight run both`) and under `bsub`, with stdout redirected to a file. A carriage-return
bar is right for the first and wrong for the second -- `\\r` erases nothing in a file, so a
500-chunk job leaves one enormous unreadable line and `tail -f` shows nothing until the
job ends. So the mode is chosen from `isatty()`:

  * a terminal gets a real bar, redrawn in place with `\\r`
  * anything else gets ordinary lines, emitted only when the percentage crosses a step or
    enough seconds have passed -- a handful of timestamped lines for a long job

That also fixes what the old `print(f"chunk {idx}/{stop} written")` did badly: thousands
of chunks meant thousands of log lines, none of which said how much was left or how long
it would take.

No dependency: tqdm is not in the environment, and this is thirty lines.
"""

import shutil
import sys
import threading
import time

__all__ = ["Progress"]

# Non-tty: emit a line every this many percent, or this many seconds, whichever first.
LOG_PERCENT_STEP = 10
LOG_SECONDS = 60.0


def _duration(seconds):
    """`1:05:03` / `5:03` / `3s` -- short enough to sit inside a one-line bar."""
    if seconds is None or seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Progress:
    """Count `total` units of work and render progress.

    `advance()` is safe to call from several threads. The stages here call it from the
    event loop, but a lock costs nothing at one call per chunk and removes the question.
    """

    def __init__(self, total, label="", stream=None, min_interval=0.1):
        self.total = int(total)
        self.label = label
        self.stream = stream if stream is not None else sys.stdout
        self.min_interval = min_interval
        self.n = 0
        self._lock = threading.Lock()
        self._start = time.perf_counter()
        self._last_draw = 0.0
        self._last_logged_pct = -1
        self._last_logged_at = self._start
        self._closed = False
        try:
            self.tty = self.stream.isatty()
        except (AttributeError, ValueError):
            self.tty = False
        if self.tty:
            self._draw(force=True)

    # ── the two ways of rendering ────────────────────────────────────────────

    def _bar(self, width=28):
        frac = self.n / self.total if self.total else 1.0
        filled = int(width * frac)
        elapsed = time.perf_counter() - self._start
        rate = self.n / elapsed if elapsed > 0 and self.n else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else None
        return (f"{self.label} [{'#' * filled}{'.' * (width - filled)}] "
                f"{frac * 100:3.0f}% {self.n}/{self.total} "
                f"{rate:.1f}/s eta {_duration(eta)}")

    def _draw(self, force=False):
        now = time.perf_counter()
        if not force and now - self._last_draw < self.min_interval:
            return
        self._last_draw = now
        line = self._bar()
        # Pad to the terminal width so a shortening line (eta 1:05 -> 9s) leaves no
        # trailing characters from the previous, longer draw.
        pad = max(0, shutil.get_terminal_size((80, 24)).columns - len(line) - 1)
        self.stream.write("\r" + line + " " * pad)
        self.stream.flush()

    def _log(self, force=False):
        """One ordinary line, rate-limited. For logs, not terminals."""
        now = time.perf_counter()
        pct = int(100 * self.n / self.total) if self.total else 100
        step = pct // LOG_PERCENT_STEP
        due = (force
               or step > self._last_logged_pct // LOG_PERCENT_STEP
               or now - self._last_logged_at >= LOG_SECONDS)
        if not due:
            return
        self._last_logged_pct = pct
        self._last_logged_at = now
        elapsed = now - self._start
        rate = self.n / elapsed if elapsed > 0 and self.n else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else None
        print(f"{self.label} {pct}% ({self.n}/{self.total}) "
              f"elapsed {_duration(elapsed)} eta {_duration(eta)}",
              file=self.stream, flush=True)

    # ── the API ──────────────────────────────────────────────────────────────

    def advance(self, n=1):
        with self._lock:
            self.n += n
            if self._closed:
                return
            if self.tty:
                self._draw()
            else:
                self._log()

    def close(self):
        """Finish the line. Idempotent, so `with` plus an explicit close is fine."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            elapsed = time.perf_counter() - self._start
            if self.tty:
                self._draw(force=True)
                self.stream.write("\n")
                self.stream.flush()
            else:
                # Always a final line, whatever the rate limit last decided -- the
                # summary is the one line a log reader actually looks for.
                print(f"{self.label} done: {self.n}/{self.total} in "
                      f"{_duration(elapsed)}", file=self.stream, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
