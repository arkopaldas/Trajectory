"""
progress.py - a live "clock" progress bar. Drop-in replacement for tqdm.

    from progress import progress

    for x in progress(items, desc="Working"): ...          # like tqdm(items, desc=...)
    with progress(total=100, desc="Loading") as bar:        # manual updates
        bar.update(10)

What it does
  * Elapsed and remaining time tick every second like a clock, even while a single
    task (an epoch, a solver call, a file chunk) is still running. The remaining time
    counts down from the last estimate and is re-synchronised at every update().
  * The colons of both clocks and a small dot at the start of the line blink every second.
  * Before the first estimate exists (nothing finished yet, or the total is unknown) the bar is
    an animated sweeping loading bar and the remaining time shows "calculating".
  * The bar is green and grows with the fraction of work done (in 1/8 character steps).

Notes
  * Uses a background thread that only redraws the line; it never changes your results.
  * Works on Linux, macOS and Windows (colors need a terminal with ANSI support: Windows
    Terminal, VS Code, PowerShell 7 and modern cmd.exe are fine). Set NO_COLOR=1 to disable colors.
  * When the output is not a terminal (log file, pipe) the clock thread and colors are switched off
    and it behaves like a plain progress bar.
  * Several processes can each draw their own line: set the environment variable
    PROGRESS_POSITION (0, 1, 2, ...) before starting each process, or pass position=.
"""
import os
import sys
import time
import shutil
import threading
from tqdm import tqdm

try:  # make ANSI colors work in older Windows consoles
    import colorama
    colorama.just_fix_windows_console()
except Exception:
    pass

# ANSI escape codes used to colour the bar (the reset code returns to the terminal default colour).
_GREEN = "\033[32m"
_BRIGHT_GREEN = "\033[92m"
_GREY = "\033[90m"
_RESET = "\033[0m"
# Partial-block characters: the bar is drawn in steps of 1/8 of a character.
_EIGHTHS = " ▏▎▍▌▋▊▉"


# Row of the terminal this process draws its bar on; set the PROGRESS_POSITION environment variable per process so that
# several processes do not draw on top of each other.
def _default_position():
    try:
        return int(os.environ.get("PROGRESS_POSITION", "0"))
    except ValueError:
        return 0


# The actual progress bar. It is a tqdm subclass, so it accepts all tqdm arguments, but it redraws the line itself.
class ClockBar(tqdm):
    """tqdm with a live clock, blinking indicators and a green animated bar."""

    # Sets up the clock state and (when the output is a terminal) starts the background redraw thread.
    def __init__(self, *args, **kwargs):
        # everything the renderer needs must exist before tqdm.__init__ draws the first frame
        self._finished = False
        self._eta_ref = None          # estimated remaining seconds at the last update()
        self._eta_ref_t = None        # wall-clock time of that estimate
        self._stop = threading.Event()
        self._thread = None
        self._blink = kwargs.pop("blink", True)
        kwargs.setdefault("position", _default_position())
        kwargs.setdefault("leave", True)
        kwargs.setdefault("dynamic_ncols", True)
        super().__init__(*args, **kwargs)
        if self.disable:
            return
        stream = getattr(self, "fp", sys.stderr)
        self._tty = bool(getattr(stream, "isatty", lambda: False)())
        self._color = self._tty and "NO_COLOR" not in os.environ
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            "█░●○▏".encode(encoding)
            self._unicode = True
        except (UnicodeEncodeError, LookupError):
            self._unicode = False
        if self._tty:
            self._thread = threading.Thread(target=self._tick, name="progress-clock", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ clock thread
    # True while no estimate exists yet (nothing finished, or the total is unknown): the bar then shows a sweeping block.
    def _indeterminate(self):
        return (not self.total) or self.n <= 0

    # Background-thread loop: redraws the bar every second so that the clock keeps running during long tasks.
    def _tick(self):
        while not self._stop.is_set():
            if self._indeterminate():
                wait = 0.1                                   # smooth sweeping animation
            else:
                wait = 1.0 - (time.time() % 1.0) + 0.005     # tick exactly on the second
            if self._stop.wait(wait):
                break
            try:
                self.refresh()
            except Exception:
                break

    # ------------------------------------------------------------------ tqdm hooks
    # Advances the bar by n units (normal tqdm behaviour) and re-synchronises the remaining-time estimate.
    def update(self, n=1):
        result = super().update(n)
        self._sync_eta()
        return result

    # Re-synchronises the counting-down remaining time with the real progress.
    def _sync_eta(self):
        total = self.total
        if not total or self.n <= 0:
            return
        elapsed = self.format_dict["elapsed"]
        if elapsed <= 0:
            return
        self._eta_ref = max(0.0, (total - self.n) * elapsed / self.n)
        self._eta_ref_t = time.time()

    # Stops the redraw thread, then lets tqdm close the bar.
    def close(self):
        if not getattr(self, "_finished", True):
            self._finished = True
            self._stop.set()
            if self._thread is not None and self._thread is not threading.current_thread():
                self._thread.join(timeout=2.0)
        super().close()

    # Text of the current frame (tqdm calls this to draw).
    def __str__(self):
        return self._render()

    # Text of the finished bar; used by ProgressHub to print the completed bar into the scroll-back.
    def final_text(self):
        """The bar rendered as finished (steady dot and colons) - used when a bar is left in the scroll-back."""
        previous, self._finished = self._finished, True
        try:
            return self._render()
        finally:
            self._finished = previous

    # ------------------------------------------------------------------ rendering
    # Formatting helpers for the counter and the rate shown on the right of the bar.
    def _fmt_count(self, value):
        if value is None:
            return "?"
        if self.unit_scale:
            return tqdm.format_sizeof(value, divisor=self.unit_divisor)
        return str(int(value)) if float(value).is_integer() else f"{value:.2f}"

    def _fmt_rate(self, n, elapsed):
        if n <= 0 or elapsed <= 0:
            return "?" + (self.unit or "it") + "/s"
        rate = n / elapsed
        unit = self.unit or "it"
        if self.unit_scale:
            return tqdm.format_sizeof(rate, divisor=self.unit_divisor) + unit + "/s"
        return f"{rate:.2f}{unit}/s" if rate >= 1 else f"{1.0 / rate:.2f}s/{unit}"

    @staticmethod
    # mm:ss text of a time span; when `on` is False the colons become spaces (this makes the colons blink).
    def _clock(seconds, on):
        text = tqdm.format_interval(max(0, int(seconds)))
        return text if on else text.replace(":", " ")      # blinking colons, like a digital clock

    # Wraps text in an ANSI colour code (no-op when colours are disabled).
    def _paint(self, text, code):
        return f"{code}{text}{_RESET}" if self._color and text else text

    # Draws the bar: a sweeping block while the progress is unknown, otherwise a green bar growing in 1/8 steps.
    def _draw_bar(self, width, fraction, indeterminate, now):
        full_ch, empty_ch = ("█", "░") if self._unicode else ("#", "-")
        if indeterminate:
            block = max(4, width // 6)
            span = max(1, width - block)
            phase = (now * 12.0) % (2 * span)
            pos = int(phase if phase <= span else 2 * span - phase)
            return (self._paint(empty_ch * pos, _GREY) + self._paint(full_ch * block, _GREEN)
                    + self._paint(empty_ch * (width - block - pos), _GREY))
        cells = max(0.0, min(1.0, fraction)) * width
        full = int(cells)
        partial = _EIGHTHS[int((cells - full) * 8)] if (self._unicode and full < width) else ""
        bar = full_ch * full + (partial if partial.strip() else "")
        rest = width - full - (1 if partial.strip() else 0)
        return self._paint(bar, _GREEN) + self._paint(empty_ch * max(0, rest), _GREY)

    # Builds the whole line: '<dot> description  pct%|bar| count [elapsed<remaining, rate]'.
    def _render(self):
        if not hasattr(self, "_tty"):                      # first frame, drawn from tqdm.__init__
            self._tty, self._color, self._unicode = False, False, True
        d = self.format_dict
        now = time.time()
        n, total, elapsed = d["n"], d["total"], d["elapsed"]
        indeterminate = (not total) or n <= 0
        done = self._finished
        on = True if (done or not self._blink) else int(now) % 2 == 0

        if indeterminate:
            remaining_txt = "--:--"
            rate_txt = "calculating"
        else:
            if done or n >= total:
                remaining = 0
            elif self._eta_ref is not None:
                remaining = max(0.0, self._eta_ref - (now - self._eta_ref_t))   # counts down like a clock
            else:
                remaining = (total - n) * elapsed / n
            remaining_txt = self._clock(remaining, on)
            rate_txt = self._fmt_rate(n, elapsed)

        desc = (self.desc or "").rstrip().rstrip(":").rstrip()   # tqdm appends ": " itself when the description is updated
        desc = f"{desc}: " if desc else ""
        pct = f"{100.0 * n / total:3.0f}%" if total else "  ?%"
        dot_on, dot_off = ("●", "○") if self._unicode else ("*", " ")
        dot = dot_on if on else dot_off
        count = f"{self._fmt_count(n)}/{self._fmt_count(total)}"
        postfix = f", {d['postfix']}" if d.get("postfix") else ""

        left_plain = f"{dot} {desc}{pct}|"
        right_plain_fn = lambda p: f"| {count} [{self._clock(elapsed, on)}<{remaining_txt}, {rate_txt}{p}]"
        columns = shutil.get_terminal_size((100, 24)).columns
        room = columns - 1 - len(left_plain) - len(right_plain_fn(postfix))
        if room < 8 and postfix:                           # drop the postfix before squeezing the bar too much
            postfix = ""
            room = columns - 1 - len(left_plain) - len(right_plain_fn(postfix))
        width = max(8, min(40, room))

        bar = self._draw_bar(width, (n / total) if total else 0.0, indeterminate, now)
        right = f"| {count} [{self._clock(elapsed, on)}<{remaining_txt}, {rate_txt}{postfix}]"
        left = f"{self._paint(dot, _BRIGHT_GREEN)} {desc}{pct}|"
        return left + bar + right


# ---------------------------------------------------------------------- several processes, one screen
# Bars of different processes cannot share a terminal safely (a print in one process lands on the bar line of
# another). So the PARENT owns the screen: every child calls attach_remote() and its progress() calls become
# messages; its printed text is forwarded too and shown above the bars with tqdm.write. See train_all.py.
# Message protocol between a child process and the ProgressHub in the parent (tuples put on a multiprocessing.Queue):
# ('new', slot, kwargs) create a bar | ('update', slot, n) advance | ('desc', slot, text) set description
# ('postfix', slot, dict) set the postfix | ('close', slot, None) finish | ('write', slot, line) print a text line
# ('stop', None, None) end the hub thread.
_REMOTE = None


# Used in train_all.py: after this call every progress() in the process reports to the parent instead of drawing itself.
def attach_remote(queue, slot):
    """Call once in a child process: from now on progress() reports to the parent's ProgressHub (bar line = slot)."""
    global _REMOTE
    _REMOTE = (queue, slot)


# Same interface as ClockBar; updates are batched (at most one message per 50 ms) to keep the queue small.
class RemoteBar:
    """Stand-in for ClockBar inside a child process: forwards bar updates to the parent."""

    def __init__(self, iterable=None, **kwargs):
        self._queue, self._slot = _REMOTE
        self._iterable = iterable
        if kwargs.get("total") is None and hasattr(iterable, "__len__"):
            kwargs["total"] = len(iterable)
        kwargs.pop("position", None)
        self._pending, self._last_send = 0, 0.0
        self._queue.put(("new", self._slot, kwargs))

    def _flush(self, force=False):
        now = time.time()
        if self._pending and (force or now - self._last_send >= 0.05):
            self._queue.put(("update", self._slot, self._pending))
            self._pending, self._last_send = 0, now

    def update(self, n=1):
        self._pending += n
        self._flush()

    def set_description(self, desc=None, refresh=True):
        self._flush(force=True)
        self._queue.put(("desc", self._slot, desc))

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        self._flush(force=True)
        self._queue.put(("postfix", self._slot, dict(ordered_dict or {}, **kwargs)))

    def close(self):
        self._flush(force=True)
        self._queue.put(("close", self._slot, None))

    def __iter__(self):
        try:
            for item in self._iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# Replacement for sys.stdout / sys.stderr in a child process: complete lines are sent to the parent.
class QueueWriter:
    """File-like object that forwards complete lines to the parent (replaces sys.stdout / sys.stderr in a child)."""

    def __init__(self, queue, slot):
        self._queue, self._slot, self._buffer = queue, slot, ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._queue.put(("write", self._slot, line.rstrip("\r")))
        return len(text)

    def flush(self):
        if self._buffer:
            self._queue.put(("write", self._slot, self._buffer))
            self._buffer = ""

    def isatty(self):
        return False


# One hub runs in the parent process: it creates a ClockBar per child (position = slot) and prints their text above the bars.
class ProgressHub:
    """Runs in the parent: draws one ClockBar per child (position = slot) and prints the children's text above them."""

    def __init__(self, queue):
        self._queue = queue
        self._bars = {}
        self._thread = threading.Thread(target=self._run, name="progress-hub", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def write(self, text):
        tqdm.write(text)

    def stop(self):
        self._queue.put(("stop", None, None))
        self._thread.join()

    def _run(self):
        while True:
            kind, slot, payload = self._queue.get()
            if kind == "stop":
                break
            bar = self._bars.get(slot)
            if kind == "new":
                payload["leave"] = False                       # finished bars are re-printed in the scroll-back instead
                self._bars[slot] = ClockBar(position=slot, **payload)
            elif kind == "update" and bar is not None:
                bar.update(payload)
            elif kind == "desc" and bar is not None:
                bar.set_description(payload, refresh=False)
            elif kind == "postfix" and bar is not None:
                bar.set_postfix(payload, refresh=False)
            elif kind == "close" and bar is not None:
                final, stream = bar.final_text(), bar.fp
                bar.close()
                stream.write("\r")                              # tqdm leaves the cursor mid-line after clearing a lower row
                stream.flush()
                tqdm.write(final)
                del self._bars[slot]
            elif kind == "write":
                tqdm.write(payload)


# Public entry point. Returns a ClockBar, or a RemoteBar when attach_remote() was called in this process.
def progress(iterable=None, **kwargs):
    """Create a ClockBar. Accepts the same arguments as tqdm (desc, total, unit, unit_scale, leave, ...)."""
    if _REMOTE is not None:
        return RemoteBar(iterable, **kwargs)
    return ClockBar(iterable, **kwargs)
