"""A live search dashboard that runs in any terminal, with no dependencies.

:class:`TerminalDashboard` is a :class:`~jupyddl.trace.SearchObserver` that
repaints a small block of the terminal while the planner works — sparklines for
the heuristic and the ``f`` frontier, a frontier gauge, live counters and a node
rate. It is pure standard library (ANSI escapes and Unicode block characters),
so watching a search never costs you a dependency::

    from jupyddl import build_task, solve_task
    from jupyddl.live import TerminalDashboard

    task = build_task("domain.pddl", "problem.pddl")
    solve_task(task, "astar", "lmcut", observer=TerminalDashboard())

On a non-interactive stream it degrades to a periodic one-line progress report,
so it is also safe to use in CI logs and notebooks.
"""

from __future__ import annotations

import math
import shutil
import sys
import time

from .trace import SearchObserver

SPARKS = "▁▂▃▄▅▆▇█"
GAUGE_FULL = "█"
GAUGE_EMPTY = "░"

# 256-colour ANSI approximations of the jupyddl palette slots.
_ANSI = {
    "blue": "\x1b[38;5;33m",
    "orange": "\x1b[38;5;208m",
    "aqua": "\x1b[38;5;36m",
    "yellow": "\x1b[38;5;178m",
    "green": "\x1b[38;5;34m",
    "muted": "\x1b[38;5;245m",
    "bold": "\x1b[1m",
    "reset": "\x1b[0m",
}

__all__ = ["TerminalDashboard", "sparkline"]


def sparkline(values, width: int = 40) -> str:
    """Render ``values`` as a Unicode sparkline of at most ``width`` cells."""
    values = [v for v in values if v is not None and not math.isinf(v)]
    if not values:
        return ""
    if len(values) > width:  # keep the shape, drop the resolution
        bucket = len(values) / width
        values = [values[min(len(values) - 1, int(i * bucket))] for i in range(width)]
    low, high = min(values), max(values)
    if high == low:
        return SPARKS[3] * len(values)
    span = high - low
    return "".join(SPARKS[min(7, int((v - low) / span * 7.999))] for v in values)


def _gauge(fraction: float, width: int = 16) -> str:
    fraction = min(1.0, max(0.0, fraction))
    filled = int(round(fraction * width))
    return GAUGE_FULL * filled + GAUGE_EMPTY * (width - filled)


def _fmt(value) -> str:
    if value is None:
        return "–"
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1000:.1f}k"
    return f"{value:,}"


class TerminalDashboard(SearchObserver):
    """Repaint a live view of the search in the terminal.

    ``interval`` throttles repaints (seconds); ``history`` bounds the sparkline
    buffers so a multi-million-node search still costs constant memory.
    """

    def __init__(
        self,
        stream=None,
        interval: float = 0.08,
        history: int = 400,
        color: bool = True,
    ):
        self.stream = stream if stream is not None else sys.stderr
        self.interval = interval
        self.history = history
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.color = color and self.interactive

        self._h: list = []
        self._f: list = []
        self._open: list = []
        self._peak_open = 1
        self._best_h = math.inf
        self._start = time.perf_counter()
        self._last_paint = 0.0
        self._lines = 0
        self._title = "jupyddl"
        self._latest = None
        self._bounds = 0

    # ------------------------------------------------------------- painting
    def _c(self, key: str, text: str) -> str:
        if not self.color:
            return text
        return f"{_ANSI[key]}{text}{_ANSI['reset']}"

    def _width(self) -> int:
        try:
            return max(48, min(shutil.get_terminal_size().columns - 2, 100))
        except Exception:  # pragma: no cover - very unusual terminals
            return 72

    def _clear(self) -> None:
        if self._lines and self.interactive:
            self.stream.write(f"\x1b[{self._lines}A\x1b[0J")

    def _render(self, final: bool = False) -> None:
        width = self._width()
        spark_width = max(16, width - 30)
        event = self._latest
        expanded = event.expanded if event else 0
        generated = event.generated if event else 0
        evaluated = event.evaluated if event else 0
        elapsed = max(1e-9, time.perf_counter() - self._start)
        rate = expanded / elapsed

        current_h = self._h[-1] if self._h else None
        current_f = self._f[-1] if self._f else None
        open_now = self._open[-1] if self._open else 0
        best_h = None if math.isinf(self._best_h) else self._best_h

        lines = [
            self._c("bold", self._title),
            self._c("muted", "─" * width),
            "  h  "
            + self._c("aqua", sparkline(self._h, spark_width))
            + self._c("muted", f"   now {_num(current_h)}  best {_num(best_h)}"),
            "  f  "
            + self._c("blue", sparkline(self._f, spark_width))
            + self._c("muted", f"   now {_num(current_f)}"),
            "  frontier  "
            + self._c("orange", _gauge(open_now / max(1, self._peak_open), 18))
            + self._c("muted", f"  {_fmt(open_now)}  (peak {_fmt(self._peak_open)})"),
            "  "
            + self._c("bold", _fmt(expanded))
            + self._c("muted", " expanded   ")
            + self._c("bold", _fmt(generated))
            + self._c("muted", " generated   ")
            + self._c("bold", _fmt(evaluated))
            + self._c("muted", " evaluated"),
            self._c("muted", f"  {_fmt(rate)} nodes/s · {elapsed:.2f}s")
            + (self._c("muted", f" · {self._bounds} bounds") if self._bounds else ""),
        ]
        self._clear()
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self._lines = len(lines)

    def _maybe_paint(self) -> None:
        now = time.perf_counter()
        if now - self._last_paint < self.interval:
            return
        self._last_paint = now
        if self.interactive:
            self._render()
        else:
            # Non-tty: a single appended line, much less often.
            if now - self._start > 1 and int(now) % 2 == 0:
                event = self._latest
                if event is not None:
                    self.stream.write(
                        f"  ... {_fmt(event.expanded)} expanded, "
                        f"{_fmt(event.generated)} generated, "
                        f"{now - self._start:.1f}s\n"
                    )
                    self.stream.flush()

    # -------------------------------------------------------- observer hooks
    def on_start(self, task, planner: str, heuristic: str = "") -> None:
        self._start = time.perf_counter()
        label = f"{planner}/{heuristic}" if heuristic else planner
        name = getattr(task, "name", "") or "task"
        facts = getattr(task, "num_facts", 0)
        operators = len(getattr(task, "operators", ()))
        self._title = (
            f"jupyddl · {label} · {name}  "
            f"({facts} facts, {operators} ground actions)"
        )
        if self.interactive:
            self._render()

    def on_expand(
        self,
        state,
        g: float = 0.0,
        h: float = 0.0,
        f: float = 0.0,
        depth: int = 0,
        open_size: int = 0,
        stats=None,
        parent=None,
        action: str = "",
    ) -> None:
        self._latest = stats
        if h is not None and not math.isinf(h):
            self._h.append(h)
            self._best_h = min(self._best_h, h)
        self._f.append(f if f else g + (h or 0))
        self._open.append(open_size)
        self._peak_open = max(self._peak_open, open_size)
        for buffer in (self._h, self._f, self._open):
            if len(buffer) > self.history:
                del buffer[: len(buffer) - self.history]
        self._maybe_paint()

    def on_bound(self, threshold: float, iteration: int, stats=None) -> None:
        self._bounds += 1
        self._latest = stats or self._latest

    def on_finish(self, result) -> None:
        self._latest = getattr(result, "stats", None) or self._latest
        if self.interactive:
            self._render(final=True)
        elapsed = time.perf_counter() - self._start
        if result is not None and getattr(result, "solved", False):
            verdict = self._c(
                "green",
                f"  ✔ solved · cost {result.cost} · "
                f"{result.plan_length} actions · {elapsed:.2f}s",
            )
        else:
            verdict = self._c("orange", f"  ✘ no plan found · {elapsed:.2f}s")
        self.stream.write(verdict + "\n")
        self.stream.flush()
        self._lines = 0


def _num(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if math.isinf(value):
            return "∞"
        if value.is_integer():
            return str(int(value))
        return f"{value:.1f}"
    return str(value)
