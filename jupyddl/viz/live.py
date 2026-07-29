"""Live and animated matplotlib views of a search.

:class:`LiveSearchPlot` is an observer that repaints a figure *while* the planner
runs (use it from a notebook or any interactive backend). :func:`animate_search`
replays an already-recorded :class:`~jupyddl.trace.SearchTrace` into a GIF or MP4,
which is what the README animations are made of.
"""

from __future__ import annotations

import math

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from ..trace import SearchObserver
from .theme import palette, rc_params, series_color

__all__ = ["LiveSearchPlot", "animate_search"]


def _ffmpeg_available() -> bool:
    """True when matplotlib can find an ffmpeg binary (imageio-ffmpeg counts)."""
    if FFMpegWriter.isAvailable():
        return True
    try:  # imageio-ffmpeg ships a static binary; point matplotlib at it
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return FFMpegWriter.isAvailable()
    except Exception:
        return False


class _Panels:
    """The three live panels, shared by the interactive and the replay path."""

    def __init__(self, fig, dark: bool):
        pal = palette(dark)
        self.pal = pal
        self.dark = dark
        self.fig = fig
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        self.ax_cost = fig.add_subplot(gs[0, :])
        self.ax_open = fig.add_subplot(gs[1, 0])
        self.ax_rate = fig.add_subplot(gs[1, 1])

        (self.line_f,) = self.ax_cost.plot(
            [], [], color=series_color(0, dark), label="f = g + h", zorder=4
        )
        (self.line_g,) = self.ax_cost.plot(
            [], [], color=series_color(1, dark), label="g (path cost)", zorder=3
        )
        (self.line_h,) = self.ax_cost.plot(
            [], [], color=series_color(2, dark), label="h (estimate)", zorder=3
        )
        self.ax_cost.set_title("Cost estimates as the search advances")
        self.ax_cost.set_xlabel("nodes expanded")
        self.ax_cost.set_ylabel("cost")
        self.ax_cost.legend(loc="upper left", ncol=3)

        (self.line_open,) = self.ax_open.plot(
            [], [], color=series_color(0, dark), zorder=3
        )
        self.ax_open.set_title("Frontier size")
        self.ax_open.set_xlabel("nodes expanded")
        self.ax_open.set_ylabel("open list")

        (self.line_exp,) = self.ax_rate.plot(
            [], [], color=series_color(0, dark), label="expanded", zorder=4
        )
        (self.line_gen,) = self.ax_rate.plot(
            [], [], color=series_color(1, dark), label="generated", zorder=3
        )
        self.ax_rate.set_title("Nodes touched over time")
        self.ax_rate.set_xlabel("milliseconds")
        self.ax_rate.set_ylabel("cumulative nodes")
        self.ax_rate.legend(loc="upper left")

        self.caption = fig.text(
            0.008, 0.955, "", fontsize=9, color=pal["text_secondary"], ha="left"
        )
        self.title = fig.text(
            0.008,
            0.978,
            "jupyddl",
            fontsize=13,
            fontweight="bold",
            color=pal["text"],
            ha="left",
        )

    def artists(self):
        return (
            self.line_f,
            self.line_g,
            self.line_h,
            self.line_open,
            self.line_exp,
            self.line_gen,
            self.caption,
            self.title,
        )

    def update(self, steps, gs, hs, fs, opens, elapsed, generated, caption, title=None):
        self.line_f.set_data(steps, fs)
        self.line_g.set_data(steps, gs)
        self.line_h.set_data(steps, hs)
        self.line_open.set_data(steps, opens)
        self.line_exp.set_data(elapsed, steps)
        self.line_gen.set_data(elapsed, generated)
        self.caption.set_text(caption)
        if title:
            self.title.set_text(title)

        for ax, xs, ys in (
            (self.ax_cost, steps, list(fs) + list(gs) + list(hs)),
            (self.ax_open, steps, opens),
            (self.ax_rate, elapsed, list(steps) + list(generated)),
        ):
            if not xs or not ys:
                continue
            hi_x = max(xs) or 1
            hi_y = max(ys) or 1
            ax.set_xlim(0, hi_x * 1.03)
            ax.set_ylim(0, hi_y * 1.15)


class LiveSearchPlot(SearchObserver):
    """Repaint a matplotlib figure while the search runs.

    Best used from a notebook or with an interactive backend; ``every``
    throttles the redraw to one frame per N expansions.
    """

    def __init__(self, every: int = 25, dark: bool = False, figsize=(10, 6)):
        self.every = max(1, int(every))
        self.dark = dark
        self._rc = rc_params(dark)
        with plt.rc_context(self._rc):
            self.fig = plt.figure(figsize=figsize, layout="constrained")
            self.panels = _Panels(self.fig, dark)
        self._steps: list = []
        self._g: list = []
        self._h: list = []
        self._f: list = []
        self._open: list = []
        self._elapsed: list = []
        self._generated: list = []
        self._label = "jupyddl"

    def on_start(self, task, planner: str, heuristic: str = "") -> None:
        label = f"{planner}/{heuristic}" if heuristic else planner
        self._label = f"{label} · {getattr(task, 'name', '') or 'task'}"

    def on_expand(
        self,
        state,
        g=0.0,
        h=0.0,
        f=0.0,
        depth=0,
        open_size=0,
        stats=None,
        parent=None,
        action="",
    ) -> None:
        self._steps.append(len(self._steps) + 1)
        self._g.append(g)
        self._h.append(0.0 if h is None or math.isinf(h) else h)
        self._f.append(f if f else g + (h or 0))
        self._open.append(open_size)
        self._generated.append(stats.generated if stats else 0)
        self._elapsed.append(len(self._steps))
        if len(self._steps) % self.every == 0:
            self.refresh()

    def refresh(self) -> None:
        """Redraw now (called automatically every ``every`` expansions)."""
        with plt.rc_context(self._rc):
            self.panels.update(
                self._steps,
                self._g,
                self._h,
                self._f,
                self._open,
                self._elapsed,
                self._generated,
                f"{len(self._steps):,} expanded · "
                f"{self._generated[-1] if self._generated else 0:,} generated",
                self._label,
            )
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:  # pragma: no cover - headless backends
            pass

    def on_finish(self, result) -> None:
        self.refresh()

    def save(self, path: str) -> None:
        """Write the final state of the live figure to ``path``."""
        self.fig.savefig(path)


def animate_search(
    trace,
    path: str,
    dark: bool = False,
    fps: int = 30,
    seconds: float = 8.0,
    figsize=(10, 6),
    dpi: int = 110,
) -> str:
    """Replay a recorded trace into an animation at ``path`` (``.mp4`` or ``.gif``).

    The trace is resampled to ``fps * seconds`` frames, so a search of 20 nodes
    and one of 200,000 both produce a clip of the same length.
    """
    expansions = trace.expansions
    if not expansions:
        raise ValueError("trace has no expansions to animate")

    frames = max(2, int(fps * seconds))
    total = len(expansions)
    # index of the last expansion included in each frame
    cuts = [max(1, int(round((i + 1) / frames * total))) for i in range(frames)]

    gs = [e.g for e in expansions]
    hs = [0.0 if e.h is None or math.isinf(e.h) else e.h for e in expansions]
    fs = [e.f if e.f else e.g + (e.h or 0) for e in expansions]
    opens = [e.open_size for e in expansions]
    generated = [e.generated for e in expansions]
    elapsed = [e.elapsed * 1000 for e in expansions]
    steps = list(range(1, total + 1))

    label = f"{trace.label} · {trace.task_name or 'task'}"

    with plt.rc_context(rc_params(dark)):
        fig = plt.figure(figsize=figsize, layout="constrained")
        panels = _Panels(fig, dark)

        def draw(frame_index):
            cut = cuts[frame_index]
            caption = (
                f"{cut:,} expanded · {generated[cut - 1]:,} generated · "
                f"{elapsed[cut - 1]:.0f} ms"
            )
            if frame_index == frames - 1:
                caption += (
                    f"  ·  solved, cost {trace.cost}"
                    if trace.solved
                    else "  ·  no plan found"
                )
            panels.update(
                steps[:cut],
                gs[:cut],
                hs[:cut],
                fs[:cut],
                opens[:cut],
                elapsed[:cut],
                generated[:cut],
                caption,
                label,
            )
            return panels.artists()

        animation = FuncAnimation(fig, draw, frames=frames, blit=False)
        if path.lower().endswith(".gif"):
            writer: object = PillowWriter(fps=fps)
        elif _ffmpeg_available():
            writer = FFMpegWriter(fps=fps, bitrate=2400)
        else:
            raise RuntimeError(
                "writing MP4 needs ffmpeg (pip install imageio-ffmpeg), "
                "or use a .gif path instead"
            )
        animation.save(path, writer=writer, dpi=dpi)
        plt.close(fig)
    return path
