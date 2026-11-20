"""One typography and styling system for every figure in this folder.

Typeface: Source Sans 3 (Regular and Semibold), shipped in ``figures/fonts/`` under the SIL Open Font
License and loaded from there, so every machine renders the same glyphs.

Every figure is exactly ``WIDTH`` inches wide and is saved without automatic cropping, so on GitHub
(a README column is about 880 px wide) all figures display at the same scale: about 1.36 screen
pixels per point.

Type scale (four levels; Semibold only for titles and the few labels named below):
    TITLE  11.5 pt Semibold  panel letter and panel title
    LABEL   9.5 pt Regular   row labels, axis labels, protocol names (Semibold)
    TICK    8.5 pt Regular   tick labels, legends, status lines
    NOTE    8   pt Regular   annotations: values, counts, region labels, scale labels
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
OUT_DIR = HERE / "out"
FONT_FILES = ("SourceSans3-Regular.ttf", "SourceSans3-Semibold.ttf")
FAMILY = "Source Sans 3"

# canvas
WIDTH = 9.0          # inches, every figure
DPI = 220            # 1,980 px wide: sharp on high-resolution screens at GitHub's column width
MARGIN = 0.15        # inches, outer margin on every side

# type scale (points)
TITLE, LABEL, TICK, NOTE = 11.5, 9.5, 8.5, 8.0
SEMIBOLD = "semibold"

# colours for type and structure
INK = "#222222"
SECONDARY = "#555555"
TERTIARY = "#888888"
AXIS = "#666666"
BAND = (0, 0, 0, 0.065)     # the ±0.01 band (smallest effect of interest), translucent over shading
SHADE_1, SHADE_2 = "#EBEBEB", "#DCDCDC"   # graded background regions

# data colours (approved; colour-blind-safe Okabe–Ito blue and green)
BLUE = "#0072B2"
GREEN = "#009E73"
DESCRIPTIVE = "#888888"
CHARCOAL = "#3B3B3B"

# marks
LW_INTERVAL = 1.6
LW_ZERO = 0.8
LW_AXIS = 0.7
MARKER = 5.0

MINUS = "−"


def setup() -> None:
    """Register the bundled fonts and set the shared rcParams."""
    for name in FONT_FILES:
        font_manager.fontManager.addfont(str(FONT_DIR / name))
    plt.rcParams.update({
        "font.family": FAMILY,
        "font.size": TICK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.labelsize": LABEL,
        "axes.edgecolor": AXIS,
        "axes.linewidth": LW_AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
        "axes.formatter.use_mathtext": False,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS,
        "xtick.major.pad": 3, "ytick.major.pad": 3,
        "legend.fontsize": TICK, "legend.frameon": False,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.5,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "svg.hashsalt": "figures", "pdf.fonttype": 42,
    })


def figure(height: float) -> plt.Figure:
    return plt.figure(figsize=(WIDTH, height))


def axes(fig: plt.Figure, left: float, bottom: float, width: float, height: float, **kw) -> plt.Axes:
    """Axes placed in inches from the figure's bottom-left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes((left / W, bottom / H, width / W, height / H), **kw)


def text(fig: plt.Figure, x: float, y: float, s: str, **kw):
    """Text placed in inches from the figure's bottom-left corner."""
    W, H = fig.get_size_inches()
    return fig.text(x / W, y / H, s, **kw)


TITLE_INDENT = 0.24      # inches from the panel letter to its title
TITLE_LEADING = TITLE * 1.25 / 72
TICK_LEADING = TICK * 1.3 / 72


def panel_title(fig: plt.Figure, x: float, y: float, letter: str, title: str) -> float:
    """Panel letter at ``x`` and title at ``x + TITLE_INDENT``, first baseline at ``y`` (inches).
    A title may span several lines; returns the baseline of its last line."""
    text(fig, x, y, letter, fontsize=TITLE, fontweight=SEMIBOLD, color=INK, va="baseline")
    for i, line in enumerate(title.split("\n")):
        text(fig, x + TITLE_INDENT, y - i * TITLE_LEADING, line, fontsize=TITLE, fontweight=SEMIBOLD,
             color=INK, va="baseline")
    return y - (title.count("\n")) * TITLE_LEADING


def width_of(fig: plt.Figure, artist) -> float:
    """Rendered width of a text artist, in inches."""
    return artist.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi


def two_part(fig: plt.Figure, x: float, y: float, strong: str, rest: str, *, strong_size: float,
             rest_size: float, sep: str = "  ·  ") -> None:
    """``strong`` in Semibold ink, then ``rest`` in Regular secondary grey, on one baseline."""
    t = text(fig, x, y, strong, fontsize=strong_size, fontweight=SEMIBOLD, color=INK, va="baseline")
    if rest:
        text(fig, x + width_of(fig, t), y, sep + rest, fontsize=rest_size, color=SECONDARY, va="baseline")


def status_line(fig: plt.Figure, x: float, y: float, emphasis: str | None, rest: str) -> None:
    """Quiet lines under a panel title (``rest`` may span lines); only ``emphasis`` (e.g. "TEST SET")
    is Semibold."""
    lines = rest.split("\n")
    if emphasis:
        two_part(fig, x, y, emphasis, lines[0], strong_size=TICK, rest_size=TICK)
    else:
        text(fig, x, y, lines[0], fontsize=TICK, color=SECONDARY, va="baseline")
    for i, line in enumerate(lines[1:], 1):
        text(fig, x, y - i * TICK_LEADING, line, fontsize=TICK, color=SECONDARY, va="baseline")


def signed(value: float, decimals: int) -> str:
    """A number with an explicit sign, using the true minus sign."""
    s = f"{value:+.{decimals}f}"
    return s.replace("-", MINUS)


def fixed(decimals: int):
    """Tick formatter with a fixed number of decimals and the true minus sign."""
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: f"{v:.{decimals}f}".replace("-", MINUS) if abs(v) > 1e-12 else f"{0:.{decimals}f}")


def save(fig: plt.Figure, stem: str) -> list[Path]:
    """PNG for the README and PDF for print; fixed canvas, no automatic cropping, no timestamps."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png, pdf = OUT_DIR / f"{stem}.png", OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, metadata={"Software": None})
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return [png, pdf]
