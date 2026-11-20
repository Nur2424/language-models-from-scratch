"""README Figure 1: the predeclared primary comparisons, paired by seed, and the sensitivity of
WaveNet+BN to the training protocol."""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figures import data, style as st

H = 4.45                       # figure height, inches
TOP_A = H - 0.72               # top of panel a's plot area
TOP_B = H - 1.38               # top of panel b's plot area (its title and note take three more lines)
BOTTOM = 1.32                  # bottom of both plot areas
# panel a
LABEL_RIGHT = 2.47             # right edge of the row labels
AX_A = (2.55, 3.55)            # left, width
VALUE_RIGHT = 6.52             # right edge of the value column
XLIM = (-0.042, 0.036)
# panel b
AX_B = (6.84, 1.3)
B_X = 6.74                     # left edge of panel b's title


def rows() -> list:
    w, t = data.wavenet_test_nll(), data.transformer_test_nll()
    return [
        ("header", "Window protocol", "32 eight-character windows per batch  ·  η = 0.03  ·  26 epochs"),
        ("row", "WaveNet − MLP", data.paired(w["C"], w["A"])),
        ("row", "WaveNet+BN − MLP+BN", data.paired(w["D"], w["B"])),
        ("divider",),
        ("header", "Name-batch protocol", "32 whole names per batch  ·  loss / C  ·  η = 0.1"),
        ("row", "Transformer (GPT-2) − RNN", data.paired(t["gpt2_style"], t["rnn"])),
        ("row", "Transformer (GPT-2) − WaveNet+BN", data.paired(t["gpt2_style"], t["wavenet_d"], level=0.90)),
        ("row", "Transformer (2017) − Transformer (GPT-2)", data.paired(t["transformer_2017"], t["gpt2_style"])),
    ]


def make() -> list:
    st.setup()
    fig = st.figure(H)
    ax = st.axes(fig, AX_A[0], BOTTOM, AX_A[1], TOP_A - BOTTOM)
    bx = st.axes(fig, AX_B[0], BOTTOM, AX_B[1], TOP_B - BOTTOM)

    # ---- layout of the rows (data y: 0 at the top, decreasing)
    spec = rows()
    y, placed = 0.0, []
    for item in spec:
        if item[0] == "header":
            y -= 0.75; placed.append((item, y)); y -= 0.85
        elif item[0] == "divider":
            y -= 0.05; placed.append((item, y)); y -= 0.15
        else:
            placed.append((item, y)); y -= 1.0
    ymin, ymax = y + 0.35, 0.0
    ax.set_ylim(ymin, ymax)

    def fig_y(v: float) -> float:          # data y -> inches from the figure bottom
        return BOTTOM + (v - ymin) / (ymax - ymin) * (TOP_A - BOTTOM)

    rng = np.random.default_rng(0)
    for item, yy in placed:
        if item[0] == "header":
            st.two_part(fig, st.MARGIN, fig_y(yy) - 0.04, item[1], item[2], strong_size=st.LABEL, rest_size=st.TICK)
        elif item[0] == "divider":
            fy = fig_y(yy) / H
            fig.add_artist(Line2D([st.MARGIN / st.WIDTH, VALUE_RIGHT / st.WIDTH], [fy, fy], color=st.TERTIARY,
                                  lw=0.6, ls=(0, (3, 3))))
        else:
            _, label, d = item
            lo, hi = d.interval
            ax.scatter(d.per_seed, yy + rng.uniform(-0.17, 0.17, d.per_seed.size), s=6, color=st.BLUE,
                       alpha=0.35, lw=0, zorder=2)
            ax.plot([lo, hi], [yy, yy], color=st.BLUE, lw=st.LW_INTERVAL, solid_capstyle="butt", zorder=3)
            if d.level == 0.90:
                for xe in (lo, hi):
                    ax.plot([xe, xe], [yy - 0.2, yy + 0.2], color=st.BLUE, lw=st.LW_INTERVAL - 0.2, zorder=3)
            ax.plot([d.mean], [yy], "o", ms=st.MARKER + 0.5, color=st.BLUE, zorder=4)
            st.text(fig, LABEL_RIGHT, fig_y(yy), label, fontsize=st.LABEL, color=st.INK, ha="right", va="center")
            st.text(fig, VALUE_RIGHT, fig_y(yy), st.signed(d.mean, 3), fontsize=st.NOTE, color=st.SECONDARY,
                    ha="right", va="center")

    ax.axvspan(-0.01, 0.01, color=st.BAND, lw=0, zorder=0)
    ax.axvline(0, color=st.SECONDARY, lw=st.LW_ZERO, zorder=1)
    ax.set_xlim(*XLIM)
    ax.set_xticks([-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
    ax.xaxis.set_major_formatter(st.fixed(2))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Paired difference in test NLL (nats per character), first minus second\n← first model better",
                  labelpad=5, linespacing=1.25)
    handles = [
        Line2D([], [], color=st.BLUE, lw=st.LW_INTERVAL, marker="o", ms=st.MARKER, label="primary (predeclared), 95% CI"),
        Line2D([], [], color=st.BLUE, lw=st.LW_INTERVAL, marker="|", ms=8, mew=1.4, label="equivalence test, 90% CI"),
        Line2D([], [], color=st.BLUE, lw=0, marker="o", ms=3, alpha=0.45, label="single seed (10 per row)"),
        Patch(color=st.BAND, label="±0.01: smallest effect of interest"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.31), ncol=2, borderaxespad=0,
              columnspacing=1.6, labelspacing=0.6, labelcolor=st.SECONDARY)

    # ---- panel a title and status line
    y0 = H - 0.3
    st.panel_title(fig, st.MARGIN, y0, "a", "The predeclared primary comparisons, paired by seed")
    st.status_line(fig, st.MARGIN + st.TITLE_INDENT, y0 - 0.25, "TEST SET",
                   "each row compares two models trained under the same protocol; "
                   "rows from different protocols are not comparable")

    # ---- panel b
    w, t = data.wavenet_test_nll(), data.transformer_test_nll()
    a = np.array([w["D"][s] for s in data.SEEDS]); b = np.array([t["wavenet_d"][s] for s in data.SEEDS])
    for s in data.SEEDS:
        bx.plot([0, 1], [a[s], b[s]], color=st.CHARCOAL, alpha=0.3, lw=0.9)
    bx.plot(np.zeros(10), a, "o", ms=3, color=st.CHARCOAL, alpha=0.55, mew=0)
    bx.plot(np.ones(10), b, "o", ms=3, color=st.CHARCOAL, alpha=0.55, mew=0)
    bx.plot([0, 1], [a.mean(), b.mean()], color=st.CHARCOAL, lw=2.2, marker="o", ms=st.MARKER + 0.5)
    shift = data.Paired(b - a)
    lo, hi = shift.interval
    top_y = max(a.max(), b.max()) + 0.0012
    bot_y = min(a.min(), b.min(), b.mean() - 0.01) - 0.0095
    bx.set_ylim(bot_y, top_y)
    # 0.01 scale bar, left of the points
    bar_top = a.mean() + 0.005
    bx.plot([-0.3, -0.3], [bar_top - 0.01, bar_top], color=st.TERTIARY, lw=2.2, alpha=0.6, solid_capstyle="butt")
    bx.text(-0.36, bar_top - 0.005, "0.01", ha="right", va="center", fontsize=st.NOTE, color=st.TERTIARY)
    bx.text(0.5, bot_y + 0.0007, f"shift {st.signed(shift.mean, 4)}\n95% CI\n[{st.signed(lo, 4)}, {st.signed(hi, 4)}]",
            ha="center", va="bottom", fontsize=st.NOTE, color=st.SECONDARY, linespacing=1.15)
    bx.set_xlim(-0.55, 1.25)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["window\nprotocol", "name-batch\nprotocol"], linespacing=1.15)
    bx.tick_params(axis="x", length=0, pad=5)
    bx.spines["left"].set_visible(False)
    bx.spines["right"].set_visible(True)
    bx.yaxis.tick_right(); bx.yaxis.set_label_position("right")
    bx.yaxis.set_major_formatter(st.fixed(3))
    bx.set_ylabel("Test NLL of WaveNet+BN", labelpad=6)
    last = st.panel_title(fig, B_X, y0, "b", "Sensitivity of WaveNet+BN\nto the training protocol")
    st.status_line(fig, B_X + st.TITLE_INDENT, last - 0.25, None,
                   "Seeds 0–9 matched by initial weights;\nseveral protocol components differ\n(see caption)")
    return st.save(fig, "readme_fig1")


if __name__ == "__main__":
    for p in make():
        print(p)
