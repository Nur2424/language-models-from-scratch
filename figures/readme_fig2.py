"""README Figure 2: where the architectural advantages come from.
(a) WaveNet vs MLP by context frequency (WaveNet study, test set, exploratory);
(b) Transformer (GPT-2) vs WaveNet+BN by position (Transformer study, dev set, name-clustered)."""
from __future__ import annotations

import numpy as np

from figures import data, style as st

H = 4.7
TOP = H - 0.72                 # top of both plot areas (as in Figure 1, panel a)
BOTTOM = 0.98                  # bottom of both plot areas
AX_A = (0.98, 3.0)             # left, width
B_X = 4.5                      # left edge of panel b's title
AX_B = (5.32, 3.38)
STRIP_H, STRIP_GAP = 0.72, 0.1
Z = 1.959964                   # 95% normal quantile for the name-clustered intervals

CATEGORIES = (("unseen", "unseen\n(0)"), ("seen:low", "rare\n(1–17)"),
              ("seen:mid", "common\n(18–999)"), ("seen:high", "frequent\n(≥1000)"))
POSITIONS = tuple(str(i) for i in range(9)) + ("9+",)


def make() -> list:
    st.setup()
    fig = st.figure(H)
    ax = st.axes(fig, AX_A[0], BOTTOM, AX_A[1], TOP - BOTTOM)
    main_bottom = BOTTOM + STRIP_H + STRIP_GAP
    bx = st.axes(fig, AX_B[0], main_bottom, AX_B[1], TOP - main_bottom)
    cx = st.axes(fig, AX_B[0], BOTTOM, AX_B[1], STRIP_H, sharex=bx)

    # ---- panel a
    freq = data.wavenet_context_frequency()
    x = np.arange(len(CATEGORIES))
    for off, comp, colour, label, marker in ((-0.12, "C-A", st.BLUE, "WaveNet − MLP", "o"),
                                             (0.12, "D-B", st.GREEN, "WaveNet+BN − MLP+BN", "s")):
        r = [freq[(k, comp)] for k, _ in CATEGORIES]
        m = np.array([v["mean"] for v in r])
        err = [m - np.array([v["low"] for v in r]), np.array([v["high"] for v in r]) - m]
        ax.errorbar(x + off, m, yerr=err, fmt=marker, color=colour, ms=st.MARKER + (0.5 if marker == "o" else 0),
                    lw=st.LW_INTERVAL, capsize=0, label=label, mfc=colour if marker == "o" else "white", mew=1.4)
    ax.axhspan(-0.01, 0.01, color=st.BAND, lw=0, zorder=0)
    ax.axhline(0, color=st.SECONDARY, lw=st.LW_ZERO, zorder=1)
    for i, (k, _) in enumerate(CATEGORIES):
        ax.text(i, 0.0145, f"{freq[(k, 'C-A')]['targets']:,} targets", ha="center", va="center",
                fontsize=st.NOTE, color=st.TERTIARY)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CATEGORIES], linespacing=1.15)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(-0.5, len(CATEGORIES) - 0.5)
    ax.set_ylim(-0.05, 0.02)
    ax.set_yticks(np.arange(-0.05, 0.0201, 0.01))
    ax.yaxis.set_major_formatter(st.fixed(2))
    ax.set_xlabel("How often the 8-character context occurs in training", labelpad=6)
    ax.set_ylabel("Paired difference in TEST NLL (nats per character)\n← WaveNet better", linespacing=1.25,
                  labelpad=6)
    ax.legend(loc="lower right", labelcolor=st.SECONDARY, labelspacing=0.6)
    y0 = H - 0.3
    st.panel_title(fig, st.MARGIN, y0, "a", "WaveNet vs MLP, by context frequency")
    st.status_line(fig, st.MARGIN + st.TITLE_INDENT, y0 - 0.25, "TEST SET",
                   "WaveNet study  ·  exploratory  ·  95% CI over seeds")

    # ---- panel b
    d = data.position_by_index("gpt2_style - wavenet_d")
    xb = np.arange(len(POSITIONS))
    m = np.array([d[k]["mean"] for k in POSITIONS])
    se = np.array([d[k]["se_name_clustered"] for k in POSITIONS])
    n = [d[k]["targets"] for k in POSITIONS]
    for axis in (bx, cx):
        axis.axvspan(7.5, 8.5, color=st.SHADE_1, lw=0, zorder=0)
        axis.axvspan(8.5, 9.5, color=st.SHADE_2, lw=0, zorder=0)
    bx.errorbar(xb, m, yerr=Z * se, fmt="o", color=st.BLUE, ms=st.MARKER + 0.5, lw=st.LW_INTERVAL, capsize=0)
    bx.axhspan(-0.01, 0.01, color=st.BAND, lw=0, zorder=0.5)
    bx.axhline(0, color=st.SECONDARY, lw=st.LW_ZERO, zorder=1)
    bx.set_ylim(-0.075, 0.155)
    bx.set_yticks([-0.05, 0, 0.05, 0.10])
    bx.yaxis.set_major_formatter(st.fixed(2))
    bx.text(3.5, 0.150, "WaveNet+BN's window holds the whole name so far", ha="center", va="top",
            fontsize=st.NOTE, color=st.TERTIARY)
    for xi, label in ((8, "no\nstart\ntoken"), (9, "chars\nlost")):
        bx.text(xi, 0.150, label, ha="center", va="top", fontsize=st.NOTE, color=st.SECONDARY, linespacing=1.15)
    bx.set_ylabel("Paired difference in DEV NLL\n← Transformer better", linespacing=1.25, labelpad=6)
    bx.tick_params(axis="x", which="both", labelbottom=False, length=0)
    cx.bar(xb, n, color="#B5B5B5", width=0.7, zorder=2)
    for xi, k in zip(xb, n):
        cx.text(xi, k * 1.18, f"{k:,}", ha="center", va="bottom", fontsize=st.NOTE, color=st.TERTIARY)
    cx.set_yscale("log")
    cx.set_ylim(80, 40000)
    cx.set_yticks([100, 1000, 10000])
    cx.set_yticklabels(["100", "1k", "10k"])
    cx.minorticks_off()
    cx.set_ylabel("dev targets", labelpad=6)
    cx.set_xticks(xb)
    cx.set_xticklabels(POSITIONS)
    cx.tick_params(axis="x", length=0, pad=4)
    cx.set_xlim(-0.5, len(POSITIONS) - 0.5)
    cx.set_xlabel("Characters already read (i);  9+ pools positions 9–13", labelpad=6)
    st.panel_title(fig, B_X, y0, "b", "Transformer (GPT-2) vs WaveNet+BN, by position")
    st.status_line(fig, B_X + st.TITLE_INDENT, y0 - 0.25, "DEV SET",
                   "Transformer study  ·  secondary analysis  ·  95% CI clustered by name")
    return st.save(fig, "readme_fig2")


if __name__ == "__main__":
    for p in make():
        print(p)
