"""Report figures (matplotlib, static PNG). Palette and mark specs follow one validated reference palette:
categorical slots in fixed order, 2px lines, hairline solid grid, text in ink colors (never the series color),
a legend whenever two or more series are drawn."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
GRAY = "#b9b8b2"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9.5, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "text.parse_math": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES)})


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def cumulative(series: dict[str, pd.Series], title: str, path: Path, highlight: list[str] | None = None,
               log: bool = True) -> None:
    """Growth of $1 (compounded) for a few named series; others (if any) in gray."""
    style()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    hl = highlight or list(series)
    for k, s in series.items():
        if k in hl:
            continue
        ax.plot(s.index, (1 + s.fillna(0)).cumprod(), color=GRAY, linewidth=1.0, alpha=0.8)
    for i, k in enumerate(hl):
        s = series[k]
        w = (1 + s.fillna(0)).cumprod()
        # final value in the legend: end labels collide when lines converge
        ax.plot(s.index, w, color=SERIES[i % len(SERIES)], label=f"{k} ({w.iloc[-1]:.1f}x)")
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_title(title)
    ax.legend(loc="upper left", ncols=min(4, len(hl)))
    save(fig, path)


def dot_ci(df: pd.DataFrame, value: str, lo: str, hi: str, title: str, xlabel: str, path: Path,
           ref: float | None = None, ref_label: str = "", groups: dict[str, str] | None = None) -> None:
    """One dot per row with its interval; rows top to bottom in the frame's order."""
    style()
    n = len(df)
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * n + 1.2))
    y = np.arange(n)[::-1]
    ax.hlines(y, df[lo], df[hi], color=SERIES[0], linewidth=2.0, alpha=0.55)
    ax.scatter(df[value], y, s=42, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=INK2, linewidth=1.0)
        ax.annotate(ref_label, (ref, n - 0.4), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_yticks(y, df.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    save(fig, path)


def small_multiples(frames: dict[str, pd.DataFrame], x: str, y: str, title: str, path: Path, ref: float | None = None,
                    xlabel: str = "", ylabel: str = "", logx: bool = False) -> None:
    style()
    k = len(frames)
    cols = min(3, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.5 * rows), sharey=True, squeeze=False)
    for ax, (name, f) in zip(axes.flat, frames.items()):
        ax.plot(f[x], f[y], color=SERIES[0], marker="o", markersize=4, markeredgecolor=SURFACE)
        if ref is not None:
            ax.axhline(ref, color=INK2, linewidth=1.0)
        ax.set_title(name, fontsize=10)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    for ax in axes.flat[k:]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def heatmap(M: pd.DataFrame, title: str, path: Path, fmt: str = "{:.2f}", center: float | None = None,
            cbar_label: str = "") -> None:
    style()
    fig, ax = plt.subplots(figsize=(0.62 * M.shape[1] + 2.2, 0.42 * M.shape[0] + 1.4))
    vals = M.to_numpy(float)
    if center is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])
        im = ax.imshow(vals, cmap=cmap, aspect="auto")
    else:
        lim = np.nanmax(np.abs(vals - center))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [DIVERGING[2], DIVERGING[1], DIVERGING[0]])
        im = ax.imshow(vals, cmap=cmap, vmin=center - lim, vmax=center + lim, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=45, ha="right")
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                rgb = im.cmap(im.norm(v))[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if lum < 0.5 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(title)
    save(fig, path)


def lines(df: pd.DataFrame, title: str, path: Path, ylabel: str = "", zero: bool = True) -> None:
    style()
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for i, c in enumerate(df.columns[:8]):
        ax.plot(df.index, df[c], color=SERIES[i], label=str(c))
    if zero:
        ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", ncols=4)
    save(fig, path)


def bars(s: pd.Series, title: str, path: Path, xlabel: str = "", ref: float | None = None) -> None:
    style()
    fig, ax = plt.subplots(figsize=(7.4, 0.3 * len(s) + 1.2))
    y = np.arange(len(s))[::-1]
    ax.barh(y, s.to_numpy(), height=0.55, color=SERIES[0])
    if ref is not None:
        ax.axvline(ref, color=INK2, linewidth=1.0)
    ax.set_yticks(y, s.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    save(fig, path)
