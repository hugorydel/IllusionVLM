"""
Figure 2 - Illusion magnitude against signed illusion strength.

Each panel shows, for one illusion, how far the illusion displaces the point of
subjective equality as it strengthens in either of its two spatial directions.

WHERE THE ESTIMATES COME FROM
    by_magnitude.csv, the joint two-direction fit: at each |strength| the 16
    response cells (8 physical differences x 2 illusion directions) are fitted
    as one model in which the two directions share a slope and a lapse pair but
    each takes its own point of subjective equality,

        PSE(d) = side_bias - d * magnitude

    Only `magnitude` is plotted. `side_bias` is a standing preference for one
    response option, present whichever way the surround pushes, and carrying it
    into the figure would shift an observer's whole curve by a constant - on
    Muller-Lyer by +0.151 for the model against -0.017 for humans, enough to
    make comparable illusions look three times apart.

    Intervals are the profile interval on magnitude, signed by direction.

X AXIS
    Signed illusion strength normalised to each illusion's own tested maximum,
    spanning -1 to +1 with 0 meaning no illusion. The sign is the illusion's
    SPATIAL DIRECTION - which way the surround displaces the percept - not
    congruency. A direction slice mixes congruent and incongruent trials by
    construction, which is what makes a PSE definable for it, so congruency
    cannot label this axis. Congruency is the axis of Figure 3.

    With the bias removed the two halves are exact mirrors, so the left half
    carries no information the right does not. It is kept because the mirror is
    the readable form of "this illusion has two directions and they behave
    alike", and because it matches Figure 3's axis.

Y AXIS
    The illusion effect divided by ONE number per illusion - the largest human
    effect for that illusion - with both species divided by the same one. The
    human curve therefore reaches exactly -1 and +1, and the model is read
    directly against it: 1.5 is half again the human effect, 0.5 is half of it.

Rod-Frame is held out; see conventions.FIGURES_EXCLUDED. Open markers with a
dotted join are levels whose fit is not identified by the data.

All explanatory text belongs in the caption, not in the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pipeline.paper.conventions import DISPLAY_NAMES, FIGURE_ILLUSIONS
from pipeline.paper.figstyle import (
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    SPECIES_LABEL,
    SPECIES_MARKER,
    SURFACE,
    apply_style,
    centred_panel_grid,
    hide_spines,
)

YLIM = (-2.15, 2.15)
N_COLS = 3


def signed_magnitude(magnitudes: pd.DataFrame) -> pd.DataFrame:
    """
    The illusion effect alone, signed by direction, with side bias removed.

    The joint fit gives PSE(d) = side_bias - d * magnitude. Only the second
    term is the illusion; `side_bias` is a standing preference for one response
    option that is there whichever way the surround pushes. Plotting the PSE
    therefore adds a constant offset to an observer's whole curve - on
    Muller-Lyer that offset is +0.151 for the model against -0.017 for humans,
    enough to make comparable illusions look three times apart. So the effect
    is taken on its own and the bias is left out of this figure.

    Returns one row per (species, illusion, |strength|, direction).
    """
    rows = []
    for r in magnitudes.itertuples():
        for direction in (-1, 1):
            edge_a = -direction * r.magnitude_ci_low
            edge_b = -direction * r.magnitude_ci_high
            rows.append(
                {
                    "species": r.species,
                    "illusion": r.illusion,
                    "k_abs": r.k_abs,
                    "direction": direction,
                    "effect": -direction * r.illusion_magnitude,
                    "lo": min(edge_a, edge_b),
                    "hi": max(edge_a, edge_b),
                    "weak": bool(r.unconstrained),
                }
            )
    return pd.DataFrame(rows)


def human_reference(effects: pd.DataFrame, illusion: str) -> float:
    """
    The largest human illusion effect for this illusion.

    One denominator for both species in the panel, so the two curves stay on
    one scale and the comparison is not rescaled per species. Because the bias
    is already removed, the human curve reaches exactly -1 and +1.
    """
    sub = effects[(effects["illusion"] == illusion) & (effects["species"] == "human")]
    if sub.empty:
        return float("nan")
    denom = float(sub["effect"].abs().max())
    return denom if np.isfinite(denom) and denom > 0 else float("nan")


def _draw_panel(ax, sub: pd.DataFrame, denom: float, k_max: float) -> None:
    """Both species' signed illusion effect for one illusion."""
    for species in ("human", "vlm"):
        g = sub[sub["species"] == species].copy()
        if g.empty:
            continue

        # Polarity oriented so a positive strength displaces the PSE upward.
        g["x"] = -g["direction"] * g["k_abs"] / k_max
        g = g.sort_values("x")

        x = g["x"].to_numpy()
        y = (g["effect"] / denom).clip(*YLIM).to_numpy()
        lo = (g["lo"] / denom).clip(lower=YLIM[0]).to_numpy()
        hi = (g["hi"] / denom).clip(upper=YLIM[1]).to_numpy()
        weak = g["weak"].to_numpy(dtype=bool)

        # Retained so the cue still appears if a future dataset has an
        # unidentified level; nothing in the current figure triggers it.
        for a, b in zip(range(len(x) - 1), range(1, len(x))):
            ax.plot(
                x[[a, b]],
                y[[a, b]],
                color=SERIES[species],
                lw=1.0,
                ls=":" if (weak[a] or weak[b]) else "-",
                zorder=3,
            )

        for mask, facecolor in ((~weak, SERIES[species]), (weak, SURFACE)):
            if not mask.any():
                continue
            ax.errorbar(
                x[mask],
                y[mask],
                yerr=[
                    np.clip(y[mask] - lo[mask], 0, None),
                    np.clip(hi[mask] - y[mask], 0, None),
                ],
                fmt=SPECIES_MARKER[species],
                ms=2.8,
                mfc=facecolor,
                mec=SERIES[species],
                mew=0.7,
                ecolor=SERIES[species],
                linestyle="none",
                elinewidth=0.5,
                capsize=1.0,
                zorder=4,
            )


def build(paper_dir: Path, out_path: Path) -> None:
    """Render Figure 2 to `out_path`."""
    apply_style(base_font=7.5)

    magnitudes = pd.read_csv(paper_dir / "by_magnitude.csv")
    magnitudes = magnitudes[
        magnitudes["illusion"].isin(FIGURE_ILLUSIONS) & magnitudes["reportable"]
    ]
    effects = signed_magnitude(magnitudes)

    fig = plt.figure(figsize=(6.0, 3.9))
    axes = centred_panel_grid(
        fig,
        len(FIGURE_ILLUSIONS),
        N_COLS,
        left=0.112,
        right=0.985,
        top=0.882,
        bottom=0.188,
        hspace=0.62,
        wspace=0.42,
    )

    for i, illusion in enumerate(FIGURE_ILLUSIONS):
        ax = axes[i]
        hide_spines(ax)

        sub = effects[effects["illusion"] == illusion]
        denom = human_reference(effects, illusion)
        k_max = float(sub["k_abs"].max()) if not sub.empty else 1.0

        if np.isfinite(denom):
            _draw_panel(ax, sub, denom, k_max)

        ax.set_title(DISPLAY_NAMES[illusion], fontsize=7.4, color=INK_PRIMARY, pad=4)
        ax.set_ylim(*YLIM)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_xlim(-1.12, 1.12)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["-1", "0", "1"])
        if i % N_COLS == 0:
            ax.set_ylabel("Illusion Magnitude", fontsize=7.2)
        else:
            ax.set_yticklabels([])
        # Every panel carries its own x label: the centred short row means the
        # panels do not line up in columns, so nothing can be inherited.
        ax.set_xlabel("Illusion Strength", fontsize=7.2)

        ax.text(
            -0.11 if i % N_COLS else -0.34,
            1.18,
            "ABCDEF"[i],
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            color=INK_PRIMARY,
            va="top",
            ha="left",
        )

    handles = [
        Line2D(
            [],
            [],
            color=SERIES[sp],
            lw=1.0,
            marker=SPECIES_MARKER[sp],
            ms=2.8,
            label=SPECIES_LABEL[sp],
        )
        for sp in ("human", "vlm")
    ]
    if effects["weak"].any():
        handles.append(
            Line2D(
                [],
                [],
                color=INK_SECONDARY,
                lw=1.0,
                ls=":",
                marker="o",
                ms=2.8,
                mfc=SURFACE,
                mec=INK_SECONDARY,
                mew=0.7,
                label="weakly identified fit",
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(handles),
        fontsize=7.4,
        handlelength=1.8,
        columnspacing=2.4,
        handletextpad=0.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    build(Path("results/_paper"), Path("results/_paper/figures/fig2_magnitude.png"))
