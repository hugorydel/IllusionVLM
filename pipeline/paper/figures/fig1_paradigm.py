"""
Figure 1 - The paradigm.

One row per illusion SHOWN IN THE PAPER's figures, so that the paradigm figure
and the result figures cover the same illusions; see
conventions.FIGURES_EXCLUDED for what is held out and why. Each row is the same
physical difference rendered at the congruent extreme, with no illusion, and at
the incongruent extreme. The correct answer
is IDENTICAL across all three images in a row - only the surround changes - so
any change in responses along a row is caused by the context rather than by
the thing being judged.

The task column carries each illusion's question sentence VERBATIM from
config.ILLUSIONS, together with the correct answer, which the prompt does not
state. The question already names both response options.

Each prompt has two further sentences that are not shown: one naming the
objects to look at, and one fixing the answer format. Those belong in the
caption, which must quote them rather than paraphrase.

Exemplars are chosen by CANONICAL strength, so Delboeuf's congruent panel is
its raw positive strength (see conventions.FLIP_VLM). Picking by raw sign would
silently mislabel that row.

Delboeuf is drawn from the Pyllusion 1.2-equivalent stimuli, the ones the human
participants saw; conventions.stimulus_dir_name resolves that.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from config import ILLUSIONS
from pipeline.paper.conventions import (
    DISPLAY_NAMES,
    FIGURE_ILLUSIONS,
    FLIP_VLM,
    stimulus_dir_name,
)
from pipeline.paper.figstyle import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    apply_style,
)

STIM_PATTERN = re.compile(
    r"^(?P<name>\w+)_str(?P<str>[+-][\d.]+)_diff(?P<diff>[+-][\d.]+)\.png$"
)

def question_line(prompt: str) -> str:
    """
    The question sentence from a prompt in config.ILLUSIONS, verbatim.

    Every prompt is three blank-line-separated parts: an orienting sentence
    naming the objects, the question itself, and the answer-format instruction.
    The middle part is the task and already contains the two response options,
    so it stands alone.

    Taken from the prompt rather than restated, because a hand-kept paraphrase
    drifts from what was actually sent and the figure then documents something
    that never ran.
    """
    parts = [p.strip() for p in prompt.split("\n\n") if p.strip()]
    return parts[1] if len(parts) >= 2 else prompt.strip()

# Difference level to illustrate, as a fraction through each illusion's
# positive difference levels. Mid-range keeps the judgement visibly non-trivial
# without sitting at the edge of the grid.
DIFF_QUANTILE = 0.5


def _index_stimuli(stim_dir: Path, illusion: str) -> list[dict]:
    """Parse the filenames of one illusion's stimuli into records."""
    out = []
    for path in sorted((stim_dir / stimulus_dir_name(illusion)).glob("*.png")):
        m = STIM_PATTERN.match(path.name)
        if not m:
            continue
        out.append(
            {
                "path": path,
                "raw_strength": float(m.group("str")),
                "diff": float(m.group("diff")),
            }
        )
    return out


def _pick_exemplars(stim_dir: Path, illusion: str) -> tuple[list, float]:
    """
    Choose the (congruent, baseline, incongruent) images for one illusion.

    Selection is on canonical strength, so the Delboeuf flip is respected.
    """
    records = _index_stimuli(stim_dir, illusion)
    if not records:
        raise FileNotFoundError(
            f"No parsable stimuli in {stim_dir / stimulus_dir_name(illusion)}"
        )

    sign = -1.0 if illusion in FLIP_VLM else 1.0
    for r in records:
        r["k_sign"] = sign * r["raw_strength"]

    positives = sorted({r["diff"] for r in records if r["diff"] > 0})
    diff = positives[int(DIFF_QUANTILE * (len(positives) - 1))]
    at_diff = [r for r in records if abs(r["diff"] - diff) < 1e-9]

    congruent = min(at_diff, key=lambda r: r["k_sign"])
    incongruent = max(at_diff, key=lambda r: r["k_sign"])
    baseline = min(at_diff, key=lambda r: abs(r["k_sign"]))
    return [congruent, baseline, incongruent], diff


def build(stim_dir: Path, out_path: Path) -> None:
    """Render Figure 1 to `out_path`."""
    apply_style(base_font=7.5)

    options = {d["name"]: d["response_options"] for d in ILLUSIONS}
    questions = {d["name"]: question_line(d["prompt"]) for d in ILLUSIONS}

    n_rows = len(FIGURE_ILLUSIONS)
    fig = plt.figure(figsize=(7.2, 1.16 * n_rows + 0.34))
    gs = fig.add_gridspec(
        n_rows,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.62],
        hspace=0.12,
        wspace=0.05,
        left=0.135,
        right=0.995,
        top=0.962,
        bottom=0.012,
    )

    col_titles = (
        "Congruent  ($k=-7$)",
        "No Illusion  ($k=0$)",
        "Incongruent  ($k=+7$)",
    )

    for row, illusion in enumerate(FIGURE_ILLUSIONS):
        exemplars, _diff = _pick_exemplars(stim_dir, illusion)
        pos, neg = options[illusion]

        for col, rec in enumerate(exemplars):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(mpimg.imread(rec["path"]))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor("#dcdbd4")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=7.0, color=INK_SECONDARY, pad=4)

            if col == 0:
                # Name only. A rotated two-line label overflowed into the
                # neighbouring rows; the judged attribute goes in the task
                # column, where it reads horizontally.
                ax.set_ylabel(
                    DISPLAY_NAMES[illusion],
                    fontsize=7.4,
                    color=INK_PRIMARY,
                    labelpad=4,
                )

        # ---- Narrow annotation column -----------------------------------
        ax = fig.add_subplot(gs[row, 3])
        ax.axis("off")
        # The question verbatim, then the one fact the prompt does not state.
        # The question already names both response options, so listing them
        # again would only repeat the sentence above.
        question = textwrap.fill(questions[illusion], width=24)
        n_q = question.count("\n") + 1
        top = 0.88
        step = 0.115

        ax.text(
            0.5, top, question,
            fontsize=5.9, color=INK_PRIMARY, va="top", ha="center", linespacing=1.32,
        )
        y = top - n_q * step - 0.07
        for label, value, colour in (("correct", pos, INK_PRIMARY),):
            ax.text(
                0.47, y, label,
                fontsize=5.8, color=INK_MUTED, va="center", ha="right",
            )
            ax.text(
                0.53, y, value,
                fontsize=5.8, color=colour, va="center", ha="left",
            )
            y -= step
        if row == 0:
            ax.set_title("Task", fontsize=7.0, color=INK_SECONDARY, pad=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    build(Path("stimuli"), Path("results/_paper/figures/fig1_paradigm.png"))
