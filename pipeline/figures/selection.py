"""
pipeline/figures/selection.py - Which illusions the figures show, and their names.

The dataset (pipeline/human_comparison/) covers every illusion in
ILLUSION_ORDER. The figures draw a subset of it; the reasons for each exclusion
are recorded here, next to the list they shape.
"""

from __future__ import annotations

import pandas as pd

from pipeline.figures.figstyle import SPECIES_ORDER
from pipeline.human_comparison.conventions import ILLUSION_ORDER

# The subset the figures show. Two illusions are held out; both stay in the
# dataset and every analysis table.
#
# Rod-Frame fails on fit quality. The criterion is whether a psychometric
# function describes the model's responses at all, measured per direction slice
# on the shared grid as the number of slices whose response curve doubles back
# (drawdown > 0.35):
#
#     Delboeuf 0/14   Muller-Lyer 0/14   Ponzo 1/14
#     Ebbinghaus 2/14  Vertical-Horizontal 2/14   Contrast 4/14
#     Rod-Frame 12/14
#
# Rod-Frame is the only illusion in a different league there; the rest form a
# continuum. Humans are 0/14 with r-squared above 0.98 on all seven.
#
# Delboeuf fits cleanly but measures something else. The model reads a ring
# drawn close around a disc as the disc's own edge: where judging the disc and
# judging the ring's outer edge give different answers, it follows the outer
# edge 94% of the time at the tightest ring (1.14x the disc) and 15% at the
# loosest (1.78x), against about 10% for humans throughout. At the tightest
# ring it errs on every trial where the ring's edge exceeds the other disc and
# on none where it does not. Its Delboeuf curve therefore reflects a failure to
# separate a figure from its surround, not susceptibility to the illusion, and
# is not the same quantity as the other panels. The stimuli also differ from
# the human ones (see conventions.FLIP_VLM), which makes the failure much
# stronger.
FIGURES_EXCLUDED = {"RodFrame", "Delboeuf"}
FIGURE_ILLUSIONS = [i for i in ILLUSION_ORDER if i not in FIGURES_EXCLUDED]


def species_present(table: pd.DataFrame) -> list[str]:
    """
    The species in `table`, humans first, then models in config.MODELS order.

    Figures draw whichever models have data, so a model joins every figure as
    soon as its results are in the tables, and keeps its place and colour.
    """
    have = set(table["species"].unique())
    return [s for s in SPECIES_ORDER if s in have]


def models_present(table: pd.DataFrame) -> list[str]:
    """The models in `table`, in config.MODELS order."""
    return [s for s in species_present(table) if s != "human"]


# Display names for axis labels and captions.
DISPLAY_NAMES = {
    "MullerLyer": "Müller-Lyer",
    "VerticalHorizontal": "Vertical-Horizontal",
    "Ponzo": "Ponzo",
    "Ebbinghaus": "Ebbinghaus",
    "Delboeuf": "Delboeuf",
    "Contrast": "Contrast",
    "RodFrame": "Rod-Frame",
}
