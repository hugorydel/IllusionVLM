"""
pipeline/delboeuf_v12.py - Delboeuf stimuli as Pyllusion 1.2 drew them.

The human Illusion Game data (collected 2022-08-05 to 2022-08-27, Pyllusion
1.2) predates Pyllusion commit 4ee0c33c of 2022-09-05, which changed the base
size of the Delboeuf outer circle:

    - outer_size = inner_size + (0.2 * size_min)
    + outer_size = inner_size

Before the commit both target circles carried a visible outline and one was
then enlarged by sqrt(1 + |illusion_strength|). After it, the unenlarged
side's outline coincides with its target and is invisible, so only one circle
is ringed and at zero strength neither is.

Stimuli generated against the installed Pyllusion therefore do not match the
ones the human participants saw. This module restores the pre-commit base size
so a matched set can be generated, and registers the result on the pyllusion
module as `DelboeufV12` so module_1_generate can reach it by name through its
existing getattr lookup.

Only `_delboeuf_parameters` is wrapped. Ebbinghaus imports
`_delboeuf_parameters_sizeinner` and `_delboeuf_parameters_sizeouter`
directly and is untouched by this.
"""

from __future__ import annotations

import pyllusion
from pyllusion.Delboeuf import delboeuf_parameters as _dp
from pyllusion.Delboeuf.Delboeuf import Delboeuf as _Delboeuf

# The additive term the 2022-09-05 commit removed, as a multiple of size_min.
PRE_FIX_OUTER_BUMP = 0.2


def _delboeuf_parameters_v12(
    illusion_strength: float = 0,
    difference: float = 0,
    size_min: float = 0.25,
    distance: float = 1,
    distance_auto: bool = False,
) -> dict:
    """
    Pyllusion's Delboeuf parameters with the pre-2022-09-05 outer base size.

    The current implementation sets the outer circles equal to the inner ones
    before applying the illusion scaling. Re-deriving the outer sizes from the
    bumped base reproduces the earlier geometry without duplicating any of the
    surrounding layout logic.
    """
    params = _dp._delboeuf_parameters(
        illusion_strength=illusion_strength,
        difference=difference,
        size_min=size_min,
        distance=distance,
        distance_auto=distance_auto,
    )

    bump = PRE_FIX_OUTER_BUMP * size_min
    outer_left, outer_right = _dp._delboeuf_parameters_sizeouter(
        params["Size_Inner_Left"] + bump,
        params["Size_Inner_Right"] + bump,
        difference=difference,
        illusion_strength=illusion_strength,
    )

    params.update(
        {
            "Size_Outer_Left": outer_left,
            "Size_Outer_Right": outer_right,
            "Size_Outer_Smaller": min(outer_left, outer_right),
            "Size_Outer_Larger": max(outer_left, outer_right),
            "Pyllusion_Variant": "1.2-equivalent",
        }
    )
    return params


class DelboeufV12(_Delboeuf):
    """Delboeuf drawn with the Pyllusion 1.2 outer-circle base size."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = _delboeuf_parameters_v12(**kwargs)


# module_1_generate resolves the generator with getattr(pyllusion, name).
pyllusion.DelboeufV12 = DelboeufV12
