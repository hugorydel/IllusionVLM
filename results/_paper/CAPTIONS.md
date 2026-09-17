# Figure captions and key numbers

Draft captions for the five main figures, plus the numbers each one rests on.
Regenerate everything with:

```bash
python -m pipeline.paper.run_figures --human <dir-with-study2-and-study3-csvs>
```

Human data: Makowski et al. (2023), *The Illusion Game*, Study 2 —
`RealityBending/IllusionGameValidation`, `data/study2_part{1,2}.csv`
(N = 256, 326,960 trials) and `data/study3.csv` (N = 250 participant-level
scores). Model data: GPT-5.2, 100–110 sampled response sets per illusion,
temperature 0.3.

All species comparisons use the **shared stimulus grid**: the 15 illusion
strengths and 8 absolute difference levels both datasets have in common
(128 cells per illusion). Our grid extends beyond the human one for
Müller-Lyer, Vertical-Horizontal, Ponzo and Rod-Frame; those cells are
retained in `cells.csv` with `shared = False` and excluded from every
cross-species contrast.

---

## Figure 1 — The paradigm

> **Figure 1. Forced-choice psychophysics applied to a vision-language model.**
> (**A**) For each of seven illusions, the same physical difference Δ rendered
> at the congruent extreme (k = −7), with no illusion (k = 0), and at the
> incongruent extreme (k = +7). The correct answer is identical across all
> three images in a row; only the surrounding context changes. (**B**) The
> verbatim forced-choice question put to the model, its two permitted
> responses, and the Δ held fixed along that row. Illusion strength is
> reported throughout as the normalised index k ∈ [−7, +7], because the raw
> strength unit differs per illusion (given in Figure 2).

Note for the text: the sign of `illusion_strength` encodes **congruency
relative to the true difference**, not a fixed spatial direction. Congruency
is determined by `sign(k)` alone and is independent of `sign(Δ)` — verified in
the human data, where `P(Incongruent | k > 0)` is exactly 0 or 1 for every
illusion.

---

## Figure 2 — The matrix figure

> **Figure 2. Illusion susceptibility in GPT-5.2 and in humans, across seven
> illusions.** Rows are illusions, ordered by effect clarity; the raw strength
> unit for each appears in its row label. (**A**, **B**) Response surfaces:
> P(positive option) over illusion strength k and signed difficulty, for the
> model and for humans on identical cells. |Δ| is pooled into four
> equal-count difficulty bands by rank, which fills every cell in both species
> despite the interleaved human design (see note below). Both species show the
> same V-shaped response around Δ = 0 at strong incongruent strengths, which
> establishes it as a property of the design rather than a model failure.
> (**C**) Sensitivity, as d′ relative to each observer's own zero-strength
> baseline. Values below 0 mean responses ran *against* the physical
> difference. (**D**) Bias, as the criterion shift c − c₀. A purely masking
> illusion leaves this at zero however large the sensitivity loss; a
> directional one moves it away from zero. Scales in C and D are shared across
> rows.

**The human design is interleaved.** Each non-zero strength samples 8 of the
16 signed difference levels (256 trials per cell, one per participant), with
all 16 at k = 0, and adjacent strengths cover complementary levels. Pooling
|Δ| in rank-pairs means every human row contributes exactly one level to each
difficulty band on each side of zero.

**Baseline competence** (d′ at k = 0), which gates interpretation of every
other measure:

| Illusion | Human d′₀ | GPT-5.2 d′₀ | Human c₀ | GPT-5.2 c₀ |
|---|---|---|---|---|
| Müller-Lyer | 2.99 | 3.48 | −0.07 | **+0.76** |
| Vertical-Horizontal | 2.47 | 4.72 | −0.18 | −0.17 |
| Ponzo | 2.76 | 4.81 | +0.11 | **−0.82** |
| Ebbinghaus | 2.75 | 3.56 | +0.02 | **−1.45** |
| Delboeuf | 2.82 | 4.44 | −0.01 | **−1.01** |
| Contrast | 4.02 | 5.20 | +0.01 | −0.30 |
| Rod-Frame | 1.87 | **0.51** | +0.02 | +0.20 |

Two things to state plainly in the text: the model is *more* sensitive than
humans at baseline on six of seven illusions (it was untimed, humans were
not), and it carries large constant side biases where humans carry none. On
Rod-Frame the model's baseline d′ of 0.51 is near chance, so its
sensitivity ratio is undefined and is left blank in panel C.

---

## Figure 3 — Congruency effects, head to head

> **Figure 3. Both species show a graded congruency effect, with aligned
> profiles.** (**A**) Error rate against signed illusion strength for each
> illusion, on the shared grid. Congruent strengths lie left of zero,
> incongruent right; the dashed line marks chance. (**B**) Each illusion's
> mean incongruent error rate in the model against that in humans, with the
> identity line. Labels abbreviate the illusion names of panel A. Pearson
> r = 0.68 (p = 0.094), Spearman ρ = 0.64 (p = 0.119), n = 7 illusions.
>
> Humans responded under time pressure (RT 500–1300 ms) and the model was
> untimed, so the absolute level of the curves is not comparable between
> species; their shape and their ordering across illusions are.

The congruent < baseline < incongruent ordering holds in both species for all
seven illusions, with two model exceptions where baseline competence is itself
poor (Ebbinghaus, Rod-Frame). The profile correlation is positive but **not
significant at n = 7** — report it as suggestive, not established.

| Illusion | Human cong / incong | GPT-5.2 cong / incong |
|---|---|---|
| Müller-Lyer | 0.029 / 0.625 | 0.030 / 0.455 |
| Vertical-Horizontal | 0.043 / 0.532 | 0.042 / 0.600 |
| Ponzo | 0.036 / 0.374 | 0.074 / 0.158 |
| Ebbinghaus | 0.044 / 0.350 | 0.093 / 0.217 |
| Delboeuf | 0.086 / 0.327 | 0.007 / 0.351 |
| Contrast | 0.044 / 0.397 | 0.016 / 0.324 |
| Rod-Frame | 0.104 / 0.454 | 0.265 / 0.534 |

---

## Figure 4 — The asymmetry dissociation

> **Figure 4. The congruency effect is symmetric in humans and directional in
> GPT-5.2.** (**A**) Mean |error asymmetry| over incongruent strengths, where
> asymmetry is the difference in error rate between trials whose correct answer
> is the positive and the negative option. A purely masking illusion gives
> zero. (**B**) |Baseline criterion| with the illusion switched off — the
> constant side bias each observer brings to the task. (**C**) The model's
> asymmetry against its own constant side bias. The two are unrelated
> (r = −0.31, p = 0.50, n = 7), so the asymmetry is illusion-specific rather
> than a constant preference showing through.

This is the strongest human–model contrast in the data. Human |asymmetry| is
≤ 0.075 on every illusion (mean 0.043); the model reaches 0.54.

| Illusion | Human | GPT-5.2 |
|---|---|---|
| Müller-Lyer | 0.072 | **0.541** |
| Vertical-Horizontal | 0.018 | **0.461** |
| Rod-Frame | 0.026 | **0.375** |
| Delboeuf | 0.025 | 0.254 |
| Ebbinghaus | 0.035 | 0.249 |
| Contrast | 0.075 | 0.185 |
| Ponzo | 0.048 | 0.144 |

**Limitation to state in the text.** The response options are always named in
the same order in the prompt, so a preference for the first- or last-named
option cannot be distinguished from a preference for that side. Panel C rules
out the *constant* side bias as the explanation but not option ordering. The
control is a re-run with the option order reversed; it has not been done. The
direction is not consistent across illusions (Müller-Lyer favours the
second-named option, Ebbinghaus the first), which argues against a pure
ordering effect but does not settle it.

---

## Figure 5 — Where physical evidence stops governing the response

> **Figure 5. The strength at which the illusion overrides the physical
> evidence.** (**A**) Every (illusion, strength) cell classified by the shape
> of its response function, using a model-free criterion: *monotonic* (rises
> with Δ and accurate at large |Δ|), *bias-dominated* (accurate at large |Δ|,
> errors concentrated at small |Δ|), *reversal* (P(positive) falls as Δ rises,
> so no monotonic function describes the data and any fitted PSE is an
> artifact), and *breakdown* (inaccurate even at the largest |Δ|). The first
> two admit a PSE; the last two do not. (**B**) Each row reduced to its
> crossover: the smallest incongruent k from which no higher k remains
> PSE-estimable. Markers are dodged vertically so coincident values stay
> visible; "never" marks an illusion that retains PSE-estimability throughout.

Crossover k, per species:

| Illusion | Human | GPT-5.2 |
|---|---|---|
| Müller-Lyer | 2 | 5 |
| Vertical-Horizontal | 3 | 1 |
| Ponzo | 5 | never |
| Ebbinghaus | 5 | never |
| Delboeuf | never | never |
| Contrast | 5 | 5 |
| Rod-Frame | 4 | 2 |

**Why this figure exists.** It is the disclosure that makes the PSE-based
framing of the earlier pilot analysis safe to abandon in print: a reviewer who
plots the raw cells will see the V-shape, and this figure says exactly where
it starts and that humans show it too.

---

## Method notes that belong in the manuscript

**Sign conventions.** Canonically, k > 0 is incongruent for every illusion.
Two corrections were needed, both established from the data and the rendered
stimuli rather than from library changelogs:

- **Delboeuf is inverted in our stimuli** and its strength axis is flipped at
  analysis time. Model accuracy was 0.812 at extreme negative strength versus
  0.992 at extreme positive — the opposite ordering to every other illusion
  and to the human data. Confirmed visually: at `str-2.17_diff+0.70` the
  surrounding ring sits on the smaller circle, opposing the correct answer.
- **Contrast needs no flip**, contrary to the note in `config.py` taken from
  Pyllusion's changelog. Human accuracy is 0.950 (negative) versus 0.420
  (positive) and the model's 0.958 versus 0.516 — both already canonical.
  Applying the documented flip would misalign the congruency axis.

Figure 1 independently confirms both: the congruent Delboeuf panel shows the
ring on the larger circle, and the congruent Contrast panel shows the lighter
patch on the darker surround.

**Psychometric fitting.** Where a PSE is reported it comes from a
four-parameter cumulative Gaussian (PSE, slope, two lapse rates) fitted by
binomial maximum likelihood, with bounds scaled to each illusion's own Δ grid
and 95% intervals from the profile likelihood. Median fit R² is 0.986 for
retained PSEs and 0.595 for rejected ones.

**Aggregation caveat.** The human curves are population averages over 256
participants with one trial per cell each, so they mix between- and
within-participant variability and their slopes are shallower than any
individual's. Bias comparisons are robust to this; slope and JND comparisons
are not, and should use the hierarchical per-participant scores in
`study3.csv` instead.

## Not available from the current data

- **Confidence dip at the point of maximum conflict.** The participant JSONL
  records only `response` and `correct`; no `top_logprobs`. Needs a re-run.
- **Reaction-time analogue.** Same reason.
- **Cross-model or scaling comparisons.** Only `gpt-5.2` in `results/`.
- **3D / naturalistic transfer.** No stimuli generated yet.
