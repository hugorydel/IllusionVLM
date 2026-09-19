# Figure captions

Captions for the four main figures and two supplementary figures, and the
numbers each one rests on. Regenerate everything (Module 3's per-illusion
tables, then Module 4's comparison tables and figures) with:

```bash
python run_pipeline.py --modules 3 4
```

Module 4 alone is `python -m pipeline.module_4_figures`; add `--skip-dataset`
to re-render from the tables already in `results/_paper/`. The human data are
read from `data/human/`.

## Data

- **Humans.** Makowski et al. (2023), *The Illusion Game*, Study 2
  (`RealityBending/IllusionGameValidation`, `data/study2_part{1,2}.csv`),
  N = 256. Each participant saw each stimulus once, giving 247–256 trials per
  cell. Responses were speeded.
- **GPT-5.2.** 100 runs per illusion (110 for Müller-Lyer), each answering
  every stimulus once. Temperature 0.3, reasoning effort "none", untimed.
- **Shared grid.** Every comparison uses only the cells both datasets
  contain: 15 illusion strengths crossed with 16 signed physical differences,
  interleaved as in the human design. Each non-zero strength samples 8 of the
  16 differences and zero strength samples all 16, giving 128 cells per
  illusion. GPT-5.2 was tested on more: the full crossing of strengths and
  differences for every illusion (112 extra cells each), and larger
  differences for Müller-Lyer, Vertical-Horizontal and Ponzo (60, 180 and 240
  more). These are kept in `cells.csv` with `shared = False` and excluded
  from every figure; Figure 1's example images are also drawn from the
  shared cells only.
- **Illusions shown.** Müller-Lyer, Vertical-Horizontal, Ponzo, Ebbinghaus,
  Contrast. Rod-Frame and Delboeuf are in the dataset but not in the figures;
  see *Held out of the figures* below.

---

## Figure 1 — The paradigm

> **Figure 1. The paradigm.** Each row shows one illusion at a single
> physical difference, held fixed along the row. The correct answer is
> therefore the same in all three images; only the surrounding context
> changes. *Negative Illusion*: the strongest tested illusion in the congruent
> direction, where the context supports the correct answer. *No Illusion*:
> illusion strength zero. *Positive Illusion*: the strongest tested illusion
> in the incongruent direction, where the context opposes the correct answer.
> *Task*: the question put to GPT-5.2, verbatim, and the correct answer for
> that row. Each prompt also opened with a sentence naming the targets (e.g.
> "Look at the two red horizontal lines in this image.") and closed with an
> answer-format instruction (e.g. 'Answer with only "Top" or "Bottom".'); the
> full prompts are given in the Methods. Every image uses a combination of
> illusion strength and difference on which the human participants were also
> tested, in the Illusion Game (Makowski et al., 2023).

Full prompts, verbatim. The three parts are separated by blank lines in the
prompt; the figure shows the middle one.

| Illusion | Opening sentence | Question (shown in Figure 1) | Answer format |
|---|---|---|---|
| Müller-Lyer | Look at the two red horizontal lines in this image. | Which red line looks longer — the TOP one or the BOTTOM one? | Answer with only "Top" or "Bottom". |
| Vertical-Horizontal | Look at the two red lines in this image. | Which red line looks longer — the LEFT one or the RIGHT one? | Answer with only "Left" or "Right". |
| Ponzo | Look at the two red horizontal lines in this image. | Which red line looks longer — the TOP one or the BOTTOM one? | Answer with only "Top" or "Bottom". |
| Ebbinghaus | Look at the two red circles in the centre of each group in this image. | Which central red circle looks bigger — the LEFT one or the RIGHT one? | Answer with only "Left" or "Right". |
| Contrast | Look at the two small grey rectangles in this image. | Which grey rectangle looks lighter (brighter) — the TOP one or the BOTTOM one? | Answer with only "Top" or "Bottom". |

---

## Figure 2 — Perceptual shift

> **Figure 2. How far each illusion shifts perception, in humans and
> GPT-5.2.** (**A–E**) Perceptual shift against illusion strength. The
> perceptual shift is the displacement of the point of subjective equality
> (PSE), the physical difference at which the two answers are given equally
> often. At each absolute strength, the responses from the illusion's two
> spatial directions were fitted jointly by a cumulative Gaussian with a
> shared slope and shared lapse rates; the shift is half the distance between
> the two directions' PSEs. Strength is normalised to the strongest tested
> level, so it runs from −1 to +1 in every panel, and its sign is the
> direction in which the context displaces the percept. Because each shift is
> estimated from both directions at once, it is plotted at both +x and −x,
> and every curve is point-symmetric about the origin by construction. Curves
> are natural cubic regression splines (4 basis functions) through the seven
> tested strengths and the origin, weighted by the inverse variance of each
> estimate; bands are 95% confidence intervals. Both species are divided by
> the peak of the human curve for that illusion, so the human curve reaches
> ±1 and the GPT-5.2 curve reads as a fraction of the human shift.

The divisor for each illusion, in that illusion's own difference units (the
peak of the smoothed human curve):

| Müller-Lyer | Vertical-Horizontal | Ponzo | Ebbinghaus | Contrast |
|---|---|---|---|---|
| 0.415 | 0.193 | 0.252 | 0.355 | 12.04 |

Two levels are less well constrained than the rest:

- **Vertical-Horizontal, GPT-5.2.** At three of the seven strengths
  (|k| = 1, 4 and 7) the PSE in one direction lies beyond the largest tested
  difference. Their intervals are wide and one-sided, so they carry little
  weight in the smooth.
- **Müller-Lyer, humans, |k| = 7.** The PSE sits at the fitting bound.

---

## Figure 3 — Error rate

> **Figure 3. Error rate against illusion strength, in humans and GPT-5.2.**
> (**A–E**) Percentage of incorrect answers at each illusion strength, pooled
> over all differences tested at that strength. Strength is normalised to the
> strongest tested level. Negative strengths are congruent (the context
> supports the correct answer) and positive strengths are incongruent (the
> context opposes it). Curves are natural cubic regression splines in strength
> (6 basis functions), fitted by binomial regression to the counts at the 15
> tested strengths. Bands are 95% confidence intervals, widened for
> overdispersion (quasi-binomial) because each strength pools observers who
> differ in susceptibility. Humans responded under time pressure and GPT-5.2
> was untimed, so the absolute level of the curves is not comparable between
> species; their shape is.

Trials per strength: humans 2,048 (4,024 at zero strength). GPT-5.2: 800
(1,600 at zero), or 880 (1,760) for Müller-Lyer.

---

## Figure 4 — Relative to humans

> **Figure 4. Each illusion's effect on GPT-5.2, relative to its effect on
> humans.** Each bar is GPT-5.2's summary divided by the human one. The dashed
> line marks the human effect, which is 1 by definition, so a bar at the line
> means the illusion affects the model as much as it affects people. (**A**)
> Perceptual shift: the Figure 2 curve averaged over strengths from 0 to +1.
> The curves are symmetric by construction, so the negative half adds nothing.
> (**B**) Error rate: the Figure 3 curve averaged over incongruent strengths,
> minus its average over congruent strengths. This is the error the illusion
> adds; subtracting the congruent side removes the baseline error, which
> differs between species because humans were speeded and GPT-5.2 was not.
> Both summaries come from the same fits as Figures 2 and 3. Whiskers are 95%
> confidence intervals for the ratio (Fieller's method), which carry the
> uncertainty of both the GPT-5.2 and the human estimate. In each panel,
> illusions are ordered by GPT-5.2's value, largest first.

GPT-5.2 relative to humans, with 95% intervals, in the figure's order. Both
panels rank the illusions identically.

| Illusion | Perceptual shift | Error rate |
|---|---|---|
| Vertical-Horizontal | 1.02 (0.87–1.19) | 1.17 (0.93–1.43) |
| Contrast | 0.91 (0.75–1.07) | 0.86 (0.62–1.15) |
| Müller-Lyer | 0.62 (0.56–0.69) | 0.70 (0.58–0.82) |
| Ebbinghaus | 0.34 (0.11–0.57) | 0.43 (0.19–0.69) |
| Ponzo | 0.20 (0.05–0.35) | 0.25 (0.00–0.50) |

The error effects behind panel B, in percentage points (incongruent minus
congruent), human / GPT-5.2: Müller-Lyer 54.0 / 37.6, Vertical-Horizontal
44.4 / 51.8, Ponzo 29.1 / 7.2, Ebbinghaus 26.8 / 11.5, Contrast 31.4 / 27.1.

The two panels are computed from separate data and separate fits, yet agree:
on every illusion the two intervals overlap. On both measures,
Vertical-Horizontal and Contrast are consistent with the human effect,
Müller-Lyer is below it, and Ponzo and Ebbinghaus are far below it.

**How the intervals are computed.** Each summary is a function of its spline
coefficients. The mean shift is linear in them, so its variance is exact
given their covariance. The error effect goes through the logistic link, so
its variance comes from the delta method. Both use the same dispersion
correction as the drawn bands. The GPT-5.2 and human summaries come from
independent fits, so Fieller's interval combines their variances directly.
The pointwise bands in Figures 2 and 3 are not used, since they carry no
information on how the points along a curve covary.

**Why not the value at full strength.** An earlier version used each curve's
value at +1. That is where a smooth is least well pinned, and two of those
values differed markedly from the raw data at the strongest tested level
(Ponzo humans, raw 1.26 against a drawn 1.00; Vertical-Horizontal GPT-5.2,
raw 1.41 against 0.81). The average over the range uses all seven tested
strengths and is not driven by the end point.

---

## Figure S1 — Error rate by difficulty

> **Figure S1. Error rate against illusion strength, split by task difficulty,
> in (A) humans and (B) GPT-5.2.** Each line is one difficulty band. The eight
> tested physical differences are pooled in adjacent pairs, from *Hardest*
> (the two smallest differences) to *Easiest* (the two largest). Pooling in
> pairs gives every band a value at every strength in both species; the human
> design shows only every other difference at each strength, and each pair
> contains one of each. Points are raw error rates on the shared grid, with
> both signs of the difference pooled; Figure 3 shows the same data smoothed
> and without the split. Strength is normalised to the strongest tested level:
> negative strengths are congruent, positive strengths incongruent.

---

## Figure S2 — Response surfaces

> **Figure S2. The raw choices behind Figures 2 and 3, in (A) humans and (B)
> GPT-5.2.** Each cell is the proportion of trials on which the first-named
> option (Top or Left) was chosen, at one illusion strength (x) and one signed
> difficulty band (y; the same bands as Figure S1). Rows below 0 are
> differences favouring the second option, easiest at the bottom; rows above 0
> favour the first option, easiest at the top. With no illusion the switch
> between answers sits at 0 on the y-axis; an illusion that shifts perception
> moves it, and at strong incongruent strengths it pushes responses across the
> physical difference. The scale is greyscale so that it is not read as the
> blue and orange that mark the two species elsewhere.

---

## Figure S3 — Exact response probabilities (draft, awaiting the open-model runs)

> **Figure S3. Exact response probabilities against the sampled responses,
> for the open-weight models.** For each stimulus, each model's probability of
> choosing each option was read directly from its output, from the same pass
> that produced its 100 sampled answers. Lines: the error rate these
> probabilities predict at the sampling temperature (0.3), averaged over the
> differences tested at each strength. Dots: the error rate of the 100 sampled
> answers at the same strengths. Where the dots follow the line, the samples
> behave as draws from the model's own probabilities, and the line is the
> same curve as in Figure 3 without sampling noise. GPT-5.2 is not shown
> because its probabilities were not stored. Strength is normalised as in
> Figure 3; shared-grid cells only.

---

## Notes for the Methods

**Held out of the figures.** Both illusions stay in the dataset and in every
analysis table.

- **Rod-Frame.** A psychometric function does not describe GPT-5.2's
  responses. In 12 of its 14 direction slices, the response curve doubles
  back. The other illusions have at most 4 of 14, and humans have none.
- **Delboeuf.** GPT-5.2 treats a ring drawn close around a disc as the disc's
  own edge. Where judging the disc and judging the ring's outer edge give
  different answers, it follows the outer edge on 94% of trials at the
  tightest ring (1.14× the disc). At the loosest ring (1.78×) it does so on
  15% of trials. Humans do so on about 10% throughout. Its Delboeuf curve
  therefore measures a failure to separate a figure from its surround, not
  susceptibility to the illusion. Our Delboeuf stimuli also differ from the
  human ones (next note).

**Stimulus versions.** The human data were collected in August 2022 with
Pyllusion 1.2, and our stimuli were generated with a later version. Of the
five illusions shown, only the Ebbinghaus generator changed in between. Commit
f472ebac (13 October 2022) mirrors the distractor layout between the two
sides, and a later commit adds a colour argument. The other four illusions
are generated by identical code. The Delboeuf generator changed materially
(commit 4ee0c33c), which is part of why it is held out.

**Sign convention.** For every illusion, negative strength is congruent and
positive strength is incongruent. Contrast follows this without the axis flip
that Pyllusion's changelog suggests for comparisons with human data: in both
species, errors are rare at negative strength and common at positive strength
(Figure 3E).

**Psychometric fits (Figure 2).** At each absolute strength, the two
directions are fitted as one cumulative-Gaussian model by binomial maximum
likelihood. Each direction has its own PSE; the slope and the two lapse rates
are shared. 95% intervals come from the profile likelihood. The smooth weights
each level by 1/SE², with SE taken from the width of that interval.

**Aggregation.** The human curves pool 256 participants with one trial per
cell each, so they mix between- and within-participant variability. This
overdispersion is why the Figure 3 bands are widened; a model with a random
effect per participant is the stricter alternative.
