# A note for whoever owns the pathway model

**One decision is waiting on you: the value of `pathway_engagement_min_probability`.** It ships as
`0.5`, and at that value the honesty of every pathway call in the platform collapses. This file
explains why, points at the exact lines, and sets out what the choice actually is. Nobody on the
application side can make it, because it depends on how the model is calibrated.

---

## Where it lives

| What | Where |
|---|---|
| The default, `0.5` | [`src/pipeline/s07_structure_mechanism/stage.py:37`](../src/pipeline/s07_structure_mechanism/stage.py) — `DEFAULT_THRESHOLDS` |
| The schema default a request inherits | [`src/schemas/stage_configs.py:176`](../src/schemas/stage_configs.py) — `pathway_engagement_min_probability: float = 0.5` |
| Where it is applied | [`src/pipeline/s07_structure_mechanism/stage.py`](../src/pipeline/s07_structure_mechanism/stage.py) — `build_mechanistic_evidence_summary()` |
| The example request that carries it | [`src/backend/api_e2e/example_request.json:77`](../src/backend/api_e2e/example_request.json) |
| The model | [`model_store/pathway_mapping_predictor_v2/`](../model_store/pathway_mapping_predictor_v2/) |
| Tests describing the current behaviour | [`tests/unit/pipeline/test_s07_structure_mechanism.py`](../tests/unit/pipeline/test_s07_structure_mechanism.py) |

---

## What the model's probability means

From the predictor's own README, and it is the crux of everything below:

> Task per pathway: **activator (+1) vs inhibitor (-1)**; label-0 rows were dropped. Unlike v1
> (pathway engagement, PU-bagging), the probability here is **P(activator)**, so a low value means
> "inhibitor", not "not involved".

Two consequences follow, and both are easy to miss:

1. **The model cannot say a peptide leaves a pathway alone.** The "not involved" class was dropped
   from training. Every peptide is assigned a direction on all 9 pathways, whether or not it
   interacts with any of them.
2. **Confidence is the only gate available.** With no abstain class, the only way to express "we
   do not know" is to decline to call probabilities near the middle.

## What the threshold does

```python
activated   = [p for p in pathways if probability[p] >= cutoff]
inhibited   = [p for p in pathways if probability[p] <= 1.0 - cutoff and p not in activated]
undetermined = everything else
```

Symmetric by construction: `cutoff` gates activation, `1 - cutoff` gates inhibition, and the gap
between them is where the model is not confident enough to call it.

**At `cutoff = 0.5` that gap is empty.** `p >= 0.5` and `p <= 0.5` meet, so every pathway lands in
one of the two lists and **a probability of 0.501 is reported as definitely activating**. Measured,
not reasoned about — nine labels spanning the range, five of them within 0.05 of a coin flip, all
resolved to a direction:

| cutoff | activated | inhibited | undetermined |
|---|---|---|---|
| **0.5 (shipped)** | 5 | 4 | **0** |
| 0.7 | 2 | 2 | **5** |

## Why this matters beyond the display

`activated_pathways` is not only shown to a user. `functions_supported` is derived from it, so an
over-confident activation call becomes **evidence that a candidate supports a desired biological
function**, which feeds the ranking. A coin flip at stage 7 can promote a candidate at stage 11.

## What was changed, and what was not

**Changed.** A pathway between the two cutoffs used to vanish from the output — in neither list,
mentioned nowhere. A reader takes an absent pathway as "not relevant" when it means "the model
could not call it". There is now an `undetermined_pathways` list, carried through the API to the
run view and named in the rendered summary.

**Not changed: the default itself.** Raising it alters every run's output and the right value
depends on how well-calibrated the model's probabilities are, which is a question for whoever
trained it. `test_at_the_default_cutoff_nothing_is_undetermined` documents the current behaviour
without endorsing it.

## The decision, concretely

**Is `P(activator)` calibrated?** That is the question underneath. If 0.6 genuinely means "60% of
peptides scoring this are activators", a cutoff near 0.5 is defensible and the UI should lean on
the probabilities rather than the lists. If it is not calibrated — likely, for an unweighted mean
of an LDA and a random forest on 11 physicochemical descriptors — then a probability near 0.5
carries almost no information and reporting it as a direction is a fabrication.

Three options, in the order I would consider them:

1. **Raise the default** to something like 0.7 and accept more `undetermined`. Honest, cheap,
   loses recall. Needs only a one-line change in both files above, plus a baseline update.
2. **Calibrate** on held-out data (Platt scaling or isotonic regression) and then pick the cutoff
   from the calibration curve rather than by eye. The right answer if the data exists.
3. **Retrain with the label-0 rows** so the model can express "not involved" at all. The most work
   and the only one that fixes the underlying limitation rather than hiding it behind a threshold.

Whichever is chosen, two things should land with it: a `model_card.json` for
`pathway_mapping_predictor_v2`, which currently ships none, and an update to
[`docs/BASELINE.md`](BASELINE.md), where this is recorded as open.

## One thing to be careful of

Changing the threshold **changes stored results' meaning, not just new ones**. Runs recorded
before the change will have had their pathways called at 0.5, and nothing in a stored result says
which threshold produced it — the same gap as `docs/BASELINE.md` records for model identity. If
the default moves, old runs and new runs are no longer comparable, and nobody reading them can
tell.
