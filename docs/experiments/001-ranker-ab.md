# Experiment 001: does the second-stage ranker actually help readers?

**Status: PRE-REGISTERED, not yet run.** Everything below the results section was written and
committed before a single reader was simulated. The results section is deliberately empty. If the
outcome is null or negative it gets written up here unchanged.

## Why run this at all

The LightGBM LambdaMART reranker went live on 2026-09-25 on the strength of offline metrics:
NDCG@20 improved from 0.241 to 0.312, a 30% gain over stage-1 retrieval alone.

Two things make that number less convincing than it looks.

1. **Catalog coverage fell from 54.3% to 45.9%.** The ranker concentrates recommendations on fewer
   books.
2. **The pure ranker recommended bestsellers to everyone.** Scoring 0.254 against stage 1's 0.236 on
   validation, it still had to be anchored to stage 1 at `STAGE1_WEIGHT = 0.5` to stop it. Dropping
   the popularity features did not fix it.

Offline relevance metrics reward popularity, because popular books are the ones users have rated.
So a 30% NDCG gain is consistent with two very different stories: the ranker genuinely orders books
better for each reader, or the ranker learned to show everyone the same well-known books. Offline
evaluation cannot separate those. That is what this experiment is for.

## Hypothesis

**H1 (primary, directional).** Readers served the two-stage ranking give positive feedback on a
greater share of the recommendations they are shown than readers served stage-1 retrieval alone.

**H0.** No difference in positive feedback rate between the two arms.

## Design

|  |  |
|---|---|
| Type | Two-arm, between-subjects, randomized |
| Unit of randomization | Reader |
| Unit of analysis | Reader |
| Arms | **control**: stage-1 hybrid only. **treatment**: stage-1 plus LightGBM reranker |
| Allocation | 50/50 |
| Traffic | Replayed Goodbooks readers via `alexandria_ml.simulate_traffic` |

**Why the reader is the unit.** A reader who saw both rankings would contaminate the comparison,
and feedback persists per reader in `interactions`. Analysis is also at reader level: the ~60
impressions one reader sees are correlated, so treating them as 60 independent observations would
manufacture significance. Each reader contributes exactly one number.

**Assignment.** Two API instances of the same image against the same seeded database, differing
only in the existing `RECOMMENDATION_USE_RANKER` environment variable: control on port 8001 with
it `false`, treatment on port 8000 with it `true`. Both instances are verified at run time to have
resolved the flag as intended rather than trusted to have picked it up.

Readers are drawn **once** for the whole experiment, `readers * arms` of them under a shared seed,
then sliced per arm. Drawing separately per arm would let the same Goodbooks reader appear in both,
which breaks the independence the between-arm comparison assumes. Arms are disjoint by
construction, and `ml/tests/test_simulate_split.py` asserts it.

Reader accounts are tagged by arm, `simA_` for control and `simB_` for treatment, so the split is
reconstructable from the database alone and does not depend on keeping the output files.

**Amendment, 2026-09-27, before any treatment data existed.** This replaces an earlier plan to
assign in-app by hashing the user id and to store a `variant` column on `events`. Two instances
achieve the same randomization while touching no shipped code, needing no migration against the
production database, and leaving `recommender.py:44` untouched. It also lets both arms run
**concurrently**, so that machine load or drift over the hours of the run cannot be confounded with
the arm, which sequential single-instance running would not have given.

The condition this design depends on is that the arms cannot interfere through the shared database.
Verified: every write in the recommendation and feedback path is a per-user row (`interactions`,
`events`). The `books` table is written only by `app.seed`, never at serve time, and `popularity`
and `ratings_count` are read-only while serving. There is no shared mutable state between arms.

## Metrics

**Primary: positive feedback rate per impression.** Per reader, `positive / shown`, where positive
means a `loved` or `liked` signal. One metric, fixed now. No switching after seeing the data.

**Guardrails.** Any of these failing means the experiment does not support shipping, even if the
primary metric wins:

| Guardrail | Why | Fails if |
|---|---|---|
| Catalog coverage | Already known to drop 8 points offline | Treatment coverage falls below 40% |
| Bestseller skew, share of recommendations in the top 1% by popularity | The specific failure mode suspected | Treatment exceeds control by more than 10 points |
| p95 recommendation latency | Two stages cost more than one | Treatment exceeds 500 ms |

**Secondary, reported but not decisive.** Opinion rate (`rated / shown`), precision given an
opinion (`positive / rated`), and the popularity-weighted positive rate described next.

## The confound that matters most

In this simulation a reader can only respond to a book they actually rated in Goodbooks. Rating
data is itself popularity-biased: popular books are far more likely to have been rated. So a ranker
that pushes bestsellers will score **higher on the primary metric almost by construction**, which
is exactly the failure mode this experiment exists to detect.

This is a real limitation and the reason the guardrails above are load-bearing rather than
decorative. As a partial correction, the secondary analysis reports an inverse-popularity-weighted
positive rate, weighting each positive response by `1 / log(1 + popularity)` so that a positive
response to an obscure book counts for more than one to a bestseller. A treatment that wins on the
raw metric but loses on the weighted one is showing bestsellers, not better recommendations, and
will be reported as such.

## Power analysis

Two-sample comparison of reader-level means, two-sided, alpha 0.05, power 0.80, computed by
`alexandria_ml.power`.

**Procedure, committed in advance.**

1. Run a 30-reader pilot on the control arm only. Estimate the reader-level SD with
   `power.estimate_from_pilot`.
2. Set the target MDE at **15% relative** and compute n from the observed SD.
3. Cap total simulation at 300 readers per arm. If the computed n exceeds the cap, run at the cap
   and report the effect the experiment was actually powered to detect instead of pretending to the
   original MDE.
4. Discard the pilot. It estimates variance only and does not contribute to the result.

### Pilot result, 2026-09-27

30 readers, control arm, 1,800 impressions, seed 1. Raw per-reader records:
[`data/001-pilot-control.jsonl`](data/001-pilot-control.jsonl).

| | |
|---|---|
| Mean positive rate | **0.2006** |
| Reader-level SD | **0.1404** |
| Range across readers | 0.017 to 0.533 |

Two things to note. The baseline is **0.20, not the 0.39** assumed when this document was drafted:
that earlier figure was the *opinion* rate (any response, positive or negative), and the positive
rate per impression is half of it. And the between-reader spread is enormous, a factor of 30 from
the least to the most responsive reader, which is exactly why the unit of analysis is the reader.
Pooling these impressions would badly understate the true variance.

Sample sizes at the measured SD of 0.1404, at roughly 30 seconds per reader:

| relative lift | absolute | readers/arm | simulation time |
|---|---|---|---|
| 10% | 0.0201 | 771 | 12.8 h |
| 15% | 0.0301 | **344** | 5.7 h |
| 20% | 0.0401 | 195 | 3.2 h |
| 30% | 0.0602 | 88 | 1.5 h |

**The 15% target needs 344 readers per arm and the pre-committed cap is 300.** Rule 3 applies: run
at the cap of **300 per arm, 600 readers, about 5 hours**, and report the effect the experiment can
actually resolve, which at that n is **0.0321 absolute, a 16% relative lift**.

The difference between the 15% target and the 16% achieved is immaterial, but the rule is followed
as written rather than quietly moving the cap to fit. Consequence, stated plainly: **this
experiment cannot detect a lift smaller than about 16%.** A null result will mean "no effect of 16%
or larger was detected", not "there is no effect".

## Stopping rule

Run to the fixed sample size from step 2. No interim looks, no early stopping, no extending the run
because the result is nearly significant. Committing to this in writing before starting is the only
thing that makes the p-value mean anything.

## Analysis plan

- Difference in mean reader-level positive rate, treatment minus control
- 95% confidence interval by bootstrap over readers, 10,000 resamples
- Welch's t-test, which does not assume equal variances
- Report the effect size and interval as the headline. The p-value is secondary and a null result is
  reported as a null result
- Guardrail table with all three metrics, regardless of the primary outcome
- Assignment balance check: arm sizes, and mean reader history length per arm, to confirm
  randomization did not produce lopsided arms

## Threats to validity

1. **Simulated readers are not real users.** They respond from historical Goodbooks ratings, so they
   cannot express interest in a book they never encountered. This is a randomized comparison on
   replayed real preferences, not a production A/B test, and it will not be described as one.
2. **Popularity confound**, covered above.
3. **No novelty effect and no long-term measurement.** Each reader answers three pages in one
   sitting. Nothing here speaks to retention.
4. **Single dataset.** Goodbooks-10k, ratings stopping at 2017.

## Results

Run 2026-09-27. 300 readers per arm, 18,000 impressions per arm, 36,000 total. Arms ran
concurrently, seed 7. Validity checks passed: arms disjoint, balanced, every reader saw exactly 60
recommendations.

### Primary metric: no effect detected

| arm | readers | impressions | mean positive rate | SD |
|---|---|---|---|---|
| control | 300 | 18,000 | 0.1908 | 0.1208 |
| treatment | 300 | 18,000 | 0.2047 | 0.1196 |

| | |
|---|---|
| Absolute difference | **+0.0139** |
| Relative to control | **+7.3%** |
| 95% bootstrap CI | **[-0.0051, +0.0328]**, i.e. [-2.6%, +17.2%] |
| Welch t | 1.415 (dof 598) |
| two-sided p | **0.157** |

The interval includes zero. **H0 is not rejected.**

The important part is not the null itself but where the interval sits. The offline evaluation
reported a **30% NDCG@20 improvement**. This experiment's upper confidence bound is **+17.2%**, so
an online effect of the size the offline metric implied is **excluded by the data**. Whatever the
reranker gained offline, most of it did not reach readers.

Stated carefully: the experiment was powered for a 16% relative lift, so this is "no effect of
roughly 16% or larger", not "no effect". A true lift of 5% would be entirely consistent with these
numbers and this design could never have found it.

### Guardrails

| guardrail | control | treatment | change |
|---|---|---|---|
| Catalog coverage | 26.3% | 22.9% | **-3.4 pts** |
| Share of impressions in the top 1% most popular | 39.5% | **48.6%** | **+9.1 pts** |
| p95 latency | not instrumented | not instrumented | **not measured** |

The ranker concentrates recommendations. It shows fewer distinct books (2,793 against 3,208) and
leans substantially harder on bestsellers. The bestseller guardrail was set to fail above a 10
point gap and came in at 9.1, so it passes as written, but only just, and in exactly the direction
the experiment was designed to be suspicious of.

**A pre-registration error worth recording.** The coverage guardrail was written as "fails if
treatment falls below 40%", a threshold lifted from offline evaluation. Both arms are far below it,
because offline coverage is measured over a full evaluation set while this measures 300 readers
seeing 60 books each, a completely different denominator. The absolute threshold was meaningless
and should have been specified as a relative change between arms. The relative comparison, -3.4
points, is the number that means anything. The threshold is left as originally written rather than
retconned.

### Secondary: popularity-weighted positive rate

Each positive response weighted by `1 / ln(1 + popularity)`, so a positive on an obscure book
counts for more than one on a bestseller.

| arm | mean weighted rate | SD |
|---|---|---|
| control | 0.01951 | 0.01014 |
| treatment | 0.02031 | 0.00988 |

Relative difference **+4.1%**, against **+7.3%** on the raw metric. Roughly 40% of the apparent
(and already non-significant) advantage disappears once popularity is discounted. Combined with the
9.1 point rise in bestseller share, the most consistent reading is that the reranker's offline
advantage comes substantially from surfacing well-known books, which is what the confound section
predicted and what the anchoring to stage 1 was already compensating for.

### Conclusion

The reranker is **not demonstrated to help readers**, and the 30% offline gain does not survive
contact with a randomized online comparison. It costs catalog diversity and measurably increases
bestseller concentration.

This does not mean it should be reverted. The effect is directionally positive, the confidence
interval is wide, and the simulation's own popularity bias works *in the treatment's favour*, so
the real effect may well be smaller than measured rather than larger. What it does mean is that
**NDCG@20 is not a trustworthy proxy for reader benefit in this system**, and shipping decisions
based on it alone are not well founded.

Next question worth an experiment, and deliberately not answered by peeking at this data: whether
`STAGE1_WEIGHT` is at the right point, since it is currently the only thing restraining the
popularity concentration measured above.
