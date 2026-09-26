# Research gaps

Derived from the 30 verified papers in `literature/literature_review.xlsx`
(sheets `Literature Review`, `Literature Analysis` and `Research Gaps`) and the
corpus assembled by `tools/literature/`. The machine-readable form of this
analysis is `tools/literature/research_gaps.json`, with per-gap evidence in
`literature/verification/gap_evidence.json`.

A gap is only listed here if the reviewed papers give evidence for it. Where the
evidence is mixed, the disagreement is preserved rather than resolved. Paper
numbers below are the serial numbers in the `Research Gaps` sheet of the workbook.

---

## G1 — Aggregate-only evaluation dominates the literature

**Evidence.** A majority of the reviewed papers report a single test-period
aggregate (RMSE and/or MAE, often with R²) and stop there. Regime-, season- and
time-of-day-stratified evaluation appears in a small minority, and the number that
report it is smaller still for models whose *ranking* changes between conditions.

**Why it matters.** A single aggregate hides exactly the conditions that decide
whether a forecast is usable. Two models with identical annual nRMSE can behave
completely differently at the morning ramp or under broken cloud, and the aggregate
is blind to both. Reporting R² without also reporting a skill score against a
persistence reference compounds the problem: on a diurnally dominated target, a
model that has learned only the time of day can post a high R² while being unable
to predict a single cloud transition.

**What the literature does.** Aggregate nRMSE/MAE/R², frequently against
persistence or a linear fit, sometimes with a per-horizon breakdown.

**Limitation.** Condition-dependence of the error and of the *ranking* is largely
unreported, so a reader cannot tell whether a model that looks best on average
would be the one to deploy on a cloudy morning.

**Opportunity for this project.** Report every model against both persistence
references, stratified by weather regime, season, time-of-day band, generation
level and a ramping flag, with the persistence reference rescored inside each
stratum.

**Testable question.** Does the ranking of model families change across weather
regimes, and is skill against persistence more stable across regimes than absolute
error?

---

## G2 — Horizon sensitivity is asserted more often than measured

**Evidence.** Papers that cover a single horizon dominate. Where several horizons
are studied, the horizon is usually fixed in the data description and the
comparison is between models at that one horizon, rather than a model's behaviour
as the horizon grows. Multi-horizon studies are present but are a clear minority of
the corpus, and few of them reuse the same inputs and the same split across
horizons.

**Why it matters.** Horizon changes the information content of the input: a
one-hour forecast can lean on persistence, a 24-hour forecast must lean on
irradiance and calendar structure. A model that wins at one hour can lose at six
without that being visible from the one-hour result.

**Opportunity.** One protocol, one split, one input set, four horizons
(15 minutes, 1 hour, 6 hours, 24 hours), all exact integer multiples of the working
resolution so that no interpolation enters the comparison.

**Testable question.** Is any architectural advantage preserved as the horizon
grows, and does skill against persistence decay monotonically?

---

## G3 — Attention and transformer architectures are rarely compared against well-tuned tabular baselines

**Evidence.** The recurrent and attention-based papers compare most often against
each other and against persistence, and much less often against a tuned
gradient-boosted tree model on the same inputs. Where tree models do appear, they
are commonly at default settings. The feature-importance papers go further and
argue that tree ensembles remain competitive for tabular structured time series.

**Why it matters.** The common framing "deep learning versus classical machine
learning" is answered with a weak classical baseline and generalised as a finding
about methods. If the classical arm is under-tuned, the comparison measures the
baseline, not the method.

**Opportunity.** Give every family a small validation-only grid search, and report
the tree arm at its tuned operating point. The cost of doing this is bounded and
it makes the headline comparison defensible.

**Testable question.** Under a single protocol with a tuned classical arm, do
attention-based architectures improve on gradient boosting at short horizons?

---

## G4 — Persistence is under-used as a *strong* reference

**Evidence.** Persistence appears in most of the corpus, but usually as the
simplest of several baselines, and the clear-sky-ratio variant — which corrects
persistence for the deterministic diurnal shape of the solar resource and is far
harder to beat — appears in only a few papers. A skill score is often reported
against the weakest available reference, or not at all.

**Why it matters.** A model that beats naive persistence by a wide margin may still
be beaten by a clear-sky persistence on the same day, so the practical conclusion
drawn from the first comparison overstates the model's value.

**Opportunity.** Two references, both reported per model per stratum, with skill
defined explicitly and evaluated on exactly the timestamps used for the model.

**Testable question.** Which models retain positive skill against *smart*
persistence, not merely against naive persistence?

---

## G5 — Uncertainty quantification is largely absent, and rarely calibrated

**Evidence.** Probabilistic PV forecasting is a visible but small strand of the
corpus. Most of those papers are evaluated with a proper scoring rule or a
pinball loss, and interval papers report empirical coverage. The gap is on the
benchmark side: the deterministic studies that dominate the corpus report no
interval at all, so the accuracy-versus-calibration trade-off is rarely measured.

**Why it matters.** A point forecast without an interval cannot be used for
reserve planning, and an interval that is too narrow is worse than no interval
because it is trusted. Accuracy alone does not answer whether the model knows what
it does not know.

**Opportunity.** Distribution-free split conformal intervals around the
deterministic models, with empirical coverage reported per model, per level and
per month, and the exchangeability caveat of the calibration choice stated rather
than glossed over.

**Testable question.** What interval width is needed to reach nominal coverage, and
does coverage degrade seasonally?

---

## G6 — Explainability is often asserted from attention weights or single-method rankings

**Evidence.** A minority of the corpus includes any interpretability analysis. Of
those, tree-model work uses SHAP or impurity importances, while the recurrent and
attention papers more often present attention maps as explanations. Direct
cross-family comparison of importance methods on one dataset and one split is
close to absent.

**Why it matters.** Attention weights are internal activations, not validated
attributions; reading them as explanations is a claim that has not been measured.
A single ranking method on a single model also cannot be checked against anything.

**Opportunity.** SHAP for the tree ensemble, grouped permutation importance for
the sequence models, on one split with one input set, reported side by side, with
an explicit statement that both are predictive rather than causal attributions.

**Testable question.** Do the two importance families agree on which variables
matter, and where do they disagree?

---

## G7 — Reproducibility is uneven, and comparison across studies is not possible

**Evidence.** Code availability is stated in a minority of papers and dataset
availability in fewer still. Split boundaries, random seeds, preprocessing details,
feature lists and package versions are reported inconsistently: several papers give
neither the split dates nor the horizon resolution, so their numbers cannot be
placed on the same axis as anyone else's. Reported metrics are not accompanied by
uncertainty in most cases, so a 1% difference between two papers may be noise.

**Why it matters.** Without seeds, splits and versions, a ranking in one paper
cannot be compared with a ranking in another, and the field's aggregate picture
stays qualitative. Without intervals or tests, small differences are read as
findings.

**Opportunity.** Per-run machine-readable records (split dates, seed, feature list,
hyperparameters, durations, package versions, git revision, configuration
fingerprint), a registry rebuildable from those records, and statistical comparison
of the models that were actually run.

**Testable question.** How much of the apparent difference between two models
survives a paired test with a serial-correlation-aware variance estimate?

---

## G8 — Deployment cost is rarely measured alongside accuracy

**Evidence.** Accuracy is reported almost universally; training cost, inference
latency, parameter count and memory footprint are reported in only a few papers, and
almost never together with the accuracy figure that would justify the cost.

**Why it matters.** A benchmark that reports only accuracy implicitly recommends
the most expensive model. On the CPU-only hardware typical of small rooftop
operators, a model that is 3% worse in nRMSE and ten times cheaper may be the
correct choice, and that trade-off is invisible without the cost axis.

**Opportunity.** Training duration, per-window inference latency and parameter
count recorded with every run, thread counts pinned for comparability, and a
Pareto frontier of accuracy against cost.

**Testable question.** Which models are Pareto-optimal for deployment, and does the
best model by error differ from the best model by cost?

---

## What this project does *not* claim as a gap

Two candidate gaps were considered and **not** claimed, because the evidence does
not support them:

* **"Deep learning is always better for PV forecasting."** The corpus contains
  both systematic reviews reporting that deep architectures dominate and studies
  reporting that tree ensembles match or beat them on tabular structured data. That
  disagreement is preserved in the review rather than resolved, so it is not used
  as evidence for a gap.
* **"Weather variables are under-used."** The evidence points the other way: the
  reviewed literature uses irradiance, temperature and calendar features nearly
  universally. The interesting question is not whether they are used but which
  *subset* carries the signal, which is G-level feature ablation territory rather
  than a gap in the literature.

## Where this project's evidence is weakest

Stated so the gap list is not read as stronger than it is:

* The corpus is 30 papers, not a systematic review of the field. It is a
  convenience sample of the accessible, verifiable literature, collected through
  DOI-verified sources; it is not a PRISMA search and is not claimed to be
  exhaustive.
* A single paper can support or weaken a gap, and several gaps above rest on
  "absence of evidence in the reviewed set" rather than on a paper that
  demonstrates the absence. Those are marked as such in the `Research Gaps` sheet.
* The gap list is therefore best read as a *motivation for a protocol* — a
  controlled, leakage-audited, stratified, statistically tested benchmark — rather
  than as a claim that no such benchmark exists in the literature.
