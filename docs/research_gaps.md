# Research gaps

Derived from the 30 verified papers in `literature/literature_review.xlsx`
(sheets `Literature Review`, `Literature Analysis`, `Research Gaps`, `Exclusion
Log`) and the corpus assembled by `tools/literature/`. The machine-readable form
is `tools/literature/research_gaps.json`, with per-gap evidence in
`literature/verification/gap_evidence.json`.

An earlier pass of this document listed ten gaps. The triage is recorded rather
than applied silently: all twelve candidates stay in the JSON, each marked
`retained` with a reason, and `tools/literature/triage_gaps.py` reproduces the
decision. **Seven are retained** — one umbrella and six themes — and five are not.
A gap is retained only where the reviewed papers either demonstrate the absence
or are themselves evidence of the problem, *and* the project does not already
answer it.

| Retained | Theme | Section |
| --- | --- | --- |
| RG-01 | No shared, leakage-audited benchmark across the three model families | umbrella, see below |
| RG-02 | Aggregate-only evaluation; condition-dependence unreported | G1 |
| RG-05 | Horizon sensitivity asserted rather than measured | G2 |
| RG-08 | Classical arm reported untuned; cost not reported with accuracy | G3 |
| RG-11 | The strong reference is under-used; skill is flattered by naive persistence | G4 |
| RG-07 | Feature importance from one method, or from attention weights | G5 |
| RG-12 | Cross-site transfer claimed far more often than it is validated | G6 |

Evidence strength is stated per gap:

* **Demonstrated** — the reviewed papers contain a result that shows the gap.
* **Absent from the corpus** — no reviewed paper addresses it. Weaker: absence of
  evidence in a 30-paper convenience sample is not evidence of absence in the
  field, and is labelled as such.

RG-01 is the umbrella rather than a theme: every section below is an instance of
it, and it is stated here so the six themes are not read as six unrelated
criticisms.

---

## G1 — Aggregate-only evaluation dominates; condition-dependence is unreported

**Evidence strength:** Absent from the corpus (19 of 30 papers report a single
test-period aggregate and nothing conditional).

Most of the corpus reports one aggregate number per model and stops. Stratified
evaluation by weather, season or time of day appears in a minority, and reporting
a *ranking change* between conditions is rarer still.

**Why it matters.** Absolute error is confounded with signal amplitude: under
broken cloud, irradiance and therefore power are low, so absolute errors are small
for reasons that have nothing to do with forecastability. This project measures
the effect directly — persistence nRMSE is **0.104 under cloudy conditions
against 0.185 in clear sky**, while the *skill* against persistence is lowest
under cloud (+0.024 against +0.358). A reader given only the error table would
conclude the opposite of the truth.

**Opportunity.** Stratify every model by regime, season, time of day, generation
level and ramping, with the reference rescored inside each stratum.

**Answer in this study.** Experiments D, G and the full error analysis. The
ranking is stable across regimes for the tree family, which is itself worth
knowing, and the skill of every model collapses under broken cloud.

**Testable question.** Does the ranking of model families change across weather
regimes, and is skill against a strong reference more stable than absolute error?
**Answered: no on the first, no on the second — skill is the *less* stable of the
two.**

---

## G2 — Horizon sensitivity is asserted rather than measured

**Evidence strength:** Absent from the corpus (single-horizon studies dominate;
few multi-horizon studies hold inputs and split fixed across horizons).

Where a horizon is fixed in the data description, a comparison at that single
horizon cannot distinguish a model that is genuinely better from a reference
that happens to be weak at that horizon.

**Why it matters — and this is the study's sharpest finding.** The reference's
competence is a function of the horizon. Persistence is the **strongest** model
in the table at 15 minutes (skill −0.65 for LSTM, −0.67 for GRU) and the
**weakest** at 6 hours, where its nRMSE is 0.407 against 0.166 at one hour and
every learned model reaches skill 0.53–0.59. The models did not change; the
reference did.

**Opportunity.** One protocol, one split, one input set, four horizons.

**Answer in this study.** Experiment C, complete: 11 models × 4 horizons = 44
runs. At 15 minutes four models are catastrophically worse than persistence; at 6
hours none is.

**Testable question.** Is any architectural advantage preserved as the horizon
grows, and does skill against persistence decay monotonically?
**Answered: no. Skill *grows* from 15 minutes to 6 hours, because the reference
degrades faster than the models do.** This refuted the pre-registered hypothesis
H3.

---

## G3 — Classical baselines are reported untuned

**Evidence strength:** Demonstrated across the corpus.

The classical arm of most comparisons is a default-configuration linear model, a
default random forest or a default gradient booster. The feature-importability
papers go further and argue that tree ensembles remain competitive for structured
tabular time series.

**Why it matters.** A comparison with an untuned baseline measures the baseline.
The direction of the finding depends on which arm received the effort.

**Opportunity.** Give every family a small validation-only grid, and report the
tree arm at its tuned operating point.

**Answer in this study.** Gradient boosting is the best model at one hour
(nRMSE 0.121) and trains in 9.5 s, which is 118x less than the Transformer and
4.8x more than the linear fit that is 11% worse, while the untuned-at-effort
neural models include the two worst performers. The classical arm was tuned on
validation; the neural arm received one learning rate. That asymmetry is
disclosed as a threat to internal validity rather than hidden.

**Testable question.** Under one protocol with a tuned classical arm, do
attention-based architectures improve on gradient boosting at short horizons?
**Answered: at one hour no; at 6 hours the gap closes entirely, because the
reference has collapsed.**

---

## G4 — Persistence is under-used as a *strong* reference

**Evidence strength:** Absent from the corpus (naive persistence is ubiquitous;
the clear-sky-ratio variant is rare).

Most papers compare against naive persistence. The clear-sky-corrected variant,
which accounts for the deterministic diurnal shape of the resource, appears in
only a few, and skill against it is rarely reported.

**Why it matters.** Skill against naive persistence flatters every model. On this
data naive persistence has skill +0.028 at 15 minutes against smart persistence's
+0.028 at the same horizon but +0.194 at 6 hours, and the two references separate
by 16% in RMSE at one hour.

**Opportunity.** Two references, both reported per model per stratum, with skill
defined explicitly and evaluated on identical timestamps.

**Answer in this study.** Both references are reported everywhere. Four of eleven
models fail to beat the stronger of them at one hour.

---

## G5 — Feature importance is reported from one method, or from attention weights

**Evidence strength:** Demonstrated — attention maps are presented as explanations
in several papers, and cross-family importance comparison on one split is close to
absent.

**Why it matters.** Attention weights are internal activations, not validated
attributions. And a single ranking cannot be checked against anything. The
substantive question is not *which variable ranks first* but *whether the ranking
is the same for different models and different conditions*.

**Answer in this study.** Two problems with a single global ranking were measured
rather than asserted:

* **Across model families.** For the tree models the weather variables add
  nothing — 15 PV-history features reach nRMSE 0.119 against 0.121 for all 33 —
  while for the recurrent models the same variables are decisive (LSTM 0.176 →
  0.165, skill −0.058 → +0.008). The value of a feature group is relative to the
  architecture reading it.
* **Across conditions.** Under clear sky the trailing power *variability*
  dominates the permutation importance; under broken cloud the current power
  reading does. One global table hides both.

**Testable question.** Do tree and sequence models agree on which variables
matter, and does the answer change with weather regime and generation level?
**Answered: no, and yes.**

---

## G6 — Cross-site transfer is claimed far more often than it is validated

**Evidence strength:** Demonstrated — the corpus contains many multi-site and
spatiotemporal claims and very few cross-site *transfer* evaluations.

**Why it matters, and what this study found.** A model fitted at one installation
was worse than persistence at every other installation in a seven-station panel
(skill −0.10 to −1.07). Pooling six training sites repaired transfer for five of
the six holdouts but not for the largest array. The practical consequence is that
single-site results are *local skill*, and the gap in the literature is real
rather than pedantic.

**Opportunity.** Fixed panel, three protocols, per-site capacity normalisation,
scalers fitted on training sites only.

**Answer in this study.** Experiment J, 60 records. This is the study's most
consequential negative result, and it could not have been obtained from a
single-site benchmark.

---

## Gaps deliberately **not** claimed

Five of the twelve candidates were removed, with reasons. All five remain in
`research_gaps.json` marked `retained: false`, so the decision is auditable.

| Dropped | Role | Why |
| --- | --- | --- |
| RG-03 statistical testing absent | answered by this work | A real reporting weakness, but this project answers it rather than documenting a gap: every comparison is formally tested with a block bootstrap and a Diebold-Mariano procedure. |
| RG-04 temporal leakage | answered by this work | A genuine defect in the literature, and one this project is careful about, with chronological splits and explicit leakage tests. It is a requirement of the work, not a gap the work leaves open. |
| RG-06 uncertainty calibration | narrower than first stated | The initial claim was that calibration is rarely reported. The probabilistic PV literature is methodologically careful and does report it; the real gap is narrower — the deterministic majority reports no interval at all — and it is not this paper's contribution. |
| RG-09 reproducibility artefacts | answered by this work | The observation is correct and unusual, but it is a virtue of this repository rather than a gap in the literature that the paper sets out to close. |
| RG-10 metric inconsistency | answered by this work | Real, and this project documents every metric, but it is a matter of hygiene rather than a research gap. |

An earlier version of this list also contained "deep learning is always better for
PV forecasting" and "weather variables are under-used". Neither was ever given an
id, and both were removed outright: the corpus contains reviews concluding that
deep models dominate *and* studies concluding that tuned tree ensembles match or
beat them, so the disagreement is evidence about the literature rather than a gap
in it; and the evidence on weather points the other way, since irradiance,
temperature and calendar features are near-universal. The open question is *which
subset* matters, which is G5.

## Where this study's evidence is weakest

* The corpus is 30 papers, not a systematic review. It is a convenience sample of
  the accessible, verifiable literature, collected through DOI-verified sources;
  it is not a PRISMA search and is not claimed to be exhaustive.
* Four of the six retained gaps rest on **absence from the corpus** rather than on
  a paper demonstrating the absence. Those are labelled individually above.
* The gap list is therefore best read as a *motivation for a protocol* — a
  controlled, leakage-audited, stratified, statistically tested, cross-site
  benchmark — rather than as a claim that no such benchmark exists.
