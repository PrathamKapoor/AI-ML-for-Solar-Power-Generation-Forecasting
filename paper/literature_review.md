# Literature review

Synthesised from the 30 verified papers in `literature/literature_review.xlsx`.
This is a synthesis, not a sequence of summaries: the structure is by theme, and
where the corpus disagrees, the disagreement is preserved.

The review is organised in the seven categories the assignment prescribes
(A classical ML, B recurrent, C hybrid, D advanced architectures, E
physics-informed, F probabilistic/explainable, G systematic reviews), and the
findings below are attributed to the paper numbers in the workbook's
`Research Gaps` sheet so that every claim is traceable.

---

## 1. How the field got here, and what each step bought

The progression is not a clean replacement. Each generation of method absorbed the
previous one's strengths, and the reviews in the corpus (G category) are largely
unanimous that the practical gains between successive deep architectures on a
single site are smaller than the gains from the first step from linear models to
nonlinear ones.

**Classical and conventional ML (category A, ~5 papers).** Linear regression,
support-vector regression, random forests and gradient boosting appear throughout
the corpus, usually as the baselines against which a new deep architecture is
evaluated. The methodological weakness is consistent and identifiable: these
baselines are frequently reported at default settings. Where a paper does tune its
tabular baseline, that paper is the exception that makes the comparison meaningful,
and it is also the paper whose conclusion is least favourable to the deep model
being proposed. Gradient boosting in particular is the strongest classical arm,
and it is the arm most often left untuned.

**Recurrent models (category B, ~5 papers).** LSTM and GRU studies are
numerous, and they report the strongest version of the deep-learning claim: large
improvements over persistence and over linear models at short horizons. They also
report the version that this project reproduces — recurrent models whose advantage
over a strong reference can be small or absent on a single site. The difference
between those two reports is, on the evidence of the corpus, mostly a matter of what
the reference was and how the data were split, not of the architecture. Several
papers in this category evaluate at a single horizon, which this study suggests is
the decisive omission.

**Hybrid convolutional-recurrent models (category C, ~6 papers).** Adding a
convolutional encoder before the recurrent unit, and sometimes an attention stage
after it, is the most common way the corpus improves on plain LSTMs. The reported
gains are consistent and modest, and they are largest where the input is a
multi-station or multi-source tensor, because a convolution over neighbouring sites
or sensors is doing real work there. This project's CNN-LSTM result is consistent
with that reading: the hybrid recovers the loss of its backbone, while the
attention-only architectures do not.

**Advanced architectures (category D, ~5 papers).** Transformers, iTransformers,
PatchTST-style patch embeddings, TimesNet and graph neural networks over station
graphs are the newest strand. Two patterns are visible. First, the strongest
results in this category come from the *multi-site* and *spatiotemporal* variants,
where the architecture has structure to exploit — station graphs, cross-attention
between sites — rather than from the plain temporal Transformer. Second, the
single-site temporal Transformer is reported with gains that are modest relative to
its cost, which is the same trade-off measured directly here: the Transformer cost
118× the training time of gradient boosting for a worse RMSE.

**Physics-informed and domain-aware models (category E, ~3 papers).** Solar-position
features, clear-sky normalisation and physically constrained output layers appear
in a small strand. The reported benefit is real but modest, and it is generally
achieved by a feature rather than by an architectural constraint. This project takes
the feature-based route explicitly: solar elevation, its sine, the clear-sky
reference and the calibrated clearness index are inputs, and the value of each input
group is measured by ablation rather than asserted.

**Probabilistic, explainable and robust forecasting (category F, ~4 papers).** This
is where the methodological care in the corpus is highest and where the gap against
the deterministic majority is widest. Papers in this category report proper scoring
rules, coverage, or explicit attribution. Notably, the SHAP-based interpretability
papers tend to conclude that tree ensembles remain competitive on structured
tabular time series — the same conclusion this study reaches, from a different
protocol.

**Systematic reviews (category G, ≥2 papers).** These establish the field's
self-assessment. Their consistent finding is that deep learning has improved PV
forecasting, with the largest contributions coming from hybrid and multi-source
designs, and that the comparison bases in the primary literature are frequently
weak. One strand of the review literature is more equivocal, reporting that
classical ensembles match or beat deep models once tuned — and that strand is where
this project's hypotheses come from.

## 2. The role of meteorological variables

The literature agrees, and the agreement is near-universal, that irradiance is the
dominant input and that temperature, humidity and wind contribute secondarily.
There is no meaningful disagreement here, and this review does not manufacture one.
The interesting question, which the literature rarely asks, is *which* subset
carries the signal and how the answer changes with architecture: a tree model can
use a raw irradiance history directly, while a recurrent model may need it
re-expressed as differences and rolling statistics to extract the same information.
That is why feature-group ablation, applied to one model from each family, is a
necessary complement to the accuracy comparison rather than an optional extra.

## 3. Multi-source and multi-site forecasting

The strongest results in category D come from multi-site designs, and the practical
reason is visible in the architectures: a graph network over stations or
cross-attention between sites has genuine structure to exploit. It also explains why
single-site results in the corpus are so often modest — a single site gives an
architecture that is designed to fuse many sites nothing to fuse. This is also the
clearest methodological gap in the reproduction literature: cross-site claims are
made far more often than cross-site validation is performed. This project closes
that gap for one panel — Experiment J trains on one site and tests on six unseen
ones, and finds that transfer fails: skill against persistence runs from −0.10 to
−1.07 across the panel, and pooling six training sites repairs it for five of the
six holdouts. The negative result is itself the evidence that the gap in the
literature is a real one.

## 4. Attention, transformers and explainability

The corpus is consistent that attention improves sequence modelling. It is equally
consistent — and this is the uncomfortable part — that attention maps are presented
as explanations without validation against any perturbation. The interpretability
strand of category F takes a different and more defensible route: permutation-based
and SHAP-based attribution, each of which can be checked by re-running the model
with a variable destroyed. This project follows that route and treats attention
weights as internal activations only. The disagreement between the two communities
is not resolved by this review; it is flagged as a live methodological question.

## 5. Uncertainty

Probabilistic PV forecasting is a coherent but small strand, and it is
methodologically careful: quantile regression with pinball loss, distribution-free
intervals, hybrid parametric models. What it rarely provides is the other half of
the picture — a comparison of how much accuracy is lost, or how much width must be
paid, to obtain a calibrated interval around a *deterministic* model that is already
deployed. That is the gap this project's Experiment K addresses, and its most useful
output is a width, not a coverage: a 90% interval around a one-hour XGBoost forecast
on this plant costs roughly ±4.8 kW on a 55 kW array.

## 6. Generalisation, robustness and reproducibility

Two findings recur. First, models that perform well on the year they were trained on
degrade on the following year, and few papers report the degradation explicitly
because most evaluate on a contiguous test block inside the same distribution as the
training data. This project makes the test period a full calendar year following
the training period, which is a harder setting than most of the corpus uses, and it
is one reason the learned models' advantage here looks smaller than the literature's
typical presentation.

Second, reproducibility is uneven in a specific, actionable way: split boundaries,
seeds, preprocessing details, feature lists and package versions are reported
inconsistently, and reported metrics almost never carry uncertainty. The practical
consequence is that differences of a few percent between two papers cannot be
interpreted. This project addresses the local version of that problem — its own
results are accompanied by intervals, its assumptions are written down, and its
runs are reconstructible from machine-readable records.

## 7. Where the literature disagrees

Three disagreements are preserved here rather than resolved, because this review
has no basis for resolving them:

1. **Do deep architectures beat tuned tree ensembles on single-site PV data?**
   Some category G reviews say yes; others say the advantage disappears once the
   classical baseline is tuned, and the category F interpretability papers tend to
   side with the latter. This project finds the second, on one station, with one
   fixed protocol — evidence for one side, not a resolution.
2. **Does more input detail help?** Several papers report that adding
   meteorological or satellite-derived inputs gives large gains; others report
   marginal gains once irradiance and calendar features are present. The
   reconciliation may be site-specific (cloud types that irradiance alone cannot
   resolve) rather than resolvable in general.
3. **Is a high R² evidence of skill?** PV power is strongly diurnal, so R² is
   structurally inflated and cannot be compared across different evaluation
   windows. Several papers nevertheless report R² as a headline metric. This
   project reports it, labels it secondary, and always pairs it with a skill score
   against a persistence reference.
