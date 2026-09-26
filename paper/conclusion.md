# Conclusion

This study set out to ask how robust and generalisable different machine-learning
and deep-learning approaches are for short-term solar PV power forecasting across
horizons and weather conditions, and answered it with a controlled, leakage-audited
benchmark of eleven models on one three-year record with co-located meteorology.

**What the benchmark establishes.**

1. A clear-sky-corrected persistence forecast is a strong enough reference that two
   of eleven models — LSTM and GRU — are worse than it at a one-hour horizon, and a
   third is statistically indistinguishable from it. Comparisons made only against
   a fully fitted model, or only against naive persistence, do not surface this.
2. Under one fixed protocol, a depth-limited gradient-boosted tree model attains
   the lowest capacity-normalised error of the set while being the cheapest model
   to train, and it is the only model that is Pareto-optimal on both accuracy and
   cost. Neural capacity did not pay for itself on this record.
3. A single aggregate error misleads in a specific, reproducible way. Absolute error
   is *lowest* in cloudy conditions and *highest* in clear conditions purely because
   of signal amplitude, while skill against the reference is *lowest* in cloudy
   conditions and *highest* in clear conditions. Any conclusion about where
   forecasting is hard, drawn from error alone, is therefore inverted.
4. The reference's competence is a function of the horizon. Persistence is the best
   model in the table at 15 minutes and the worst at 6 hours. A benchmark that
   reports one horizon cannot know whether its reference is strong.
5. Statistical testing on this data requires the variance assumption to be stated.
   With serially correlated errors, a Diebold-Mariano test rejects nothing while a
   serial-correlation-aware block bootstrap separates eight of the eleven models;
   the honest report gives both, with the autocorrelation that explains them.

**What it does not establish.** One station, one climate, one information regime
without NWP, one seed per configuration, a partially completed horizon matrix, an
explainability analysis that is implemented but not run, and no cross-site
validation. Nothing here supports a claim about PV forecasting in general, and the
paper says so.

**The three experiments that would matter most next**, in order:

1. **Cross-site transfer** over the six declared holdout stations. This is the only
   experiment that can distinguish "these models suit this array" from "these
   models suit PV forecasting", and it is the most likely to change the ranking.
2. **Completion of the horizon matrix.** The persistence degradation from 15 minutes
   to 6 hours is currently measured on three models; extending it to all eleven
   would establish whether learned models' long-horizon advantage is general.
3. **Multi-seed replication** of the neural models, which is the only way to
   separate architecture from initialisation and therefore the only way to make the
   H2 verdict ("trees match or beat recurrent networks") a defensible claim rather
   than a single-seed observation.

Finally, the reproducibility apparatus is part of the contribution rather than
packaging: per-run records that carry the split dates, seed, feature list,
hyperparameters, durations and package versions; a registry rebuildable from those
records; an evaluation report that distinguishes "not run" from "no result"; and a
test suite that asserts the leakage controls at the point where they would
otherwise be silent.
