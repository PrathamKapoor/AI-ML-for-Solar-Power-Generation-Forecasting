# Conclusion

This study asked when additional model complexity provides meaningful forecasting
value over strong physical and statistical baselines for solar PV power, and
answered it with a controlled benchmark of 11 models over a three-year,
15-minute rooftop record with co-located meteorology: four horizons, four
weather regimes, four seasons, five additive input regimes, multi-seed
replication of every neural architecture, three cross-site protocols over a
seven-station panel, conformal intervals, and a measured account of the
statistical dependence in the errors.

## What the benchmark establishes

1. **A clear-sky-corrected persistence forecast is a strong reference, and its
   strength depends entirely on the horizon.** At 15 minutes it is competitive
   with the best learned model and four of eleven models are catastrophically
   worse than it (skill −0.53 to −0.67). At one hour three models fail to beat it.
   At 6 hours it collapses and every learned model reaches skill 0.53–0.59. A
   single-horizon benchmark cannot know whether its reference is strong, and most
   published comparisons are made at one hour.
2. **Model capacity did not pay for itself on this record.** Gradient boosting
   attains the lowest capacity-normalised error and is simultaneously the cheapest
   model to train; the Transformer costs 118× more for a worse result. The
   explanation is a combination of single-site data volume, a signal dominated by
   the deterministic diurnal shape, and a fixed protocol that gave the tree arm a
   validation grid and the neural arm a single setting. The claim is about this
   record under this protocol, not about deep learning in general.
3. **A single aggregate error misleads in a specific, reproducible way.** Absolute
   error is *lowest* under broken cloud, because the signal amplitude is small,
   while forecast skill is *also* lowest there. Any conclusion about where
   forecasting is hard, drawn from absolute error alone, is inverted.
4. **The value of an input group depends on the architecture that reads it.**
   Fifteen PV-history features alone match the full 33-input set for gradient
   boosting, while the same twelve weather variables are decisive for the
   recurrent models. Feature importance is therefore relative to a model family,
   and a single global ranking is a category error.
5. **Importance rankings change with the weather.** Under clear sky the trailing
   power variability dominates; under broken cloud the current power reading does.
6. **A model fitted at one installation does not transfer to another.** Skill
   against persistence runs from −0.10 to −1.07 across a seven-station panel
   within one campus. Pooling six training sites repairs transfer for five of the
   six holdouts, so the failure is a consequence of single-site fitting, but it
   persists on the largest array. Published single-site PV results should be read
   as local skill.
7. **Statistical treatment has to match the data.** Errors are strongly
   autocorrelated, and a Diebold-Mariano test therefore rejects nothing while a
   serial-correlation-aware moving-block bootstrap separates eight of the eleven
   models. Both are reported, with the diagnostics that explain the gap, and no
   result was adjusted towards significance.
8. **Seed spread does not change the conclusions.** Across three to five seeds the
   RMSE standard deviation is 44–253 W against a 2 900 W gap between the best and
   worst model, so the ranking is not an artefact of one initialisation.

## What it does not establish

One climate, although with a seven-station cross-site panel; no NWP inputs; three
seeds at the low end rather than five, and no transformer replication; a
cross-site panel restricted to three cheap models; a per-family tuning asymmetry
rather than a matched tuning budget; approximate conformal coverage; and
machine-specific absolute cost figures.

## The three experiments that would matter most next

1. **A second climate.** Transfer within one campus is the easy case. Whether the
   cross-site failure is a site-specificity problem or something deeper is
   unknown.
2. **A matched tuning budget per family.** The largest threat to the H2 verdict is
   that the tree arm was tuned and the neural arm was not. A study that gives each
   family the same tuning effort would turn "capacity did not pay under this
   protocol" into a defensible claim about methods rather than about a protocol.
3. **Numerical weather prediction inputs.** Without them, the reference is
   unusually strong at short horizons and the ceiling is low. The most likely
   setting in which an attention architecture pays for itself is the one this
   study cannot reach.

Finally, the reproducibility apparatus is part of the contribution rather than
packaging: per-run records carrying split dates, seed, feature list,
hyperparameters, durations, git revision and package versions; a registry
rebuildable from those records; an evaluation report that distinguishes "not run"
from "no result"; a machine-generated manuscript whose every table is rendered
from a results file; a checker that fails the build if the prose and the results
disagree; and 146 tests that assert the leakage controls at the point where they
would otherwise be silent — two of which found real defects, a stale-index mask in
the chronological split and a sign error in the persistence reference.
