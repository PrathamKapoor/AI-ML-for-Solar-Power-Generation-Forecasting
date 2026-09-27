# Discussion

The purpose of this section is to say what the numbers mean, and equally what
they do not.

## The most important result is not about any model

The headline table invites the reading "gradient boosting is best, at 6 664 W
RMSE". That is the least informative sentence available. The informative one is
in the horizon and cross-site tables.

**At a one-hour horizon, four of eleven models fail to beat a rule with no fitted
parameters.** At 15 minutes the failure is far worse — LSTM and GRU have skill of
−0.65 and −0.67 against persistence, which means they are roughly two-thirds
worse than repeating the last measurement. At 6 hours the same models reach skill
+0.53. The models did not change; the reference did. A benchmark that evaluates at
one horizon, which is what most of the literature does, cannot know whether its
reference is strong or weak, and therefore cannot know whether a 20% RMSE
improvement means anything.

**A model fitted at one installation does not transfer to another.** Skill against
persistence runs from −0.10 to −1.07 across a seven-station panel within a single
campus. Pooling six training sites repairs transfer for five of the six holdouts,
so the failure is a consequence of single-site fitting rather than of the model
family — and it still fails on the largest array. Taken together, these two
results say that most published single-site PV results are *local* skill: a number
about a plant, not about a method.

## Why capacity did not pay, and when it might

At one hour, gradient boosting attains the lowest error and the shortest training
time, and the Transformer costs 118× more for a worse result. Three mechanisms
are plausible and this study cannot separate them:

1. **Data volume.** 43 745 training windows from a single 55 kW array. Recurrent
   and attention architectures carry more capacity than that supports. The
   tree models, with depth-limited boosting and strong regularisation, sit closer
   to the data's information content.
2. **Signal structure.** The predictability here is dominated by the deterministic
   diurnal and irradiance structure, which is available in closed form and is
   therefore learnable by a model that interpolates a smooth function. The input
   regime study supports this directly: fifteen PV-history features alone give
   gradient boosting nRMSE 0.119, *better* than all 33 inputs.
3. **Optimisation budget.** A fixed protocol — 70 epochs, one learning rate, one
   batch size, one seed per family outside the multi-seed study — is a
   fairness constraint, not a tuning effort. The trees received a small
   validation grid and the neural models received a single setting. This
   asymmetry is a real limitation and is stated as a threat to internal validity.

The defensible statement is therefore: **under one fixed protocol, capacity did
not pay for itself on this record.** Not "deep learning does not work for PV
forecasting", which the literature does not support and this study does not test.

## The same information is worth different amounts to different models

The additive input-regime study produced the second-most useful result, and it is
one the leave-one-out ablation cannot show. For the tree models, the twelve
weather variables add nothing: PV history alone matches the full input set.
For the recurrent models, the same weather variables are decisive, taking LSTM
from negative skill to slightly positive. Weather alone is worse than persistence
for every family, so neither signal is redundant.

The generalisable point is that **feature importance is relative to the
architecture**. A single global importance ranking, whether from SHAP or from
permutation, describes one model on one split, and quoting it as the ranking of
"the inputs" is a category error. The regime-specific analysis sharpens this
further: under clear sky the trailing power *variability* dominates, while under
broken cloud the current power reading does. One global table would have hidden
that.

## Two procedures, two answers

The block-bootstrap intervals and the Diebold-Mariano test disagree, and both are
correct. The intervals exclude zero for eight models; the test rejects nothing. The
autocorrelation of the loss differential is the entire explanation: consecutive
15-minute errors share a cloud field, the long-run variance is far larger than an
i.i.d. estimate, and the test statistic shrinks accordingly.

This is a general problem in the energy-forecasting literature, where tests are
routinely applied to serially dependent errors and reported without
qualification. A non-significant Diebold-Mariano result on photovoltaic data is
weak evidence of equivalence, and an interval is the more informative statistic. A
reader shown only the p-value column would conclude, wrongly, that the models are
indistinguishable. Both are reported, together with the diagnostics that explain
the gap.

## The trade-off the error metric hides

The best model by RMSE is also the cheapest to train, so on this record the
accuracy-cost frontier contains only the tree models and the reference. The case
for a neural architecture at this horizon would have to be made on other grounds:
uncertainty quantification, transfer, or a regime the tree models cannot reach.
The present study finds none of those: conformal intervals are as usable around
the tree model, and transfer fails for every family tested. That is an honest
negative result and it is reported as one.

## What would change the conclusions

* **A second climate.** The cross-site panel is one campus. Transfer across
  climates is a different and easier problem than transfer across arrays, and it
  is untested here.
* **Numerical weather prediction inputs.** Under a strictly historical information
  set a one-hour forecast cannot anticipate a cloud front, which bounds the
  ceiling and makes persistence unusually strong. The ranking of model families
  would probably change with forecast-weather available.
* **Per-family tuning.** The asymmetry between a tuned tree arm and a
  single-setting neural arm is the most obvious criticism of this design, and a
  fair-budget study might close part of the gap.
* **Cross-site transfer for the neural models.** Only three cheap models were
  carried through the 20 fits per model that the transfer protocol requires. A
  neural model might transfer better or worse; it is not known here.
