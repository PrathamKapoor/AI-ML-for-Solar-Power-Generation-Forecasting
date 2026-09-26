# Discussion

The purpose of this section is to say what the numbers in `paper/results.md` mean,
and equally what they do not mean.

## The most interesting result is the reference, not the winner

The headline table invites the reading "gradient boosting is best, at 6 664 W
RMSE". That is the least informative sentence in the paper. The informative one is
that **two of the eleven models are worse than holding the last observation
forward**, and that a clear-sky-corrected persistence — a rule with no fitted
parameters at all — is beaten by only a linear regression, and matched by a
Transformer that costs 1 122 seconds to train.

This is not a new observation; it is the same one the literature repeatedly reports
and repeatedly fails to foreground, because studies that compare only against a
fully fitted model have no reason to construct a strong reference. The contribution
here is the construction: the reference is guarded (a reference floor where the
clear-sky model vanishes, a ratio clip, a capacity clip) rather than naive, and it is
rescored inside every stratum, so the comparison cannot be won by choosing an
easy condition.

## Why architecture did not pay here, and when it might

LSTM and GRU underperform persistence; the Transformer underperforms gradient
boosting by 18% in RMSE; only the convolutional-recurrent hybrid is competitive.
Three mechanisms are plausible and this study cannot separate them:

1. **Data volume.** 43 745 training windows from a single 55 kW array. Recurrent
   and attention architectures have far more capacity than that supports. The tree
   models, with depth-limited boosting and strong regularisation, are operating
   closer to the data's information content.
2. **Signal structure.** The predictability here is dominated by the deterministic
   diurnal and irradiance structure, which is available in closed form (solar
   position, clear-sky reference) and is therefore learnable by a model that
   interpolates a smooth function. Recurrent models spend capacity on temporal
   dynamics that a lagged window already exposes.
3. **Optimisation budget.** A fixed budget of 70 epochs, one learning rate, one
   batch size, one seed. This is a fair-protocol constraint, not a tuning effort, and
   a neural model with per-architecture tuning might close the gap. Reporting the
   trees at a tuned operating point and the networks at a single setting is a real
   asymmetry in the comparison, and it is the asymmetry that gap G3 in the review
   identifies — now with the roles reversed.

The honest statement is therefore: **under one fixed protocol, capacity did not
pay for itself on this record.** Not "deep learning does not work for PV
forecasting", which the literature does not support and this study does not test.

## Skill is more informative than error, and more fragile

Three observations point the same way:

* In cloudy conditions, *absolute* error is **lowest** (persistence nRMSE 0.104
  against 0.185 in clear sky) because the signal amplitude is small, while *skill*
  is **lowest** (gradient boosting +0.024 against +0.358). Reading the error table
  alone would conclude that cloudy days are the easy ones.
* At 15 minutes, persistence reaches 0.097 nRMSE — better than every learned model
  except the two gradient-boosted variants — and by 6 hours it is at 0.407, worse
  than every model in the table. The reference's competence is entirely a function
  of the horizon, so a benchmark that reports one horizon has no idea whether its
  reference is strong or weak.
* Against a *naive* persistence, the learned models' skill is uniformly flattering.
  Against the *clear-sky* reference, it is uniformly smaller.

The operational implication is direct: report skill against the strongest available
reference, and report it per regime, because a single skill number averaged over
conditions is a number whose meaning depends on the mix of conditions in the test
year.

## Two procedures, two answers

The block-bootstrap intervals and the Diebold-Mariano test disagree, and both are
correct. The intervals exclude zero for eight models; the test rejects nothing. The
autocorrelation of the loss differential is the entire explanation: consecutive
15-minute errors share a cloud field, the long-run variance is far larger than an
i.i.d. estimate, and the test statistic shrinks accordingly.

This is a general problem in the energy-forecasting literature, where tests are
routinely applied to serially dependent errors and reported without qualification.
The lesson generalises beyond this dataset: **a non-significant Diebold-Mariano
result on photovoltaic data is weak evidence of equivalence, and an interval is the
more informative statistic.** A reader who saw only the p-value column would
conclude, wrongly, that the models are indistinguishable.

## The trade-off the error metric hides

The best model by RMSE is also the cheapest to train here (gradient boosting, 9.5 s,
Pareto-optimal). The best model by MAE (CNN-LSTM) is 29× more expensive for a
slightly worse normalised error. The Transformer is dominated on both axes. Under
the thread-pinned CPU protocol — the environment a small operator would actually
have — the accuracy-cost frontier contains only the tree models and the reference,
and the case for a neural architecture at this horizon has to be made on grounds
other than the error metric: interpretability, transfer to another site, or
uncertainty quantification. Two of those three are untested in this study, and
saying so is part of the finding.

## What would change the conclusions

* A second site, or the six declared holdout stations, would show whether the
  ranking is a property of these models or of this array. Cross-site transfer is the
  experiment most likely to change the conclusion, and it is not implemented.
* Completing the 6-hour and 24-hour horizons would establish whether the learned
  models' advantage at long horizons is a general property or an artefact of the
  one-horizon result.
* An information regime with numerical weather prediction would raise the ceiling for
  every model and would probably change the ranking, since a one-hour forecast
  without NWP is fundamentally limited by cloud arrival.
* Multiple seeds would establish how much of the neural models' deficit is
  initialisation rather than architecture. With one seed per configuration, that
  question is open, and it is the most obvious criticism of this design.
