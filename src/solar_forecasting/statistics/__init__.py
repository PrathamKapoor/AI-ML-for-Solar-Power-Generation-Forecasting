"""Statistical comparison of forecast errors.

Two questions need statistics rather than a ranking table:

1. Is the observed accuracy difference between two models larger than the
   sampling noise of a single test period? (confidence intervals)
2. Is the difference systematic rather than a lucky draw? (hypothesis tests)

The implementations here are deliberately conservative about the assumptions
they make. Photovoltaic forecast errors are strongly autocorrelated: consecutive
15-minute errors share the same cloud field. Treating them as independent
observations would make every confidence interval far too narrow and every
p-value far too small, so the bootstrap uses *moving blocks* and the
Diebold-Mariano test carries the small-sample correction. Both choices are
recorded in the returned dictionaries so the report can state them.
"""

from .comparison import (  # noqa: F401
    block_bootstrap_metric,
    block_bootstrap_skill,
    paired_loss_difference,
    diebold_mariano,
    wilcoxon_signed_rank,
    rank_biserial_effect,
    compare_models,
    ASSUMPTIONS,
)
