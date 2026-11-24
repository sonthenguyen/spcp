# Changelog

## 0.5.0
Fits without the new option give the same results as 0.4.0.

New:
- `accelerate=True` (off by default) speeds up fits that converge slowly.
  After the first plain steps at the final mu, each step is extrapolated from
  the last five (Anderson acceleration); an extrapolated point that raises
  the objective is discarded, and a stop is accepted only after three plain
  steps confirm it. It switches on only when the plain steps still have a long
  way to go, so fits that are about to stop are unchanged. The solution is the
  same up to the tolerance. In our tests slow fits took 2-5 times fewer steps.
  It uses about 15 more arrays the size of M, and has no effect with
  `sigma=0`, `solver="apg"` or `svd="randomized"`. `accel_at_` is the
  iteration where it started. `sigma="auto"` and `conformal` use it for their
  fits too; with `sigma="auto"` the estimate can differ slightly from the
  plain solver's.

## 0.4.0
Fits with a numeric `sigma` give the same results as 0.3.0.

New:
- `sigma="auto"` estimates the noise level from the data: it looks for the
  value at which a fit leaves a residual with that same typical noise level.
  The value used is in `sigma_` and the search in `sigma_path_`. It costs
  several extra fits, and more data gives a more accurate estimate. It
  follows the typical size of the noise, so occasional large spikes do not
  raise it. A known value is still the better choice. `conformal` reuses the
  estimate instead of repeating it.
- `sigma=0`: no dense noise (M = L + S exactly on the observed entries),
  solved as principal component pursuit with the same `lam` and no other
  setting. Works with missing values and warm starts; `debias` does not
  apply. `sigma="auto"` switches to it when the data has no dense noise.

## 0.3.0
No change to fits: `fit` gives the same results as 0.2.0.

New:
- `conformal(windows, level, ...)`: prediction intervals for what the data
  would have been in chosen cells (a counterfactual). Fake events are planted
  in other, held-out windows, the model is refitted, and its errors there set
  the interval width (split conformal). It does not assume the low-rank model
  is right. Costs about `n_cal / per_refit` extra fits, each warm-started. By
  default the fake events copy the estimated effects of your events and are
  placed in the same columns at other rows. `n_groups` calibrates separately
  for windows of similar size, and `scale` makes windows of different size
  comparable. If the windows are missing (NaN) in the fitted data, it hides
  same-shaped observed windows instead and compares the filled-in values with
  the real ones. Coverage is approximate: it assumes your windows behave like
  the calibration windows.

## 0.2.0
Two changes in behaviour:
- `tol="auto"` is the new default. The fit stops when the remaining error is
  about 1/5000 of the total noise (sigma * sqrt(m n)), instead of at the fixed
  1e-7: stricter for precise data (never stricter than 1e-7), looser for noisy
  data. On synthetic tests this took 13-45% fewer iterations than 0.1.1 and
  changed L by about 0.0001 sigma. Pass `tol=1e-7` for the old behaviour: with
  it, results are identical to 0.1.1.
- NaN entries are now treated as missing values. 0.1.1 raised an error.

New:
- Missing values. NaN entries in M are unobserved: they are not fitted, L
  fills them in (its values there are the model's estimate), S is zero there
  and `noise_` is NaN. `missing_` holds the mask. Needs the block-coordinate
  solver (`solver="auto"` or `"bcd"`). The noise level mu is not adjusted for
  the share of missing entries, so mostly-missing data is untested.
- `debias=True` (off by default): undoes the shrinkage Stable PCP applies to
  L. The fit keeps its decisions (number of components, which entries are
  outliers) and L is refitted at full strength, ignoring clear outliers;
  `debias_rank_` is the rank kept. Options `debias_iter`,
  `debias_mask_factor`, `debias_rank_factor`. The refit is not part of the
  convex problem.
- Warm start: `fit(M, init=model.solution_)` starts from a previous solution
  (`solution_` is the solver's answer before debiasing). The answer is the
  same up to the tolerance (the problem is convex), and a refit on nearly the
  same data takes fewer iterations. A start from very different data can be
  slower than starting from scratch.
- `tol_scale` (default 1.0) multiplies the tolerance chosen by `tol="auto"`.
  Values above 1 give a rougher, faster fit, for fits repeated many times
  from a warm start; values below 1 solve more precisely. `tol_` is the
  tolerance used.

Other:
- Minimum numpy now depends on the Python version (1.23.2 for 3.11, 1.26 for
  3.12, 2.1 for 3.13, 2.3.2 for 3.14; still 1.21 below 3.11): the first numpy
  releases with ready-made wheels for each. With an older numpy, pip compiles
  it, usually without a fast linear-algebra library, and fits run many times
  slower. pip now reports a version conflict instead.
- New `docs/guide.md`: short notes on reading S, debiasing, missing values,
  warm starts and precision. The example reports clear outliers (above three
  noise levels) separately from the small noise-sized values in S.

## 0.1.1
- Lower peak memory: about 5x the size of M (was 7x) with block-coordinate
  steps and 7x (was 11x) with APG. Iterates are updated in place and
  elementwise work runs in cache-sized row blocks.
- About 10% faster per iteration on large matrices.
- Results unchanged (identical up to floating-point rounding).

## 0.1.0
- First release.
