# spcp guide

Notes on everything the package does, by topic. The README covers installing,
the model and choosing `sigma` and `lam_scale`. Each heading says which
version introduced the feature; CHANGELOG.md lists changes by version.

## Options and results

`sigma` and `lam_scale` are described in the README. The rest:

| Parameter | Default | Purpose |
|---|---|---|
| `lam`, `mu` | None | Set λ or the final μ directly instead of `lam_scale`, `sigma` |
| `svd` | `"auto"` | `"auto"`: fast Gram-matrix method with automatic precision fallback; `"exact"`: LAPACK SVD; `"randomized"`: only helps when L's rank is small relative to the matrix size |
| `max_rank` | None | Hard cap on the rank of L (heuristic, makes the problem non-convex) |
| `tol` | `"auto"` | Stopping tolerance: `"auto"` stops when the remaining error is about 1/5000 of the noise; or a number, relative to the size of M |
| `tol_scale` | 1.0 | Multiplies the `"auto"` tolerance: above 1 is rougher and faster, below 1 more precise |
| `max_iter` | 10000 | Most problems converge well before this limit |
| `solver` | `"auto"` | `"bcd"`: block-coordinate steps (usually fastest); `"apg"`: accelerated proximal gradient (robust when the dense noise is tiny compared with the outliers); `"auto"`: starts with `"bcd"` and switches to `"apg"` if it is nearly stuck |
| `debias`, `debias_mask_factor` | False, 4.0 | Undo the shrinkage of L (see Debiasing) |
| `continuation`, `eta`, `eta_apg` | True, 0.5, 0.9 | Start with a large μ and shrink it by `eta` (block-coordinate) or `eta_apg` (APG) per step down to the final value |

After fitting: `L_`, `S_`, `noise_`, `rank_`, `n_iter_`, `converged_`,
`sparse_fraction_`, `switched_at_`, `debias_rank_`, `solution_`, `missing_`,
`tol_`, `lam_`, `mu_`. The docstrings have the full list.

## Reading S (0.1.0)

An entry goes into S when it lies more than about `1.41 × lam_scale × sigma`
from L. Ordinary noise crosses that line too: with the default `lam_scale=1`,
roughly one normal entry in six ends up in S with a small value, about the
size of the noise. That is expected, not a wrong detection.

For a short list of clear outliers, either keep the entries of S above a size
you care about (for example `abs(S) > 3 * sigma`), or raise `lam_scale`, which
moves the line itself and finds slightly fewer real outliers.

## Trimming noise components from L (0.1.0)

Stable PCP shrinks every component of L by the same amount, so a few small
noise-level components usually survive next to the real ones, and `rank_`
counts them too. `model.trimmed_low_rank()` returns L without them, and the
rank that is left: it keeps the components larger than `factor × mu_`
(default factor 0.5), just above what pure noise can reach after shrinkage.

`trim_rank(L, rank=..., threshold=...)` does the same for any matrix, with a
rank or a threshold you choose. With neither, it cuts at the largest gap
between consecutive components, which can cut too early when the data has a
large common offset; prefer `trimmed_low_rank()` in that case.

## Solvers (0.1.0)

`solver="auto"` (the default) starts with block-coordinate steps, which are
usually fastest, and switches to accelerated proximal gradient (APG) if
progress is nearly stuck, which happens when the dense noise is tiny compared
with the outliers. Both reach the same solution. `solver="bcd"` or `"apg"`
fixes the method; `switched_at_` tells you whether and when a switch happened.

`svd="auto"` (the default) does the low-rank step through the smaller of the
two Gram matrices, with an automatic fallback to a full SVD when that would
lose precision (`n_svd_fallback_` counts those steps). `"exact"` always uses
the full SVD. `"randomized"` only pays off when the rank of L is small
relative to the size of the matrix.

`max_rank` caps the rank of L. It is a heuristic: with it the problem is no
longer convex, so leave it unset for the exact Stable PCP solution.

## Checking a fit (0.1.0)

- `converged_`, `n_iter_`: whether the tolerance was reached, and in how many
  iterations. A fit that stops at `max_iter` warns.
- `dist_est_`: the estimated remaining distance to the solution, relative to
  the size of M.
- `sparse_fraction_`: the share of entries with a nonzero S. Above one half
  the fit warns: that is not a sparse decomposition, and `lam_scale` or
  `sigma` is too small for the data.
- `lam_`, `mu_`: the two parameters actually used.

The input is never modified. Lists and other number types are accepted, and
multiplying the data and `sigma` by the same factor multiplies L and S by it.

## Memory and speed (0.1.1)

Each iteration costs one eigen-decomposition of the smaller side of M, so
matrices that are very large in both dimensions are slow. Peak memory is
about 5 times the size of M with block-coordinate steps and 7 times with APG.

The matrix products and decompositions run on several cores when NumPy's
linear-algebra library does (the usual case for NumPy installed from wheels).
The element-wise steps run on one core, so the gain is largest when both
sides of M are large.

## Debiasing, `debias=True` (0.2.0)

To keep L low-rank, Stable PCP shrinks it slightly. The shrinkage hits the
largest values hardest: L comes out a bit too small, and in the busiest rows or
columns the shortfall can show up in S as false outliers.

`StablePCP(sigma=..., debias=True)` keeps the method's decisions (how many
components, which entries are outliers) and refits L at full strength,
ignoring clear outliers. Use it when the *values* of L or S matter, not only
which entries are flagged. To measure the size of moderate events, also set
`debias_mask_factor=0`.

The refit is not part of the convex problem, so it has no uniqueness
guarantee. `debias_rank_` is the rank it kept.

## Missing values (0.2.0)

NaN entries in M are treated as unobserved. They are not fitted: L fills them
in (its values there are the model's estimate), S is zero there and `noise_`
is NaN. `missing_` holds the mask.

```python
M[rows, cols] = np.nan            # hide cells
model = StablePCP(sigma=sigma).fit(M)
model.L_[rows, cols]              # the model's values for the hidden cells
```

Missing values need the block-coordinate solver (`solver="auto"` or `"bcd"`).
The noise level is not adjusted for the share that is missing, so data that
is mostly missing is untested.

## Refitting similar data, warm start (0.2.0)

A fit can start from a previous solution. The answer is the same up to the
tolerance, and when the data changed only a little it takes fewer iterations:

```python
first = StablePCP(sigma=sigma).fit(M)
again = StablePCP(sigma=sigma).fit(M_changed, init=first.solution_)
```

`solution_` is the solver's answer before debiasing. A start from very
different data can be slower than starting from scratch.

## Precision, `tol` and `tol_scale` (`tol` 0.1.0; `"auto"` and `tol_scale` 0.2.0)

`tol="auto"` (the default) stops when the remaining error is about 1/5000 of
the total noise: noisy data stops sooner, precise data is solved more
precisely, down to 1e-7. A number sets the tolerance directly, relative to
the size of M; `tol=1e-7` reproduces versions before 0.2.0.

`tol_scale` multiplies the automatic tolerance. Above 1 gives a rougher,
faster fit, which suits fits repeated many times from a warm start; below 1
solves more precisely. `tol_` is the tolerance that was used.
