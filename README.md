# spcp: Stable Principal Component Pursuit

Split a data matrix into three parts:

$$M = L + S + N$$

- **L**: low-rank structure (the patterns you care about)
- **S**: sparse, large errors (outliers, gross mistakes)
- **N**: small dense noise on every entry (measurement or rounding error)

Classical PCA handles small noise but is thrown off by outliers. Robust PCA
(Candès et al., 2011) handles outliers but assumes there is no dense noise, so
the noise gets pushed into L or S. **Stable PCP** (Zhou et al., 2010) models
both, by solving

$$\min_{L,S}\ \|L\|_* + \lambda\|S\|_1 + \frac{1}{2\mu}\|M-L-S\|_F^2,$$

where the nuclear norm $\|L\|_*$ favours low rank, the $\ell_1$ norm
$\|S\|_1$ favours sparsity, and $\mu$ is set by the noise level $\sigma$.

## Installation

```bash
pip install git+https://github.com/sonthenguyen/spcp
```

Requires Python ≥ 3.9, NumPy and SciPy.

## Usage

```python
from spcp import StablePCP

model = StablePCP(sigma=0.1)   # sigma: noise std of your data
model.fit(M)

L = model.get_low_rank()      # or model.L_
S = model.get_sparse()        # or model.S_
N = model.noise_              # M - L - S

L_lean, rank = model.trimmed_low_rank()   # drop noise-level components from L
```

`examples/example.py` runs this on synthetic data:

```
converged in 26 iterations
rank of L: 5 (after trimming noise components: 3)
error in L: 2.488%
outliers found: 973 of 988, false flags: 3180
```

## Choosing the two settings

**`sigma` (required)**: the standard deviation of the small, everyday noise,
in the units of your data. It is how much a value would differ if you could
measure the same thing twice. Ways to get it:

- the instrument's or measurement process's known precision;
- repeated measurements: `sigma ≈ std(a - b) / sqrt(2)`;
- the spread in a quiet region of the data that should contain only noise.

Too small, and noise leaks into L (high rank) and S (many tiny values).
Too large, and real signal is thrown away (L is shrunk and flattened).

**`lam_scale` (default 1.0)**: how far an entry must be from L to count as an
outlier. The cutoff is roughly `1.41 × lam_scale × sigma`. The default of 1
is the standard choice from the theory (λ = 1/√max(m, n), Candès et al., 2011).

If more than half of the entries end up in S, the fit warns you: that is not
a sparse decomposition, and `lam_scale` (or `sigma`) is too small.

## Other options

| Parameter | Default | Purpose |
|---|---|---|
| `lam`, `mu` | None | Set λ or the final μ directly instead of `lam_scale`, `sigma` |
| `svd` | `"auto"` | `"auto"`: fast Gram-matrix method with automatic precision fallback; `"exact"`: LAPACK SVD; `"randomized"`: only helps when L's rank is small relative to the matrix size |
| `max_rank` | None | Hard cap on the rank of L (heuristic, makes the problem non-convex) |
| `tol` | 1e-7 | Stopping tolerance, relative to the size of M |
| `max_iter` | 10000 | Most problems converge well before this limit |
| `solver` | `"auto"` | `"bcd"`: block-coordinate steps (usually fastest); `"apg"`: accelerated proximal gradient (robust when the dense noise is tiny compared with the outliers); `"auto"`: starts with `"bcd"` and switches to `"apg"` if it is nearly stuck |
| `continuation`, `eta`, `eta_apg` | True, 0.5, 0.9 | Start with a large μ and shrink it by `eta` (block-coordinate) or `eta_apg` (APG) per step down to the final value |

After fitting: `L_`, `S_`, `noise_`, `rank_`, `n_iter_`, `converged_`,
`sparse_fraction_`, `switched_at_`, `lam_`, `mu_`. See the docstrings for the
full list.

## Limitations

- Missing values (NaN) are not supported yet; fill or remove them first.
- Each iteration costs one eigen-decomposition of the smaller side of M, so
  very large matrices in both dimensions are slow. GPU support is planned.
- Memory use peaks at roughly 5–7 times the size of M.

## Running the tests

```bash
pip install "spcp[test] @ git+https://github.com/sonthenguyen/spcp"   # or: pip install -e ".[test]" in a clone
pytest
```

## References

- Zhou, Li, Wright, Candès, Ma. "Stable principal component pursuit." ISIT 2010.
- Candès, Li, Ma, Wright. "Robust principal component analysis?" Journal of the ACM 58(3), 2011.
- Lin, Ganesh, Wright, Wu, Chen, Ma. "Fast convex optimization algorithms for exact recovery of a corrupted low-rank matrix." CAMSAP 2009.
- Tseng. "Convergence of a block coordinate descent method for nondifferentiable minimization." J. Optim. Theory Appl. 2001.

## Acknowledgement

The interface is inspired by StablePCP in
[ShunChi100/RobustPCA](https://github.com/ShunChi100/RobustPCA) (MIT license).
The implementation is independent, based on the references above.

## License

MIT. See [LICENSE](LICENSE).
