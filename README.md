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

`examples/example.py` runs this on synthetic data and prints the results.

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

The other settings have defaults that rarely need changing. After fitting,
the results are in `L_`, `S_`, `noise_` and `rank_`.
[docs/guide.md](docs/guide.md) lists every option and result and describes
each feature: reading S, trimming L, solvers, checking a fit, debiasing,
missing values, warm starts and precision.

## Future work

Planned, roughly in this order:

- **Prediction intervals** for what the data would have been in chosen cells
  (a counterfactual).
- **Automatic noise level**: estimating `sigma` from the data, and an exact
  fit for data with no dense noise.
- **A faster solver** for fits that converge slowly.
- **More cores.** The matrix decomposition in each iteration already uses
  several cores through NumPy; the element-wise steps still use one.
- **GPU support**, for matrices that are very large in both dimensions.

Memory use is several times the size of M (see the guide).

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
