"""Example: separate structure, outliers and noise in a synthetic matrix."""
import numpy as np

from spcp import StablePCP

rng = np.random.default_rng(0)
m, n, rank = 200, 100, 3

L_true = rng.standard_normal((m, rank)) @ rng.standard_normal((rank, n))
S_true = np.zeros((m, n))
outliers = rng.random((m, n)) < 0.05                  # 5% gross errors
S_true[outliers] = rng.uniform(-10, 10, outliers.sum())
sigma = 0.1                                           # dense noise level
M = L_true + S_true + sigma * rng.standard_normal((m, n))

model = StablePCP(sigma=sigma).fit(M)
L, S = model.get_low_rank(), model.get_sparse()
L_lean, r = model.trimmed_low_rank()

print(f"converged in {model.n_iter_} iterations")
print(f"rank of L: {model.rank_} (after trimming noise components: {r})")
print(f"error in L: {np.linalg.norm(L_lean - L_true) / np.linalg.norm(L_true):.3%}")
found = (S != 0) & outliers
print(f"outliers found: {found.sum()} of {outliers.sum()}, "
      f"false flags: {((S != 0) & ~outliers).sum()}")
