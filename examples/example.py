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
# S also holds many small, noise-sized values (see docs/guide.md, "Reading S"):
# count the clear outliers, those larger than three noise levels
clear = np.abs(S) > 3 * sigma
planted = np.abs(S_true) > 3 * sigma
print(f"clear outliers found: {(clear & planted).sum()} of {planted.sum()}, "
      f"false ones: {(clear & ~outliers).sum()}")
print(f"small noise-sized values in S: {((S != 0) & ~clear).sum()} "
      f"of {S.size} entries (expected, see the guide)")

# Undo the shrinkage of L when its values matter
deb = StablePCP(sigma=sigma, debias=True).fit(M)
print(f"error in L with debias=True: "
      f"{np.linalg.norm(deb.L_ - L_true) / np.linalg.norm(L_true):.3%}")

# Missing values: hide a block of cells and read the model's values for them
M_hidden = M.copy()
M_hidden[:20, :10] = np.nan
filled = StablePCP(sigma=sigma, debias=True).fit(M_hidden, init=deb.solution_)   # warm start
err = np.sqrt(np.mean((filled.L_[:20, :10] - L_true[:20, :10]) ** 2))
print(f"hidden block: typical error of the filled-in values {err:.3f} "
      f"(noise level {sigma}), refit took {filled.n_iter_} iterations")
