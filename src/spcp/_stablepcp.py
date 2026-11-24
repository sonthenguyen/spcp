"""
Stable Principal Component Pursuit (Stable PCP).

Decomposes an observed matrix

    M = L + S + N

L : low-rank structure
S : sparse, large errors (outliers)
N : small dense noise with standard deviation `sigma`

by solving

    min_{L,S}  ||L||_*  +  lam * ||S||_1  +  1/(2 mu) * ||M - L - S||_F^2

using exact block-coordinate minimisation: alternately

    L <- SVT(M - S, mu)         (singular value thresholding)
    S <- soft(M - L, lam * mu)  (elementwise soft thresholding)

each of which solves its sub-problem exactly. The smooth part is a convex
quadratic and the non-smooth part is separable in (L, S), so the iterates
converge to the global minimiser (Tseng 2001). With continuation on mu
(start large, decrease geometrically) this is usually fast.

Block-coordinate steps can crawl, e.g. when the dense noise is tiny compared
with the outliers. The default solver="auto" detects this from the
contraction rate and switches, from the current point, to the accelerated
proximal gradient method (APG) of Lin et al. (2009), which handles that case
well. Both methods converge to the same solution.

Parameter choice (Zhou, Li, Wright, Candes & Ma, 2010), n_max = max(m, n):

    mu_final = sqrt(2 * n_max) * sigma
    lam      = lam_scale / sqrt(n_max)

so an entry is classified as an outlier when its deviation from L exceeds
about  lam * mu_final = sqrt(2) * lam_scale * sigma  (in the data's units).

References
----------
Zhou, Li, Wright, Candes, Ma. "Stable principal component pursuit." ISIT 2010.
Lin, Ganesh, Wright, Wu, Chen, Ma. "Fast convex optimization algorithms for
    exact recovery of a corrupted low-rank matrix." CAMSAP 2009.
Tseng. "Convergence of a block coordinate descent method for
    nondifferentiable minimization." J. Optim. Theory Appl. 2001.
Halko, Martinsson, Tropp. "Finding structure with randomness." SIAM Review 2011.

Acknowledgement: the interface is inspired by StablePCP in ShunChi100/RobustPCA
(https://github.com/ShunChi100/RobustPCA, MIT license). The implementation is
independent, based on the references above.
"""
import warnings

import numpy as np
from scipy.linalg.blas import daxpy as _axpy
from scipy.sparse.linalg import svds

__all__ = ["StablePCP", "trim_rank"]

# accelerate=True: the stopping estimate after extrapolated steps runs low
# (measured: up to about 2x), so it is doubled before it is compared with tol
_AA_SAFETY = 2.0
# accelerate=True switches on only when plain steps are expected to need more
# than this many further steps (below that, it costs more than it saves)
_AA_MIN_STEPS = 10


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
# Elementwise work is done in row blocks of about this many entries (small
# enough to stay in CPU cache), so the
# temporary arrays it needs stay small whatever the size of the matrix.
_BLOCK = 1 << 16


def _row_blocks(X):
    rows = max(1, _BLOCK // max(1, X.shape[1]))
    for i in range(0, X.shape[0], rows):
        yield slice(i, i + rows)


def _soft(X, t, out=None):
    """Elementwise soft thresholding: sign(x) * max(|x| - t, 0).
    out=X overwrites X in place; temporaries are limited to one row block."""
    if out is None:
        out = np.empty_like(X)
    for sl in _row_blocks(X):
        x = X[sl]
        mag = np.abs(x)
        mag -= t
        np.maximum(mag, 0.0, out=mag)
        np.copysign(mag, x, out=out[sl])
    return out


def _sqdist(A, B):
    """||A - B||_F^2 without a full-size temporary."""
    total = 0.0
    for sl in _row_blocks(A):
        d = A[sl] - B[sl]
        total += float(np.vdot(d, d))
    return total


def _randomized_svd(X, k, n_iter, oversample, rng):
    """Top-k SVD by randomized range finding with power iterations
    (Halko et al. 2011, Alg. 4.4 with QR re-orthonormalisation)."""
    m, n = X.shape
    p = min(k + oversample, min(m, n))
    Q, _ = np.linalg.qr(X @ rng.standard_normal((n, p)))
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(X.T @ Q)
        Q, _ = np.linalg.qr(X @ Q)
    Ub, s, Vt = np.linalg.svd(Q.T @ X, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


# Gram route loses precision when s_max / thr is huge (squaring doubles the
# condition number). Error in a singular value near thr is about
# eps * s_max^2 / thr, so require eps * (s_max/thr)^2 <= _GRAM_RTOL.
_GRAM_RTOL = 1e-8


def _svt_gram(X, thr, cap):
    """Singular value thresholding via the Gram matrix of the smaller side.

    With A the tall orientation of X (m >= n), A^T A = V diag(s^2) V^T and

        SVT(A, thr) = A V diag(1 - thr/s) V^T      (over s > thr),

    so no left singular vectors are needed. One n x n product plus one
    symmetric eigendecomposition: typically several times faster than a full
    SVD. Returns None if precision would be lost or values are not finite.
    """
    tall = X.shape[0] >= X.shape[1]
    A = X if tall else X.T
    w, V = np.linalg.eigh(A.T @ A)              # ascending eigenvalues
    w_max = w[-1]
    if not np.isfinite(w_max):
        return None
    if w_max <= thr * thr:
        _svt_gram.nuc = 0.0
        return np.zeros_like(X), 0
    ratio = np.sqrt(w_max) / thr                # s_max / thr, no thr^2 underflow
    if np.finfo(float).eps * ratio * ratio > _GRAM_RTOL:
        return None
    k = int(np.count_nonzero(w > thr * thr))
    k = min(k, cap)
    V = V[:, -k:]                               # top-k eigenvectors
    s = np.sqrt(w[-k:])
    _svt_gram.nuc = float(np.sum(s - thr))
    if k <= A.shape[1] // 2:
        L = (A @ V * (1.0 - thr / s)) @ V.T     # m x k intermediate (small k)
    else:
        L = A @ ((V * (1.0 - thr / s)) @ V.T)   # n x n intermediate only
    return (L if tall else L.T), k


def _rank_r(X, r):
    """Best rank-r approximation of X (no shrinkage), via the smaller Gram matrix."""
    tall = X.shape[0] >= X.shape[1]
    A = X if tall else X.T
    if r <= 0:
        return np.zeros_like(X)
    w, V = np.linalg.eigh(A.T @ A)
    V = V[:, -r:]
    L = (A @ V) @ V.T
    return L if tall else L.T


def _spectral_norm(X):
    """Largest singular value via the smaller Gram matrix (eigenvalues only).
    Used only for the continuation start value, so Gram precision is ample."""
    A = X if X.shape[0] >= X.shape[1] else X.T
    return float(np.sqrt(max(np.linalg.eigvalsh(A.T @ A)[-1], 0.0)))


def trim_rank(L, rank=None, threshold=None):
    """Remove the small noise-level components that Stable PCP leaves in L.

    rank      : keep exactly this many components, or
    threshold : keep components whose singular value exceeds it, or
    neither   : cut at the largest ratio between consecutive singular values.
                Caution: if the data has a large common offset (nonzero
                mean) the first component dominates and this cuts too
                early; prefer StablePCP.trimmed_low_rank(), which uses the
                noise level.
    Returns (L_trimmed, rank).
    """
    U, s, Vt = np.linalg.svd(L, full_matrices=False)
    s_pos = s[s > s[0] * 1e-12] if s.size and s[0] > 0 else s[:0]
    if rank is None and threshold is not None:
        rank = int(np.sum(s_pos > threshold))
    if rank is None:
        if s_pos.size <= 1:
            rank = s_pos.size
        else:
            rank = int(np.argmax(s_pos[:-1] / s_pos[1:])) + 1
    return (U[:, :rank] * s[:rank]) @ Vt[:rank], rank


# ----------------------------------------------------------------------------
# estimator
# ----------------------------------------------------------------------------
class StablePCP:
    """Stable PCP: M = L (low rank) + S (sparse) + N (dense noise).

    Parameters
    ----------
    sigma : float >= 0 or "auto", required
        Standard deviation of the dense noise N, in the units of M.
        Sets the final shrinkage level. (Alternatively pass `mu` directly.)
        A value you know (instrument precision, repeated measurements, a
        variance-stabilising transform) is the best choice.
        "auto" estimates it from the data: fit, measure the typical noise
        level of the residual M - L, and repeat until the value used and the
        value measured agree within 2% (see `sigma_`, `sigma_path_`). The
        noise level is measured with a second Stable PCP fit on
        log|residual|, which turns noise of varying size into an even-noise
        problem with a known sigma; its median over the entries is the
        estimate. This costs several extra fits. More data gives a more
        accurate estimate. It follows the typical size of the noise, so
        occasional large spikes in the noise do not raise it.
        sigma=0 means there is no dense noise (M = L + S exactly, as in
        compressed video where most values repeat exactly). That case is
        solved as principal component pursuit: minimise ||L||_* + lam ||S||_1
        subject to L + S = M on the observed entries, with the same lam and no
        other setting. `debias` does not apply and noise_ is zero where
        observed. "auto" switches to sigma=0 when the data has no dense noise:
        the leftover keeps shrinking with the value tried, so no positive
        value can match it.
    lam_scale : float, default 1.0
        Multiplier on the default sparsity weight. Larger -> fewer outliers.
        Outlier cutoff is roughly sqrt(2) * lam_scale * sigma.
    lam : float, optional
        Sparsity weight used as-is (overrides lam_scale).
    mu : float, optional
        Final penalty parameter used as-is (overrides sigma).
    continuation : bool, default True
        Start with a large mu and shrink it geometrically by `eta` down to the
        final value. Usually faster; set False to use the final mu throughout.
    eta : float in (0, 1), default 0.5
        Continuation decay factor for block-coordinate steps.
    solver : {"auto", "bcd", "apg"}, default "auto"
        "bcd": block-coordinate minimisation. "apg": accelerated proximal
        gradient. "auto": start with "bcd" and switch to "apg" when progress
        is slow (see switch_rate).
    eta_apg : float in (0, 1), default 0.9
        Continuation decay factor for APG, which needs a slower decrease.
    switch_rate : float in (0, 1), default 0.995
        solver="auto" switches when, after 5 steps at the final mu, the
        median per-step contraction of the change exceeds this value.
    max_rank : int, optional
        Hard cap on the rank of L. NOTE: this turns the problem non-convex
        (a heuristic); leave None for the exact Stable PCP solution.
    svd : {"auto", "exact", "randomized"}, default "auto"
        "auto": thresholding through the eigendecomposition of the smaller
        Gram matrix (X^T X or X X^T), typically several times faster than a
        full SVD with the same result. If the largest singular value is so much bigger
        than the threshold that squaring would cost precision
        (eps * (s_max/thr)^2 > 1e-8, e.g. a huge constant offset with tiny
        noise), that step falls back to the LAPACK SVD automatically.
        "exact": always the LAPACK SVD (gesdd).
        "randomized" uses an adaptive randomized partial SVD for the bulk of
        the iterations and an accurate partial SVD (ARPACK) to finish, so
        convergence to `tol` is still reached. It only pays off when the rank
        of L is small relative to min(M.shape), e.g. with a large sigma or
        max_rank. With the sigma-based threshold, part of the noise-level
        components usually survive, so "auto" is usually faster.
    tol : "auto" or float, default "auto"
        "auto" stops when the remaining error is about 1/5000 of the total
        noise (sigma * sqrt(m n)): noisy data stops sooner, precise data is
        solved more precisely, down to 1e-7. In tests the solver's error
        stayed below about 1% of the estimation error of L. Needs sigma (or mu); with lam and mu
        given directly, sigma is taken from mu. A number:
        Stop when the estimated distance of (L, S) to the solution, relative
        to ||M||_F, falls below tol, once mu has reached its final value.
        The distance is estimated as change / (1 - rate), with `rate` the
        recent contraction factor, so slow-converging problems are not
        stopped early. When L is much smaller than M, the relative error of
        L itself is larger than tol.
    accelerate : bool, default False
        Speed up fits that converge slowly. After the first few plain
        block-coordinate steps at the final mu, each step is extrapolated from
        the last five (Anderson acceleration). An extrapolated point that
        raises the objective is discarded for the plain step, and a stop is
        accepted only after three plain steps confirm it. It switches on only
        when the plain steps still have a long way to go, so fits that are
        about to stop are unchanged. The solution is the same up to the
        tolerance. With sigma="auto" the estimate can differ slightly from
        the plain solver's. Uses about 15 more arrays the size of M.
        Has no effect with sigma=0, solver="apg" (or after solver="auto"
        switches to it) or svd="randomized". `accel_at_` is the iteration
        where it started (None if it did not).
    tol_scale : float, default 1.0
        Multiplies the tolerance chosen by tol="auto" (no effect with a
        numeric tol). Values above 1 stop sooner: for example 100 gives a
        rougher, faster fit, useful when a fit is repeated many times from a
        warm start and only needs to follow the data. Values below 1 solve
        more precisely. The tolerance never goes below 1e-7.
    max_iter : int, default 10000
        Most problems converge well before this limit. Settings where most
        entries end up in S (lam_scale far too small for the data) converge
        slowly and trigger a warning (sparse_fraction_).
    random_state : int or None
        Seed for the randomized SVD.
    verbose : bool, default False
    debias : bool, default False
        Undo the shrinkage that Stable PCP applies to L. The fit keeps its
        decisions (how many components, which entries are outliers) but L is
        refitted at full strength, ignoring the outlier entries, and S is
        recomputed. Useful when the values of L or S matter (effect sizes,
        counterfactuals, cleaned data): without it, L is biased toward zero,
        most at the largest values, and that bias can appear in S as false
        outliers. Note: the refit is not part of the convex problem.
    debias_iter : int, default 30
        Maximum refit rounds (stops early once the outlier set is stable).
    debias_mask_factor : float, default 4.0
        During debiasing, outliers larger than debias_mask_factor x sigma
        (beyond the cutoff) are ignored when refitting L. The default masks
        only clear outliers and gave the most accurate L and the fewest false
        outliers in our tests. Use 0 (mask every flagged entry) when you need
        the *size of moderate events*, e.g. the effect of street closures on
        quiet areas: it estimated such effects much better in our tests, at
        the cost of keeping some false outliers in the busiest rows/columns.
    debias_rank_factor : float or None, default None
        Components of the Stable PCP solution with singular value above
        debias_rank_factor * mu_ are kept. None chooses it from the matrix
        shape: just above the largest value pure noise can reach after
        shrinkage (about 0.5 for square matrices, 0 for tall or wide ones,
        where noise does not survive). Assumes sigma matches the noise.
    rsvd_n_iter, rsvd_oversample, rsvd_margin, rsvd_init_ratio, polish_tol
        Tuning for svd="randomized" only: power iterations, extra samples,
        rank margin, initial rank guess (fraction of min(M.shape)), and the
        change level at which it switches to the accurate partial SVD.

    Attributes (after fit)
    ----------------------
    L_, S_ : ndarray           low-rank and sparse components (full shape)
    noise_ : ndarray           M - L - S
    rank_ : int                rank of L_
    n_iter_ : int
    converged_ : bool
    errors_ : list of float    relative change per iteration
    dist_est_ : float          final estimated relative distance to solution
    sparse_fraction_ : float   share of entries with nonzero S (warns if > 0.5)
    n_svd_fallback_ : int      iterations where svd='auto' fell back to LAPACK
    switched_at_ : int or None iteration where solver='auto' switched to APG
    lam_, mu_, mu_init_ : float  parameters actually used
    tol_ : float               stopping tolerance actually used
    sigma_ : float or None     noise level used: `sigma`, or the estimate when
                               sigma="auto" (0 = no dense noise; None if only
                               mu was given)
    sigma_path_ : list or None with sigma="auto": (value tried, value measured)
                               for each round
    accel_at_ : int or None    iteration where accelerate=True started
                               extrapolating (None if it did not)
    solution_ : (L, S)         the fit before debiasing (L_, S_ when
                               debias=False): pass it as `init` to warm-start
                               a fit of similar data
    missing_ : ndarray or None boolean mask of the missing (NaN) entries
    debias_rank_ : int or None rank kept by debias (None if debias=False)

    Methods (after fit)
    -------------------
    conformal(windows, ...)    prediction intervals for what the data would
                               have been in chosen cells
    """

    def __init__(self, sigma=None, lam_scale=1.0, lam=None, mu=None,
                 continuation=True, eta=0.5, max_rank=None, svd="auto",
                 solver="auto", eta_apg=0.9, switch_rate=0.995,
                 tol="auto", max_iter=10000, random_state=None, verbose=False,
                 # randomized-SVD tuning (rarely needed)
                 rsvd_n_iter=10, rsvd_oversample=10, rsvd_margin=5,
                 rsvd_init_ratio=0.2, polish_tol=1e-3,
                 debias=False, debias_iter=30, debias_rank_factor=None,
                 debias_mask_factor=4.0, tol_scale=1.0, accelerate=False):
        self.sigma = sigma
        self.lam_scale = lam_scale
        self.lam = lam
        self.mu = mu
        self.continuation = continuation
        self.eta = eta
        self.max_rank = max_rank
        self.svd = svd
        self.solver = solver
        self.switch_rate = switch_rate
        self.eta_apg = eta_apg
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.rsvd_n_iter = rsvd_n_iter
        self.rsvd_oversample = rsvd_oversample
        self.rsvd_margin = rsvd_margin
        self.rsvd_init_ratio = rsvd_init_ratio
        self.polish_tol = polish_tol
        self.debias = debias
        self.debias_iter = debias_iter
        self.debias_rank_factor = debias_rank_factor
        self.debias_mask_factor = debias_mask_factor
        self.tol_scale = tol_scale
        self.accelerate = accelerate

    # ------------------------------------------------------------------
    def _check_params(self):
        if self.mu is None:
            if self.sigma is None:
                raise ValueError(
                    "sigma (std of the dense noise) is required; results depend "
                    "strongly on it. Alternatively pass mu directly.")
            if isinstance(self.sigma, str):
                if self.sigma != "auto":
                    raise ValueError("sigma must be a number >= 0 or 'auto'")
            elif not self.sigma >= 0:
                raise ValueError("sigma must be >= 0")
        else:
            if not self.mu > 0:
                raise ValueError("mu must be > 0")
            if isinstance(self.sigma, str) and self.sigma != "auto":
                raise ValueError("sigma must be a number >= 0 or 'auto'")
        if self.lam is not None and not self.lam > 0:
            raise ValueError("lam must be > 0")
        if not self.lam_scale > 0:
            raise ValueError("lam_scale must be > 0")
        if not 0 < self.eta < 1:
            raise ValueError("eta must be in (0, 1)")
        if self.solver not in ("auto", "bcd", "apg"):
            raise ValueError("solver must be 'auto', 'bcd' or 'apg'")
        if not 0 < self.eta_apg < 1:
            raise ValueError("eta_apg must be in (0, 1)")
        if not 0 < self.switch_rate < 1:
            raise ValueError("switch_rate must be in (0, 1)")
        if self.svd not in ("auto", "exact", "randomized"):
            raise ValueError("svd must be 'auto', 'exact' or 'randomized'")
        if self.max_rank is not None and int(self.max_rank) < 1:
            raise ValueError("max_rank must be a positive integer")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be >= 1")
        if isinstance(self.tol, str):
            if self.tol != "auto":
                raise ValueError("tol must be a number >= 0 or 'auto'")
            if self.sigma is None and self.mu is None:
                raise ValueError("tol='auto' needs sigma (or mu)")
        elif not self.tol >= 0:
            raise ValueError("tol must be >= 0")
        if not self.tol_scale > 0:
            raise ValueError("tol_scale must be > 0")
        if not self.polish_tol > 0:
            raise ValueError("polish_tol must be > 0")
        if int(self.rsvd_n_iter) < 0 or int(self.rsvd_oversample) < 0 \
                or int(self.rsvd_margin) < 0:
            raise ValueError("rsvd_n_iter, rsvd_oversample and rsvd_margin "
                             "must be >= 0")
        if not 0 < self.rsvd_init_ratio <= 1:
            raise ValueError("rsvd_init_ratio must be in (0, 1]")
        if int(self.debias_iter) < 1:
            raise ValueError("debias_iter must be >= 1")
        if self.debias_rank_factor is not None and not self.debias_rank_factor >= 0:
            raise ValueError("debias_rank_factor must be >= 0 or None")
        if not self.debias_mask_factor >= 0:
            raise ValueError("debias_mask_factor must be >= 0")

    # ------------------------------------------------------------------
    def _fit_pcp(self, M, miss, lam, init):
        """sigma = 0: principal component pursuit by the inexact augmented
        Lagrangian method (Lin, Chen and Ma, 2010):

            minimise ||L||_* + lam ||S||_1   subject to  L + S = M (observed).

        Missing entries are filled from L at each step (S is zero there), so
        the constraint only binds where M is observed. M has its missing
        entries set to 0 on entry.
        """
        scale = float(np.max(np.abs(M)))
        if scale == 0.0:
            scale = 1.0
        M = M / scale
        has_miss = miss is not None
        normM = max(np.linalg.norm(M), np.finfo(float).tiny)
        tol = 1e-7 if isinstance(self.tol, str) else float(self.tol)
        self.tol_ = tol
        self.lam_, self.mu_, self.mu_init_ = lam, 0.0, 0.0
        self._rng = np.random.default_rng(self.random_state)
        self._k_pred = max(1, int(self.rsvd_init_ratio * min(M.shape)))
        self._n_gram = self._n_fallback = 0
        if init is not None:
            L = np.array(init[0], dtype=float) / scale
            S = np.array(init[1], dtype=float) / scale
            if L.shape != M.shape or S.shape != M.shape:
                raise ValueError("init must be (L, S) with the shape of M")
        else:
            L = np.zeros_like(M)
            S = np.zeros_like(M)
        if has_miss:
            M[miss] = L[miss]
        norm2 = max(_spectral_norm(M), np.finfo(float).tiny)
        Y = M / max(norm2, np.max(np.abs(M)) / lam)    # dual start (their eq. 10)
        mu = 1.25 / norm2
        mu_max = mu * 1e7
        rho = 1.5
        self.errors_ = []
        self.converged_ = False
        self.switched_at_ = None
        self.accel_at_ = None                      # accelerate does not apply here
        rank = 0
        for it in range(1, int(self.max_iter) + 1):
            if has_miss:
                M[miss] = L[miss] + S[miss]            # unobserved: no constraint
            L, rank = self._svt(M - S + Y / mu, 1.0 / mu, False, True)
            S = M - L + Y / mu
            _soft(S, lam / mu, out=S)
            if has_miss:
                np.copyto(S, 0.0, where=miss)
            Z = M - L - S
            if has_miss:
                np.copyto(Z, 0.0, where=miss)
            Y += mu * Z
            mu = min(mu * rho, mu_max)
            err = float(np.linalg.norm(Z)) / normM
            self.errors_.append(err)
            if self.verbose and (it % 10 == 0 or it == 1):
                print(f"iter {it:5d}  pcp  rank {rank:4d}  "
                      f"nnz(S) {np.count_nonzero(S):8d}  residual {err:.2e}")
            if err <= tol:
                self.converged_ = True
                break
        self.n_iter_ = it
        self.dist_est_ = err
        if not self.converged_:
            warnings.warn(f"StablePCP (sigma=0) did not converge in "
                          f"{self.max_iter} iterations (residual {err:.2e} > "
                          f"tol {tol:.1e}). Increase max_iter.")
        frac_s = np.count_nonzero(S) / S.size
        self.sparse_fraction_ = frac_s
        if frac_s > 0.5:
            warnings.warn(f"{100 * frac_s:.0f}% of entries were classified as "
                          f"outliers (S). That is not a sparse decomposition: "
                          f"lam_scale is probably too small for this data.")
        self.n_svd_fallback_ = self._n_fallback
        self.debias_rank_ = None
        L *= scale
        S *= scale
        noise = np.zeros_like(L)
        if has_miss:
            noise[miss] = np.nan
        self.solution_ = (L, S)
        self.noise_ = noise
        self.L_, self.S_, self.rank_ = L, S, rank
        return self

    def _svt(self, X, thr, randomized, accurate):
        """Singular value thresholding: returns (U*s_shrunk) @ Vt and rank."""
        n_min = min(X.shape)
        cap = min(int(self.max_rank), n_min) if self.max_rank else n_min

        if self.svd == "auto":
            out = _svt_gram(X, thr, cap)
            if out is not None:
                self._n_gram += 1
                self._nuc = _svt_gram.nuc
                return out
            self._n_fallback += 1               # precision guard tripped

        if not randomized:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
        else:
            k = min(max(self._k_pred, 1), cap)
            while True:
                if k >= 0.5 * n_min:          # partial SVD no longer pays off
                    U, s, Vt = np.linalg.svd(X, full_matrices=False)
                    break
                if accurate:                  # Lanczos, machine precision
                    U, s, Vt = svds(X, k=k, tol=0, random_state=self._rng)
                    order = np.argsort(s)[::-1]
                    U, s, Vt = U[:, order], s[order], Vt[order]
                else:
                    U, s, Vt = _randomized_svd(X, k, self.rsvd_n_iter,
                                               self.rsvd_oversample, self._rng)
                if s[-1] <= thr or k >= cap:  # captured everything above thr
                    break
                k = min(2 * k, cap)

        r = min(int(np.sum(s > thr)), cap)
        self._k_pred = r + self.rsvd_margin
        self._nuc = float(np.sum(s[:r] - thr))
        return (U[:, :r] * (s[:r] - thr)) @ Vt[:r], r

    # ------------------------------------------------------------------
    def fit(self, M, init=None):
        """Decompose M (2-D array). NaN entries are treated as missing.

        Missing entries are not fitted: L fills them in (its values there are
        the model's estimate), S is zero there and noise_ is NaN. Use this to
        hide cells, e.g. a treated unit's period after an intervention, so
        that L gives their counterfactual. Missing values need the
        block-coordinate solver (solver="auto" or "bcd").

        init : (L, S), optional
            Warm start, e.g. a previous solution for nearly the same data.
            Continuation is skipped (the solver starts at the final mu), so
            this only helps when the start is close to the answer.
        """
        self._check_params()
        M = np.array(M, dtype=float)              # copy; never modify input
        if M.ndim != 2:
            raise ValueError("M must be 2-D")
        if M.size == 0:
            raise ValueError("M is empty")
        if np.isinf(M).any():
            raise ValueError("M contains inf")
        miss = np.isnan(M)
        has_miss = bool(miss.any())
        if has_miss:
            if miss.all():
                raise ValueError("M has no observed entries")
            if self.solver == "apg":
                raise ValueError("missing values need solver='auto' or 'bcd'")
        if isinstance(self.sigma, str):               # "auto": estimate it first
            sigma, start = self._auto_sigma(M)
            if init is None:
                init = start
        else:
            sigma, self.sigma_path_ = self.sigma, None
        self.sigma_ = None if sigma is None else float(sigma)
        if has_miss:
            M[miss] = 0.0
        self.missing_ = miss if has_miss else None

        m, n = M.shape
        n_max = max(m, n)
        lam = self.lam if self.lam is not None else self.lam_scale / np.sqrt(n_max)
        if self.mu is None and sigma == 0:
            return self._fit_pcp(M, miss if has_miss else None, lam, init)
        mu_final = self.mu if self.mu is not None else \
            np.sqrt(2.0 * n_max) * sigma

        # Work on M / scale (largest |entry| = 1). The problem is scale-
        # equivariant (M -> cM, mu -> c mu gives L -> cL, S -> cS), so this
        # changes nothing mathematically but keeps squared quantities (Gram
        # matrix, norms) far from float overflow/underflow for any input.
        scale = float(np.max(np.abs(M)))
        if scale == 0.0:
            scale = 1.0
        M /= scale
        mu_final_s = mu_final / scale
        if self.continuation and init is None:
            mu_init_s = max(0.99 * _spectral_norm(M), mu_final_s)
        else:
            mu_init_s = mu_final_s
        self.lam_, self.mu_, self.mu_init_ = lam, mu_final, mu_init_s * scale

        randomized = self.svd == "randomized"
        self._rng = np.random.default_rng(self.random_state)
        self._k_pred = max(1, int(self.rsvd_init_ratio * min(m, n)))
        polishing = False
        self._n_gram = self._n_fallback = 0

        normM = max(np.linalg.norm(M), np.finfo(float).tiny)
        if isinstance(self.tol, str):
            # "auto": stop when the remaining error is about 1/5000 of the
            # total noise, sigma * sqrt(m n) (relative to ||M||)
            noise_norm = (mu_final_s / np.sqrt(2.0 * n_max)) * np.sqrt(m * n)
            # never stricter than 1e-7: for nearly noise-free data extra
            # precision costs time and changes nothing
            tol = max(self.tol_scale * noise_norm / 5000.0 / normM, 1e-7)
        else:
            tol = self.tol
        self.tol_ = tol
        # Arrays kept between steps: M, L, S and one work buffer (plus the
        # two previous iterates in APG mode). Each step overwrites them in
        # place; only the new L from the SVT is freshly allocated.
        if init is not None:
            L = np.array(init[0], dtype=float) / scale
            S = np.array(init[1], dtype=float) / scale
            if L.shape != M.shape or S.shape != M.shape:
                raise ValueError("init must be (L, S) with the shape of M")
        else:
            L = np.zeros_like(M)
            S = np.zeros_like(M)
        buf = np.empty_like(M)
        L_prev = S_prev = None
        t_prev = t = 1.0
        mu = mu_init_s
        mode = "apg" if self.solver == "apg" else "bcd"
        if mode == "apg":
            L_prev, S_prev = np.zeros_like(M), np.zeros_like(M)
        # Anderson acceleration of the block-coordinate map (accelerate=True):
        # same fixed point, fewer steps. State W = what the low-rank step sees
        # (M - S where observed, L where missing).
        aa_on = bool(self.accelerate) and mode == "bcd" and not randomized
        aa_mem = 5
        aa_G, aa_F, aa_FF = [], [], np.zeros((0, 0))
        aa_W = aa_Wp = None
        aa_free = []
        aa_phi, aa_cand, aa_t1, aa_t2 = np.inf, False, None, None
        self.n_reject_ = 0
        self.accel_at_ = None
        aa_confirm, aa_plain, aa_wait, aa_pr = 0, False, False, []   # a stop is confirmed with plain steps
        aa_floor_rate = 0.0
        aa_ratio = None
        aa_nf = np.inf
        aa_rates = []
        self.n_restart_ = 0
        self.switched_at_ = None
        self.errors_ = []
        self.converged_ = False
        rank = 0
        res_hist = []
        n_floor = 0

        for it in range(1, int(self.max_iter) + 1):
            if mode == "aa":
                if aa_W is None:
                    aa_W = M - S
                    if has_miss:
                        aa_mf = miss.astype(float)             # 1 where missing; multiplying is faster than masked copies
                        aa_of = 1.0 - aa_mf
                        aa_W += aa_mf * L                      # M and S are 0 where missing
                L_new, rank_new = self._svt(aa_W, mu, randomized, polishing)
                np.subtract(M, L_new, out=buf)
                # Objective at L_new with its best S: mu ||L||_* + sum over the
                # observed cells of h(M - L), h(x) = x^2/2 up to the cutoff c and
                # c|x| - c^2/2 beyond. Plain steps never increase it, so an
                # extrapolated point that does is discarded for the plain step.
                if aa_t1 is None:
                    aa_t1, aa_t2 = np.empty_like(M), np.empty_like(M)
                np.abs(buf, out=aa_t1)
                if has_miss:
                    aa_t1 *= aa_of
                np.minimum(aa_t1, lam * mu, out=aa_t2)
                phi = mu * self._nuc + float(np.vdot(aa_t1, aa_t2)) - 0.5 * float(np.vdot(aa_t2, aa_t2))
                if aa_cand and phi > aa_phi + 1e-12 * abs(aa_phi):
                    np.copyto(aa_W, aa_G[-1])              # the plain step from the last accepted point
                    aa_free.extend(aa_G); aa_free.extend(aa_F)
                    aa_G, aa_F, aa_FF = [], [], np.zeros((0, 0))
                    aa_free.append(aa_Wp)
                    aa_Wp = None
                    aa_cand, aa_plain = False, True
                    self.n_reject_ += 1
                    self.errors_.append(self.errors_[-1])
                    continue
                aa_phi = phi
                rank = rank_new
                d2 = _sqdist(L_new, L)
                L = L_new
                _soft(buf, lam * mu, out=buf)             # new S
                if has_miss:
                    buf *= aa_of
                d2 += _sqdist(buf, S)
                S, buf = buf, S
                Gk = aa_free.pop() if aa_free else np.empty_like(M)
                np.subtract(M, S, out=Gk)                  # the plain step from aa_W
                if has_miss:
                    np.multiply(L, aa_mf, out=aa_t2)
                    Gk += aa_t2
                if mu > mu_final_s:                        # continuation: plain steps
                    aa_free.append(aa_W)
                    aa_W = Gk
                    aa_free.extend(aa_G); aa_free.extend(aa_F)
                    aa_G, aa_F, aa_FF = [], [], np.zeros((0, 0))
                    if aa_Wp is not None:
                        aa_free.append(aa_Wp)
                    aa_Wp = None
                    aa_nf = np.inf
                    aa_rates = []
                    aa_res = np.inf
                    aa_phi, aa_cand = np.inf, False
                else:
                    Fk = aa_free.pop() if aa_free else np.empty_like(M)
                    np.subtract(Gk, aa_W, out=Fk)
                    nf = float(np.sqrt(np.vdot(Fk, Fk)))
                    aa_res = nf / normM
                    # this point was a plain step from the last one: their step sizes give the plain rate
                    aa_ratio = nf / aa_nf if (aa_plain and np.isfinite(aa_nf) and aa_nf > 0) else None
                    if aa_ratio is None:
                        aa_pr = []
                    else:
                        aa_pr.append(aa_ratio)
                    if aa_Wp is not None and aa_G:         # how much the plain map contracted between the last two points
                        dw = _sqdist(aa_W, aa_Wp)
                        if dw > 0:
                            aa_rates.append(float(np.sqrt(_sqdist(Gk, aa_G[-1]) / dw)))
                    if aa_G and nf > aa_nf:                # no progress: forget the history
                        aa_free.extend(aa_G); aa_free.extend(aa_F)
                        aa_G, aa_F, aa_FF = [], [], np.zeros((0, 0))
                        self.n_restart_ += 1
                    aa_nf = nf
                    if aa_Wp is not None:
                        aa_free.append(aa_Wp)
                    aa_Wp = aa_W
                    aa_G.append(Gk)
                    aa_F.append(Fk)
                    if len(aa_G) > aa_mem + 1:
                        aa_free.append(aa_G.pop(0))
                        aa_free.append(aa_F.pop(0))
                        aa_FF = aa_FF[1:, 1:]
                    q = len(aa_F)
                    new = np.array([float(np.vdot(Fk, f)) for f in aa_F])
                    FF = np.empty((q, q))
                    FF[:q - 1, :q - 1] = aa_FF
                    FF[q - 1, :] = new
                    FF[:, q - 1] = new
                    aa_FF = FF
                    aa_W = aa_free.pop() if aa_free else np.empty_like(M)
                    g = None
                    if q >= 2:
                        # minimise || F_k - sum_j g_j (F_k - F_j) ||
                        kk = FF[q - 1, q - 1]
                        A = kk - FF[q - 1, :q - 1][None, :] - FF[:q - 1, q - 1][:, None] + FF[:q - 1, :q - 1]
                        b = kk - FF[q - 1, :q - 1]
                        A = A + 1e-10 * np.trace(A) * np.eye(q - 1)
                        try:
                            g = np.linalg.solve(A, b)
                        except np.linalg.LinAlgError:
                            g = None
                        if g is not None and not np.all(np.isfinite(g)):
                            g = None
                    aa_plain = g is None or aa_confirm > 0
                    aa_cand = not aa_plain
                    aa_confirm = max(aa_confirm - 1, 0)
                    if aa_plain:
                        np.copyto(aa_W, Gk)
                    else:
                        np.multiply(Gk, 1.0 - g.sum(), out=aa_W)
                        # flat views in the arrays' own memory order (all share M's layout):
                        # a plain reshape(-1) copies a column-ordered array, losing the update
                        wv = aa_W.reshape(-1, order="A")
                        for gj, Gj in zip(g, aa_G[:-1]):
                            _axpy(Gj.reshape(-1, order="A"), wv, a=float(gj))
            elif mode == "bcd":
                # exact minimisation over each block in turn
                np.subtract(M, S, out=buf)
                if has_miss:
                    np.copyto(buf, L, where=miss)          # fill from L (soft-impute)
                L_new, rank = self._svt(buf, mu, randomized, polishing)
                d2 = _sqdist(L_new, L)
                L = L_new
                np.subtract(M, L, out=buf)
                _soft(buf, lam * mu, out=buf)             # new S
                if has_miss:
                    np.copyto(buf, 0.0, where=miss)        # no outliers where unobserved
                d2 += _sqdist(buf, S)
                S, buf = buf, S
            else:
                # accelerated proximal gradient (Lin et al. 2009), in place:
                # L_prev/S_prev become the extrapolated points Y, then the
                # gradient points G = Y - R with R = (Y_L + Y_S - M) / 2.
                beta = (t_prev - 1.0) / t
                for sl in _row_blocks(M):
                    L_prev[sl] *= -beta
                    L_prev[sl] += (1.0 + beta) * L[sl]
                    S_prev[sl] *= -beta
                    S_prev[sl] += (1.0 + beta) * S[sl]
                    np.add(L_prev[sl], S_prev[sl], out=buf[sl])
                    buf[sl] -= M[sl]
                    buf[sl] *= 0.5
                L_prev -= buf                              # G_L
                S_prev -= buf                              # G_S
                L_new, rank = self._svt(L_prev, mu / 2.0, randomized, polishing)
                _soft(S_prev, lam * mu / 2.0, out=buf)     # new S
                # optimality residual (Lin et al. 2009). Its two blocks are
                # exact negatives, (Y_L - Y_S) - (L_new - S_new), and
                # Y_L - Y_S = G_L - G_S, so the norm is sqrt(2) * one block.
                r2 = 0.0
                for sl in _row_blocks(M):
                    d = L_prev[sl] - S_prev[sl]
                    d -= L_new[sl]
                    d += buf[sl]
                    r2 += float(np.vdot(d, d))
                res = np.sqrt(2.0 * r2) / normM
                res_hist.append(res)
                d2 = _sqdist(L_new, L) + _sqdist(buf, S)
                # rotate: old iterates become the previous ones; the G_S
                # array is reused as the next work buffer
                L_prev, L = L, L_new
                S_prev, S, buf = S, buf, S_prev
                t_prev, t = t, (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0

            err = np.sqrt(d2) / normM
            self.errors_.append(err)

            at_floor = mu <= mu_final_s
            if at_floor:
                n_floor += 1
            if self.continuation:
                eta = self.eta if mode == "bcd" else self.eta_apg
                mu = max(eta * mu, mu_final_s)
            if self.verbose and (it % 10 == 0 or it == 1):
                print(f"iter {it:5d}  {mode}  mu {mu * scale:.3e}  "
                      f"rank {rank:4d}  nnz(S) {np.count_nonzero(S):8d}  "
                      f"change {err:.2e}")

            # Distance to the solution is about change / (1 - rate) when the
            # iteration contracts by `rate` per step; using the raw change
            # alone stops too early when progress is slow.
            dist = np.inf
            if mode == "aa":
                # For a map that contracts by `rate`, the distance from the
                # point where it was applied to the fixed point is at most
                # (plain step size) / (1 - rate), however that point was made.
                # The rate seen between accelerated points can be too low, so
                # a stop is only accepted after three plain steps confirm it
                # with the plain solver's own estimate:
                # step size / (1 - largest ratio of successive plain steps).
                if at_floor and aa_res == 0:
                    dist = 0.0
                elif at_floor and len(aa_pr) >= 3:
                    rate = max(aa_pr[-3:])
                    aa_floor_rate = max(aa_floor_rate, min(rate, 1.0))
                    aa_wait = False
                    if rate < 1.0:
                        # after extrapolated steps this estimate runs low (measured: up
                        # to about 2x), so it is doubled before it is compared with tol
                        dist = _AA_SAFETY * aa_res / (1.0 - rate)
                elif at_floor and not aa_wait and len(aa_rates) >= 3:
                    rate = max(max(aa_rates[-3:]), aa_floor_rate)
                    if rate < 1.0 and _AA_SAFETY * aa_res / (1.0 - rate) <= tol:
                        aa_confirm, aa_wait = 3, True
            elif mode == "bcd":
                rate = np.inf
                if len(self.errors_) >= 4 and self.errors_[-4] > 0:
                    ratios = [self.errors_[i] / self.errors_[i - 1]
                              for i in range(-3, 0) if self.errors_[i - 1] > 0]
                    rate = max(ratios) if ratios else 0.0
                if at_floor:
                    if err == 0:
                        dist = 0.0
                    elif rate < 1.0:
                        dist = err / (1.0 - rate)
                # Block-coordinate steps can crawl (e.g. dense noise tiny
                # compared with the outliers). Detect it from the contraction
                # rate after a few steps at the final mu, then switch to the
                # accelerated method with its own (slower) continuation.
                if (self.solver == "auto" and not has_miss and n_floor >= 5
                        and len(self.errors_) >= 6
                        and dist > tol
                        and np.median(np.array(self.errors_[-5:]) /
                                      np.maximum(self.errors_[-6:-1], 1e-300))
                        > self.switch_rate):
                    mode = "apg"
                    self.switched_at_ = it
                    L_prev, S_prev = L.copy(), S.copy()
                    t_prev = t = 1.0
                    res_hist = []
                    n_floor = 0
                    if self.continuation:
                        mu = mu_init_s
                # Not converged after the first few plain steps at the final
                # mu: continue with accelerated steps (same fixed point).
                # Only when plain steps still have a long way to go: on a fit that
                # is about to stop, the extrapolation and its confirming plain
                # steps cost more than they save.
                if mode == "bcd" and aa_on and n_floor >= 5 and len(self.errors_) >= 6 and dist > tol:
                    left = np.inf
                    if 0.0 < rate < 1.0 and err > 0:
                        left = np.log(tol * (1.0 - rate) / err) / np.log(rate)
                    if left > _AA_MIN_STEPS:
                        mode = "aa"
                        self.accel_at_ = it
            else:
                # momentum makes single steps noisy: average the rate over
                # the last 10 residuals
                if at_floor and len(res_hist) >= 11 and res_hist[-11] > 0:
                    rate = (res_hist[-1] / res_hist[-11]) ** 0.1
                    if res_hist[-1] == 0:
                        dist = 0.0
                    elif rate < 1.0:
                        dist = max(res_hist[-1], err) / (1.0 - rate)
            self.dist_est_ = dist

            if randomized and not polishing and at_floor and err <= self.polish_tol:
                polishing = True
            if at_floor and dist <= tol and (polishing or not randomized):
                self.converged_ = True
                break

        self.n_iter_ = it
        if not self.converged_:
            warnings.warn(f"StablePCP did not converge in {self.max_iter} "
                          f"iterations (estimated distance {dist:.2e} > tol "
                          f"{tol:.1e}). Increase max_iter.")

        frac_s = np.count_nonzero(S) / S.size
        self.sparse_fraction_ = frac_s
        if frac_s > 0.5:
            warnings.warn(f"{100 * frac_s:.0f}% of entries were classified as "
                          f"outliers (S). That is not a sparse decomposition: "
                          f"lam_scale is probably too small (or sigma too "
                          f"small) for this data.")
        self.n_svd_fallback_ = self._n_fallback
        del buf, L_prev, S_prev, aa_G, aa_F, aa_W, aa_Wp, aa_free
        self.debias_rank_ = None
        # solution before debiasing: a warm start for refits on similar data
        self.solution_ = (L * scale, S * scale) if self.debias else None
        if self.debias:
            L, S, rank = self._debias(M, L, S, lam * mu_final_s, mu_final_s, rank,
                                      miss if has_miss else None)
            frac_s = np.count_nonzero(S) / S.size
            self.sparse_fraction_ = frac_s
        L *= scale
        S *= scale
        if self.solution_ is None:
            self.solution_ = (L, S)                  # same arrays as L_, S_
        M *= scale                                   # our own copy of the input
        M -= L
        M -= S                                       # M now holds the noise
        if has_miss:
            M[miss] = np.nan                         # unobserved: no noise estimate
        self.noise_ = M
        self.L_, self.S_, self.rank_ = L, S, rank
        return self

    def _auto_sigma(self, M):
        """Estimate the noise level for sigma="auto".

        Looks for the value s at which a fit with sigma = s leaves a residual
        whose typical noise level is s again. Returns (sigma, warm start for
        the final fit or None). M may contain NaN (missing).
        """
        params = {k: getattr(self, k) for k in self._param_names()}
        params.update(debias=True, verbose=False)     # the unshrunk residual
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # all-missing columns
            dev = np.abs(M - np.nanmedian(M, axis=0))
            s = 1.4826 * float(np.nanmedian(dev))     # spread of the data: a start from above
            if not (np.isfinite(s) and s > 0):
                s = float(np.nanstd(M))
        if not (np.isfinite(s) and s > 0):
            raise ValueError("sigma='auto': the data has no variation to "
                             "estimate the noise from")
        floor = 1e-6 * float(np.nanmax(np.abs(M)))    # below this the data is noise-free in practice
        NOISE_FREE = 0.2       # measured < this x tried, after a round started from a measurement
        rtol, path, lo, hi, prev = 0.02, [], None, None, None
        confirming = False
        self.sigma_path_ = path
        # Speed: each round starts from the previous round's solution (the
        # problem changes only through sigma), and with tol="auto" the trial
        # fits stop at about 2% of the noise rather than 0.02%, matching the
        # 2% acceptance rule below. A round that looks accepted is refitted
        # at full precision (from its rough solution) before it is accepted.
        rough = 100.0 if isinstance(self.tol, str) else 1.0
        start = None

        def trial_fit(s, mult, init):
            # Keep only what the search needs (the warm start and the measured
            # noise): the trial's other full-size arrays are freed at once, so
            # they do not sit in memory while the next trial runs.
            trial = StablePCP(**dict(params, sigma=s, tol_scale=mult * self.tol_scale))
            trial.fit(M, init=init)
            m = _typical_noise(M - trial.L_)
            sol, rank = trial.solution_, trial.rank_
            del trial
            return sol, rank, m

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # trial fits may warn; the final fit reports
            for _ in range(12):
                start, rank, m = trial_fit(s, rough, start)
                if rough != 1.0 and m > floor and abs(m - s) <= rtol * s:
                    start, rank, m = trial_fit(s, 1.0, start)
                path.append((float(s), float(m)))
                if self.verbose:
                    print(f"sigma='auto': tried {s:.4g}, measured {m:.4g} "
                          f"(rank {rank})")
                if m <= floor:
                    return 0.0, None                  # nothing left at all: no dense noise
                if len(path) >= 2 and m < NOISE_FREE * s:
                    # The leftover vanished when treated as noise: it was
                    # structure, not noise. Confirm once, one step lower (s/5,
                    # not the tiny measured value, whose fit would be slow):
                    # real noise reappears there (measured > tried), while in
                    # noise-free data the leftover collapses again. Then there
                    # is no positive value where tried and measured agree, and
                    # the answer is sigma = 0 (exact M = L + S).
                    if confirming:
                        return 0.0, None
                    confirming = True
                    hi = s if hi is None else min(hi, s)
                    prev, s = (s, m), max(s / 5.0, floor)
                    continue
                confirming = False
                if abs(m - s) <= rtol * s:
                    return s, start                   # same problem as the final fit
                if m > s:
                    lo = s if lo is None else max(lo, s)
                else:
                    hi = s if hi is None else min(hi, s)
                nxt = m                               # plain step: use what was measured
                if prev is not None:                  # faster: secant step on measured - tried
                    g0, g1 = prev[1] - prev[0], m - s
                    if g1 != g0:
                        sec = s - g1 * (s - prev[0]) / (g1 - g0)
                        # accept it only near the plain step (it extrapolates)
                        if np.isfinite(sec) and m / 3.0 < sec < 3.0 * m:
                            nxt = sec
                if lo is not None and hi is not None and not lo < nxt < hi:
                    nxt = float(np.sqrt(lo * hi))     # keep inside the bracket
                prev, s = (s, m), max(float(nxt), floor)
        best = min(path, key=lambda p: abs(p[1] - p[0]) / p[0])
        warnings.warn("sigma='auto' did not settle within 2% in 12 rounds; "
                      f"using {best[0]:.4g} (measured {best[1]:.4g}). "
                      "See sigma_path_.")
        return best[0], None

    def _debias(self, M, L, S, cut, mu, fit_rank, miss=None):
        """Undo the shrinkage of L while keeping Stable PCP's decisions.

        Keep the number of components and the outlier cutoff; refit L as a
        plain rank-r approximation, treating clear outliers as missing (filled
        from L itself) so they cannot pull L toward them; then recompute S
        with the same cutoff. Repeat until the outlier set stops changing.
        """
        n = min(M.shape)
        A = L if L.shape[0] >= L.shape[1] else L.T
        sv = np.sqrt(np.clip(np.linalg.eigvalsh(A.T @ A), 0, None))[::-1][:n]
        factor = self.debias_rank_factor
        if factor is None:
            # Pure noise has singular values up to about sigma*(sqrt(m)+sqrt(n))
            # = mu * edge_ratio; after shrinkage by mu at most mu*(edge_ratio-1).
            # Keep components above that, with a 20% margin. For tall or wide
            # matrices noise never survives (edge_ratio < 1) and nothing is cut.
            m_, n_ = M.shape
            edge_ratio = (np.sqrt(m_) + np.sqrt(n_)) / np.sqrt(2.0 * max(m_, n_))
            factor = 1.2 * max(0.0, edge_ratio - 1.0)
        self.debias_rank_factor_ = factor
        # never more components than the Stable PCP solution itself kept
        r = min(int(np.sum(sv > factor * mu)), int(fit_rank))
        if self.max_rank:
            r = min(r, int(self.max_rank))
        # Clear outliers (|S| > debias_mask_factor * noise level) are treated
        # as missing and filled from L; other entries enter with their outlier
        # part removed (M - S). The noise level is mu / sqrt(2 max(m, n)),
        # i.e. sigma in the scaled problem, so the rule does not depend on
        # lam_scale.
        noise = mu / np.sqrt(2.0 * max(M.shape))
        k = float(self.debias_mask_factor) * noise / cut
        Z = np.empty_like(M)
        support = S != 0
        for _ in range(int(self.debias_iter)):
            np.subtract(M, S, out=Z)
            mask = np.abs(S) > k * cut
            if miss is not None:
                mask |= miss
            np.copyto(Z, L, where=mask)
            L = _rank_r(Z, r)
            np.subtract(M, L, out=S)
            _soft(S, cut, out=S)
            if miss is not None:
                np.copyto(S, 0.0, where=miss)
            new_support = S != 0
            if np.array_equal(new_support, support):
                break
            support = new_support
        self.debias_rank_ = r
        return L, S, r

    def conformal(self, windows, level=0.9, n_cal=200, per_refit=None,
                  effect="auto", candidates="same_cols", statistic=None, scale=None,
                  n_groups=1, random_state=None, verbose=False):
        """Prediction intervals for "what would the data have been in these windows".

        Split-conformal calibration with held-out windows: fake events are
        planted in calibration windows (copies of your windows moved
        elsewhere in the matrix), the model is refitted, and its errors on
        those windows - against the original data - give the interval width.
        This does not assume the low-rank model is right.
        Costs about n_cal / per_refit extra fits.

        Parameters
        ----------
        windows : list of index tuples, e.g. (slice(r0, r0 + 4), j)
            The cells of each event of interest (indexing M / L_).
        level : float
            Target coverage of the intervals.
        n_cal : int
            Number of calibration windows.
        per_refit : int or None
            Calibration windows per refit (they never overlap each other or
            your windows). None: 40 when planting; when hiding, as many as
            keep about 5% of the observed cells hidden (at least 1).
        effect : "auto", "residual" or (low, high)
            What to plant. "auto": the estimated effect of a randomly chosen
            event (its S_ values), so fake events look like the real ones.
            "residual": the event's full departure from L (data - L_), which
            does not understate events that partly leak into L.
            (low, high): a shift per cell, uniform in [low, high] * sigma with
            a random sign per window.
        candidates : "same_cols", "same_rows", "auto", "random" or a list
            Where calibration windows go. Each is a copy of one of your
            windows moved to other rows in the same columns ("same_cols",
            the default), to other columns in the same rows ("same_rows"),
            anywhere ("random"), or half and half ("auto"). A list of windows
            (same shape as the events) uses those instead. With rows = time
            and columns = units, "same_cols" calibrates each unit on its own
            history, which matters when units differ in how predictable they
            are; transpose (or use "same_rows") if your time runs along
            columns.
        statistic : callable(V, window) -> float, optional
            Quantity of interest; default V[window].sum(). Applied to the
            original data (truth) and to the fitted L (counterfactual).
        scale : callable(estimate) -> float, optional
            Divides calibration errors so windows of different size are
            comparable, e.g. lambda c: max(c, 1) when errors grow in proportion
            to the size of the window (counts, totals).
        n_groups : int
            Calibrate separately within this many groups of windows of similar
            estimated size (quantiles of the calibration estimates). With 1,
            all windows share one interval rule, so if the scale is not quite
            right, small and large windows get too wide or too narrow
            intervals.

        Hidden windows: if your windows are missing (NaN) in the fitted
        data, e.g. a treated unit's cells after an intervention, calibration
        hides same-shaped observed windows instead of planting fake events,
        refits, and compares the filled-in values with the real ones (a
        placebo test turned into intervals). `effect` is then not used, and
        candidates="same_rows" (other units over the same period) is
        usually the natural choice.

        Returns
        -------
        dict with estimate, low, high (one per window), cal_errors (scaled
        errors on the calibration windows) and cal_estimates (the model's
        estimate on each calibration window, for diagnostics).
        Coverage is approximate: it assumes the event windows behave like
        the calibration windows.
        """
        self._check_fitted()
        if self.sigma is None:
            raise ValueError("conformal needs sigma (the noise level)")
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        modes = ("auto", "same_cols", "same_rows", "random")
        if isinstance(candidates, str) and candidates not in modes:
            raise ValueError(f"candidates must be a list or one of {modes}")
        if not (effect in ("auto", "residual") or (not isinstance(effect, str) and len(effect) == 2)):
            raise ValueError("effect must be 'auto', 'residual' or (low, high)")
        if not isinstance(effect, str) and not self.sigma_:
            raise ValueError("effect=(low, high) is in units of sigma, which is 0 "
                             "here; use effect='auto' or 'residual'")
        rng = np.random.default_rng(random_state)
        M0 = self.L_ + self.S_ + self.noise_            # the data that was fitted
        m, n = M0.shape
        stat = statistic or (lambda V, w: float(np.sum(V[w])))
        scl = scale or (lambda c: 1.0)
        windows = list(windows)
        if not windows:
            raise ValueError("no windows given")

        def cells(w):
            mask = np.zeros((m, n), bool)
            mask[w] = True
            return np.nonzero(mask)                     # (rows, cols) arrays

        ev = [cells(w) for w in windows]
        busy = np.zeros((m, n), bool)
        for rr, cc in ev:
            busy[rr, cc] = True
        miss = getattr(self, "missing_", None)
        hide = False
        if miss is not None:
            hidden = [bool(miss[rr, cc].all()) for rr, cc in ev]
            observed = [not miss[rr, cc].any() for rr, cc in ev]
            if all(hidden):
                hide = True
            elif not all(observed):
                raise ValueError("windows must be either all missing (hidden) "
                                 "or all observed")
            busy |= miss                        # calibrate only where observed
        if per_refit is None:
            if hide:
                size = np.mean([len(rr) for rr, _ in ev])
                per_refit = max(1, int(0.05 * (m * n - miss.sum()) / size))
            else:
                per_refit = 40

        def calib_window():
            """Returns (cells of the calibration window, index of source event)."""
            i = int(rng.integers(len(ev)))
            if not isinstance(candidates, str):
                return cells(candidates[rng.integers(len(candidates))]), i
            rr, cc = ev[i]
            mode = candidates
            if mode == "auto":
                mode = "same_cols" if rng.random() < 0.5 else "same_rows"
            dr = dc = 0
            if mode in ("same_cols", "random"):
                dr = int(rng.integers(-rr.min(), m - rr.max()))
            if mode in ("same_rows", "random"):
                dc = int(rng.integers(-cc.min(), n - cc.max()))
            return (rr + dr, cc + dc), i

        params = {k: getattr(self, k) for k in self._param_names()}
        params["verbose"] = False
        params["sigma"] = self.sigma_                 # refits do not re-estimate it
        errors, cal_est = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while len(errors) < n_cal:
                Mb = M0.copy()
                used = busy.copy()
                planted = []
                tries = 0
                want = min(per_refit, n_cal - len(errors))
                while len(planted) < want and tries < 100 * per_refit:
                    tries += 1
                    w, i = calib_window()
                    if used[w].any():
                        continue
                    used[w] = True
                    if hide:
                        Mb[w] = np.nan                     # hide, as the real window
                    elif effect in ("auto", "residual"):
                        er, ec = ev[i]
                        if len(er) == len(w[0]):
                            src = self.S_ if effect == "auto" else M0 - self.L_
                            Mb[w] += src[er, ec]
                    else:
                        Mb[w] += (rng.choice([-1.0, 1.0]) *
                                  rng.uniform(*effect) * self.sigma_)
                    planted.append(w)
                if not planted:
                    raise ValueError("no room for calibration windows")
                Lb = StablePCP(**params).fit(Mb, init=self.solution_).L_
                for w in planted:
                    est = stat(Lb, w)
                    errors.append((stat(M0, w) - est) / scl(est))
                    cal_est.append(est)
                if verbose:
                    print(f"conformal: {len(errors)}/{n_cal} calibration windows")
        errors = np.array(errors)
        cal_est = np.array(cal_est)
        est = np.array([stat(self.L_, w) for w in windows])
        s = np.array([scl(e) for e in est])
        a = (1.0 - level) / 2.0

        def quantiles(e):
            # finite-sample conformal quantiles
            k = len(e)
            return (np.quantile(e, max(0.0, np.floor(a * (k + 1)) / k)),
                    np.quantile(e, min(1.0, np.ceil((1 - a) * (k + 1)) / k)))

        n_groups = max(1, int(n_groups))
        if n_groups == 1:
            qlo, qhi = quantiles(errors)
        else:
            cuts = np.quantile(cal_est, np.linspace(0, 1, n_groups + 1)[1:-1])
            g_cal, g_ev = np.digitize(cal_est, cuts), np.digitize(est, cuts)
            qlo, qhi = np.empty(len(est)), np.empty(len(est))
            for g in range(n_groups):
                eg = errors[g_cal == g]
                if len(eg) < 20:
                    raise ValueError("too few calibration windows per group: "
                                     "increase n_cal or lower n_groups")
                qlo[g_ev == g], qhi[g_ev == g] = quantiles(eg)
        return {"estimate": est, "low": est + qlo * s, "high": est + qhi * s,
                "cal_errors": errors, "cal_estimates": cal_est}

    def _param_names(self):
        import inspect
        return [p for p in inspect.signature(StablePCP.__init__).parameters
                if p != "self"]

    def _check_fitted(self):
        if not hasattr(self, "L_"):
            raise RuntimeError("StablePCP is not fitted yet; call fit(M) first")

    def trimmed_low_rank(self, factor=0.5):
        """L_ without the noise-level components.

        Stable PCP shrinks every singular value by mu_. Pure i.i.d. noise has
        singular values up to about sqrt(2) * mu_, so after shrinkage noise
        components are at most ~0.41 * mu_. Components above `factor * mu_`
        are kept. Returns (L_trimmed, rank).
        """
        self._check_fitted()
        return trim_rank(self.L_, threshold=factor * self.mu_)

    def fit_transform(self, M):
        self.fit(M)
        return self.L_, self.S_

    # compatibility with the RobustPCA / StablePCP getters
    def get_low_rank(self):
        self._check_fitted()
        return self.L_

    def get_sparse(self):
        self._check_fitted()
        return self.S_

    def get_rank(self):
        self._check_fitted()
        return self.rank_


# ----------------------------------------------------------------------------
# noise level from residuals (used by sigma="auto")
# ----------------------------------------------------------------------------
_LOG_ABS_SD = float(np.pi / (2.0 * np.sqrt(2.0)))    # sd of log|z|, z standard normal (about 1.11)
_PURE_LEVEL = {}                                     # shape -> what the steps below give on pure noise of scale 1


def _log_level(A):
    """Fitted log noise scale of each entry. A = log|residual| (NaN = missing).

    Centred at its median first, so the units of the data do not matter.
    """
    a0 = float(np.nanmedian(A))
    return StablePCP(sigma=_LOG_ABS_SD, debias=True).fit(A - a0).L_ + a0


def _noise_scale(res):
    """Noise standard deviation of every entry, from residuals M - L.

    For noise of any scale s, log|noise| = log s + (a term whose spread is the
    same everywhere, sd about 1.11). So the log sizes of the residuals are
    again "regular part + a few exceptions (events) + even noise" with a known
    sigma, and a second Stable PCP fit gives the regular part: the log scale.
    Its level is set by running the same steps on pure noise of scale 1.
    """
    absr = np.abs(res)
    med = float(np.nanmedian(absr))
    if not med > 0:
        return np.zeros(res.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = np.log(np.maximum(absr, 1e-3 * med))     # NaN stays NaN
        if res.shape not in _PURE_LEVEL:
            z = np.random.default_rng(123).normal(size=res.shape)
            _PURE_LEVEL[res.shape] = float(np.median(_log_level(np.log(np.abs(z)))))
        return np.exp(_log_level(A) - _PURE_LEVEL[res.shape])


def _typical_noise(res):
    """Median noise standard deviation over the observed entries of M - L."""
    scale = _noise_scale(res)
    obs = ~np.isnan(res)
    return float(np.median(scale[obs])) if obs.any() else 0.0
