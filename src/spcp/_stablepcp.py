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
from scipy.sparse.linalg import svds

__all__ = ["StablePCP", "trim_rank"]


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
        return np.zeros_like(X), 0
    ratio = np.sqrt(w_max) / thr                # s_max / thr, no thr^2 underflow
    if np.finfo(float).eps * ratio * ratio > _GRAM_RTOL:
        return None
    k = int(np.count_nonzero(w > thr * thr))
    k = min(k, cap)
    V = V[:, -k:]                               # top-k eigenvectors
    s = np.sqrt(w[-k:])
    if k <= A.shape[1] // 2:
        L = (A @ V * (1.0 - thr / s)) @ V.T     # m x k intermediate (small k)
    else:
        L = A @ ((V * (1.0 - thr / s)) @ V.T)   # n x n intermediate only
    return (L if tall else L.T), k


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
    sigma : float, required
        Standard deviation of the dense noise N, in the units of M.
        Sets the final shrinkage level. (Alternatively pass `mu` directly.)
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
    tol : float, default 1e-7
        Stop when the estimated distance of (L, S) to the solution, relative
        to ||M||_F, falls below tol, once mu has reached its final value.
        The distance is estimated as change / (1 - rate), with `rate` the
        recent contraction factor, so slow-converging problems are not
        stopped early. When L is much smaller than M, the relative error of
        L itself is larger than tol.
    max_iter : int, default 10000
        Most problems converge well before this limit. Settings where most
        entries end up in S (lam_scale far too small for the data) converge
        slowly and trigger a warning (sparse_fraction_).
    random_state : int or None
        Seed for the randomized SVD.
    verbose : bool, default False
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
    """

    def __init__(self, sigma=None, lam_scale=1.0, lam=None, mu=None,
                 continuation=True, eta=0.5, max_rank=None, svd="auto",
                 solver="auto", eta_apg=0.9, switch_rate=0.995,
                 tol=1e-7, max_iter=10000, random_state=None, verbose=False,
                 # randomized-SVD tuning (rarely needed)
                 rsvd_n_iter=10, rsvd_oversample=10, rsvd_margin=5,
                 rsvd_init_ratio=0.2, polish_tol=1e-3):
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

    # ------------------------------------------------------------------
    def _check_params(self):
        if self.mu is None:
            if self.sigma is None:
                raise ValueError(
                    "sigma (std of the dense noise) is required; results depend "
                    "strongly on it. Alternatively pass mu directly.")
            if not self.sigma > 0:
                raise ValueError("sigma must be > 0")
        elif not self.mu > 0:
            raise ValueError("mu must be > 0")
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
        if not self.tol >= 0:
            raise ValueError("tol must be >= 0")
        if not self.polish_tol > 0:
            raise ValueError("polish_tol must be > 0")
        if int(self.rsvd_n_iter) < 0 or int(self.rsvd_oversample) < 0 \
                or int(self.rsvd_margin) < 0:
            raise ValueError("rsvd_n_iter, rsvd_oversample and rsvd_margin "
                             "must be >= 0")
        if not 0 < self.rsvd_init_ratio <= 1:
            raise ValueError("rsvd_init_ratio must be in (0, 1]")

    # ------------------------------------------------------------------
    def _svt(self, X, thr, randomized, accurate):
        """Singular value thresholding: returns (U*s_shrunk) @ Vt and rank."""
        n_min = min(X.shape)
        cap = min(int(self.max_rank), n_min) if self.max_rank else n_min

        if self.svd == "auto":
            out = _svt_gram(X, thr, cap)
            if out is not None:
                self._n_gram += 1
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
        return (U[:, :r] * (s[:r] - thr)) @ Vt[:r], r

    # ------------------------------------------------------------------
    def fit(self, M):
        """Decompose M (2-D array, all entries finite)."""
        self._check_params()
        M = np.array(M, dtype=float)              # copy; never modify input
        if M.ndim != 2:
            raise ValueError("M must be 2-D")
        if M.size == 0:
            raise ValueError("M is empty")
        if not np.isfinite(M).all():
            raise ValueError("M contains NaN or inf (missing values are not "
                             "supported in this version)")

        m, n = M.shape
        n_max = max(m, n)
        mu_final = self.mu if self.mu is not None else \
            np.sqrt(2.0 * n_max) * self.sigma
        lam = self.lam if self.lam is not None else self.lam_scale / np.sqrt(n_max)

        # Work on M / scale (largest |entry| = 1). The problem is scale-
        # equivariant (M -> cM, mu -> c mu gives L -> cL, S -> cS), so this
        # changes nothing mathematically but keeps squared quantities (Gram
        # matrix, norms) far from float overflow/underflow for any input.
        scale = float(np.max(np.abs(M)))
        if scale == 0.0:
            scale = 1.0
        M /= scale
        mu_final_s = mu_final / scale
        if self.continuation:
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
        # Arrays kept between steps: M, L, S and one work buffer (plus the
        # two previous iterates in APG mode). Each step overwrites them in
        # place; only the new L from the SVT is freshly allocated.
        L = np.zeros_like(M)
        S = np.zeros_like(M)
        buf = np.empty_like(M)
        L_prev = S_prev = None
        t_prev = t = 1.0
        mu = mu_init_s
        mode = "apg" if self.solver == "apg" else "bcd"
        if mode == "apg":
            L_prev, S_prev = np.zeros_like(M), np.zeros_like(M)
        self.switched_at_ = None
        self.errors_ = []
        self.converged_ = False
        rank = 0
        res_hist = []
        n_floor = 0

        for it in range(1, int(self.max_iter) + 1):
            if mode == "bcd":
                # exact minimisation over each block in turn
                np.subtract(M, S, out=buf)
                L_new, rank = self._svt(buf, mu, randomized, polishing)
                d2 = _sqdist(L_new, L)
                L = L_new
                np.subtract(M, L, out=buf)
                _soft(buf, lam * mu, out=buf)             # new S
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
            if mode == "bcd":
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
                if (self.solver == "auto" and n_floor >= 5 and len(self.errors_) >= 6
                        and dist > self.tol
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
            if at_floor and dist <= self.tol and (polishing or not randomized):
                self.converged_ = True
                break

        self.n_iter_ = it
        if not self.converged_:
            warnings.warn(f"StablePCP did not converge in {self.max_iter} "
                          f"iterations (estimated distance {dist:.2e} > tol "
                          f"{self.tol:.1e}). Increase max_iter.")

        frac_s = np.count_nonzero(S) / S.size
        self.sparse_fraction_ = frac_s
        if frac_s > 0.5:
            warnings.warn(f"{100 * frac_s:.0f}% of entries were classified as "
                          f"outliers (S). That is not a sparse decomposition: "
                          f"lam_scale is probably too small (or sigma too "
                          f"small) for this data.")
        self.n_svd_fallback_ = self._n_fallback
        del buf, L_prev, S_prev
        L *= scale
        S *= scale
        M *= scale                                   # our own copy of the input
        M -= L
        M -= S                                       # M now holds the noise
        self.noise_ = M
        self.L_, self.S_, self.rank_ = L, S, rank
        return self

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
