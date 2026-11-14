"""Tests for spcp.StablePCP.  Run from the repo root:  pytest"""
import sys
import warnings

import numpy as np
import pytest

from spcp import StablePCP, trim_rank


def make_data(m=200, n=200, r=5, frac=0.05, sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    L0 = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    S0 = np.zeros((m, n))
    mk = rng.random((m, n)) < frac
    S0[mk] = rng.uniform(-20, 20, mk.sum())
    return L0, S0, L0 + S0 + sigma * rng.standard_normal((m, n))


def rel(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(B)


def objective(M, L, S, lam, mu):
    return (np.linalg.svd(L, compute_uv=False).sum() + lam * np.abs(S).sum()
            + ((M - L - S) ** 2).sum() / (2 * mu))


# --- reference: independent block-coordinate solver (from the paper) ------
def bcd_reference(M, lam, mu, iters=5000, tol=1e-12):
    L = np.zeros_like(M); S = np.zeros_like(M)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(M - S, full_matrices=False)
        Ln = (U * np.maximum(s - mu, 0)) @ Vt
        Sn = np.sign(M - Ln) * np.maximum(np.abs(M - Ln) - lam * mu, 0)
        d = np.linalg.norm(Ln - L) + np.linalg.norm(Sn - S)
        L, S = Ln, Sn
        if d < tol * np.linalg.norm(M):
            break
    return L, S


def test_matches_independent_solver():
    L0, S0, M = make_data()
    m = StablePCP(sigma=0.1, svd="exact").fit(M)
    assert m.converged_
    Lr, Sr = bcd_reference(M, m.lam_, m.mu_)
    assert rel(m.L_, Lr) < 1e-5 and rel(m.S_, Sr) < 1e-5
    o1, o2 = objective(M, m.L_, m.S_, m.lam_, m.mu_), objective(M, Lr, Sr, m.lam_, m.mu_)
    assert abs(o1 - o2) / o2 < 1e-8
    assert rel(m.L_, L0) < 0.02


def test_beats_pca():
    L0, S0, M = make_data()
    m = StablePCP(sigma=0.1).fit(M)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    assert rel(m.L_, L0) < rel((U[:, :5] * s[:5]) @ Vt[:5], L0) / 5


def test_randomized_matches_exact():
    L0, S0, M = make_data(600, 600, 8)
    e = StablePCP(sigma=0.1, svd="exact").fit(M)
    r = StablePCP(sigma=0.1, svd="randomized", random_state=0).fit(M)
    assert e.converged_ and r.converged_
    assert rel(r.L_, e.L_) < 1e-5


def test_parameter_formulas_and_no_mutation():
    L0, S0, M = make_data(100, 300)
    m = StablePCP(sigma=0.2, lam_scale=2.0)
    m.fit(M)
    assert np.isclose(m.lam_, 2 / np.sqrt(300))
    assert np.isclose(m.mu_, np.sqrt(600) * 0.2)
    assert m.lam is None and m.mu is None          # constructor args untouched
    m.fit(M[:, :50])                               # refit on new shape
    assert np.isclose(m.lam_, 2 / np.sqrt(100))


def test_explicit_mu_and_no_continuation():
    L0, S0, M = make_data()
    a = StablePCP(sigma=0.1).fit(M)
    b = StablePCP(mu=a.mu_, continuation=False, max_iter=5000).fit(M)
    assert b.converged_ and rel(b.L_, a.L_) < 1e-5


def test_scale_invariance():
    L0, S0, M = make_data()
    a = StablePCP(sigma=0.1).fit(M)
    b = StablePCP(sigma=10.0).fit(100 * M)
    assert b.converged_ and rel(b.L_ / 100, a.L_) < 1e-5


def test_errors():
    M = np.ones((5, 5))
    with pytest.raises(ValueError, match="sigma"):
        StablePCP().fit(M)
    with pytest.raises(ValueError):
        StablePCP(sigma=-1).fit(M)
    with pytest.raises(ValueError):
        StablePCP(sigma=1).fit(np.ones(5))
    with pytest.raises(ValueError):
        StablePCP(sigma=1, svd="fast").fit(M)


def test_input_not_modified():
    L0, S0, M = make_data(50, 50, 2)
    M_copy = M.copy()
    StablePCP(sigma=0.1).fit(M)
    assert np.array_equal(M, M_copy)


def test_rejects_nan():
    M = np.ones((5, 5)); M[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        StablePCP(sigma=1).fit(M)


def test_trim_rank():
    L0, S0, M = make_data()
    m = StablePCP(sigma=0.1).fit(M)
    assert m.rank_ > 5
    Lt, r = trim_rank(m.L_)
    assert r == 5 and rel(Lt, L0) <= rel(m.L_, L0) * 1.05
    Lt2, r2 = m.trimmed_low_rank()
    assert r2 == 5 and rel(Lt2, L0) <= rel(m.L_, L0) * 1.05


def test_trimmed_low_rank_with_offset():
    L0, S0, M = make_data()
    m = StablePCP(sigma=0.1).fit(M + 5.0)          # large common offset
    Lt, r = m.trimmed_low_rank()
    assert r == 6                                  # 5 + the offset component


def test_max_rank_cap():
    L0, S0, M = make_data()
    m = StablePCP(sigma=0.1, max_rank=5, max_iter=3000).fit(M)
    assert m.rank_ <= 5


@pytest.mark.parametrize("shape", [(200, 200), (400, 80), (80, 400)])
def test_auto_svd_matches_exact(shape):
    L0, S0, M = make_data(*shape, r=4)
    a = StablePCP(sigma=0.1, svd="auto").fit(M)
    e = StablePCP(sigma=0.1, svd="exact").fit(M)
    assert a.converged_ and a.n_svd_fallback_ == 0
    assert rel(a.L_, e.L_) < 1e-8 and rel(a.S_, e.S_) < 1e-8


def test_auto_svd_falls_back_on_huge_offset():
    L0, S0, M = make_data(r=4)
    M = M + 1e5                                    # s_max / thr ~ 1e6
    a = StablePCP(sigma=0.1, svd="auto").fit(M)
    e = StablePCP(sigma=0.1, svd="exact").fit(M)
    assert a.n_svd_fallback_ > 0
    assert rel(a.L_, e.L_) < 1e-10


def test_more_validation():
    M = np.ones((5, 5))
    for kw in [dict(max_iter=0), dict(tol=-1), dict(eta=1.0), dict(lam=0),
               dict(lam_scale=0), dict(max_rank=0), dict(polish_tol=0),
               dict(rsvd_init_ratio=0)]:
        with pytest.raises(ValueError):
            StablePCP(sigma=1, **kw).fit(M)
    with pytest.raises(ValueError, match="empty"):
        StablePCP(sigma=1).fit(np.zeros((0, 5)))
    with pytest.raises(RuntimeError, match="not fitted"):
        StablePCP(sigma=1).get_low_rank()


def test_nonconvergence_warns():
    L0, S0, M = make_data()
    with pytest.warns(UserWarning, match="did not converge"):
        m = StablePCP(sigma=0.1, max_iter=2).fit(M)
    assert not m.converged_ and m.n_iter_ == 2


def test_mu_init_is_spectral_norm():
    L0, S0, M = make_data(150, 60)
    m = StablePCP(sigma=0.1).fit(M)
    assert np.isclose(m.mu_init_, 0.99 * np.linalg.norm(M, 2), rtol=1e-10)


def test_continuation_off_and_verbose(capsys):
    L0, S0, M = make_data()
    a = StablePCP(sigma=0.1).fit(M)
    b = StablePCP(sigma=0.1, continuation=False, verbose=True).fit(M)
    assert b.converged_ and rel(b.L_, a.L_) < 1e-5
    assert "iter" in capsys.readouterr().out


def test_randomized_polish_path_low_rank():
    # large sigma -> few surviving components -> partial SVDs are used
    L0, S0, M = make_data(500, 500, 3, sigma=0.1)
    e = StablePCP(sigma=1.0, svd="exact").fit(M)
    r = StablePCP(sigma=1.0, svd="randomized", random_state=0).fit(M)
    assert e.rank_ < 0.1 * 500 and r.converged_
    assert rel(r.L_, e.L_) < 1e-6


def test_max_rank_same_for_all_svd_modes():
    L0, S0, M = make_data()
    a = StablePCP(sigma=0.1, max_rank=5, max_iter=3000, svd="auto").fit(M)
    e = StablePCP(sigma=0.1, max_rank=5, max_iter=3000, svd="exact").fit(M)
    assert a.rank_ <= 5 and rel(a.L_, e.L_) < 1e-8


def test_float32_and_list_input():
    L0, S0, M = make_data(60, 40, 2)
    a = StablePCP(sigma=0.1).fit(M)
    b = StablePCP(sigma=0.1).fit(M.astype(np.float32))
    c = StablePCP(sigma=0.1).fit(M.tolist())
    assert rel(b.L_, a.L_) < 1e-5 and rel(c.L_, a.L_) < 1e-12


@pytest.mark.parametrize("scale", [1e-200, 1e-50, 1e50, 1e200])
def test_extreme_magnitudes(scale):
    L0, S0, M = make_data(60, 40, 2)
    a = StablePCP(sigma=0.1).fit(M)
    b = StablePCP(sigma=0.1 * scale).fit(M * scale)
    assert b.converged_ and np.isfinite(b.L_).all()
    assert rel(b.L_ / scale, a.L_) < 1e-10 and rel(b.S_ / scale, a.S_) < 1e-10


@pytest.mark.filterwarnings("ignore:.*classified as outliers")
@pytest.mark.parametrize("X", [np.zeros((30, 20)), np.full((30, 20), 7.0),
                               np.arange(10.0)[None, :], np.arange(10.0)[:, None]])
def test_degenerate_shapes_match_exact(X):
    a = StablePCP(sigma=0.1).fit(X)
    e = StablePCP(sigma=0.1, svd="exact").fit(X)
    assert a.converged_
    assert np.allclose(a.L_, e.L_, atol=1e-10) and np.allclose(a.S_, e.S_, atol=1e-10)


def test_slow_regime_reaches_tol():
    # small lam_scale + small sigma converges slowly; must not stop early
    L0, S0, M = make_data(120, 60, 3)
    with pytest.warns(UserWarning, match="outliers"):
        ref = StablePCP(sigma=0.025, lam_scale=0.25, tol=1e-12,
                        max_iter=100000).fit(M)
    with pytest.warns(UserWarning, match="outliers"):
        m = StablePCP(sigma=0.025, lam_scale=0.25).fit(M)
    assert m.converged_ and m.sparse_fraction_ > 0.5
    # tol is relative to ||M||: check the error on that scale
    assert np.linalg.norm(m.L_ - ref.L_) / np.linalg.norm(M) < 1e-6


def test_no_sparse_warning_on_normal_data():
    L0, S0, M = make_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = StablePCP(sigma=0.1).fit(M)
    assert m.sparse_fraction_ < 0.2



def toy_data(seed=123):
    rng = np.random.default_rng(seed)
    L = np.ones((50, 50)) * rng.integers(0, 10, size=50)
    L[:25] += np.ones((25, 50)) * rng.integers(-4, 4, size=50)
    S = (-100 * rng.binomial(1, 0.1, (50, 50))
         + 100 * rng.binomial(1, 0.1, (50, 50))).astype(float)
    return L, S, L + S


def test_tiny_sigma_large_outliers_converges():
    # block-coordinate steps crawl here; auto must switch to APG
    L0, S0, M = toy_data()
    m = StablePCP(sigma=1e-4).fit(M)
    assert m.converged_ and m.switched_at_ is not None
    assert np.sum((m.L_ - L0) ** 2) < 1e-3


@pytest.mark.parametrize("sigma", [1e-2, 0.1])
def test_solvers_agree(sigma):
    L0, S0, M = toy_data()
    a = StablePCP(sigma=sigma, solver="apg").fit(M)
    b = StablePCP(sigma=sigma, solver="auto").fit(M)
    c = StablePCP(sigma=sigma, solver="bcd").fit(M)
    assert a.converged_ and b.converged_ and c.converged_
    assert np.linalg.norm(a.L_ - b.L_) / np.linalg.norm(M) < 1e-6
    assert np.linalg.norm(c.L_ - b.L_) / np.linalg.norm(M) < 1e-6


def test_auto_does_not_switch_when_bcd_is_fast():
    L0, S0, M = make_data()
    m = StablePCP(sigma=0.1).fit(M)
    assert m.switched_at_ is None


def test_solver_validation():
    M = np.ones((5, 5))
    for kw in [dict(solver="newton"), dict(eta_apg=1.5), dict(switch_rate=0)]:
        with pytest.raises(ValueError):
            StablePCP(sigma=1, **kw).fit(M)


def test_apg_with_randomized_svd():
    L0, S0, M = make_data(500, 500, 3, sigma=0.1)
    e = StablePCP(sigma=1.0, solver="apg", svd="exact").fit(M)
    r = StablePCP(sigma=1.0, solver="apg", svd="randomized", random_state=0).fit(M)
    assert e.converged_ and r.converged_
    assert np.linalg.norm(r.L_ - e.L_) / np.linalg.norm(M) < 1e-6


def test_apg_without_continuation():
    L0, S0, M = make_data()
    a = StablePCP(sigma=0.1).fit(M)
    b = StablePCP(sigma=0.1, solver="apg", continuation=False, max_iter=20000).fit(M)
    assert b.converged_
    assert np.linalg.norm(b.L_ - a.L_) / np.linalg.norm(M) < 1e-6


def test_max_rank_solvers_agree():
    L0, S0, M = toy_data()
    M = M + np.random.default_rng(0).normal(0, 1, M.shape)
    a = StablePCP(sigma=1.0, max_rank=2, solver="bcd").fit(M)
    b = StablePCP(sigma=1.0, max_rank=2, solver="apg").fit(M)
    assert a.rank_ <= 2 and b.rank_ <= 2
    assert np.linalg.norm(a.L_ - b.L_) / np.linalg.norm(M) < 1e-6


def test_soft_threshold():
    from spcp._stablepcp import _soft
    X = np.array([[-3.0, -0.5, 0.0], [0.5, 2.0, 5.0]])
    expected = np.array([[-2.0, 0.0, 0.0], [0.0, 1.0, 4.0]])
    assert np.array_equal(_soft(X, 1.0), expected)
    Y = X.copy()
    out = _soft(Y, 1.0, out=Y)
    assert out is Y and np.array_equal(Y, expected)


@pytest.mark.parametrize("solver", ["bcd", "apg"])
@pytest.mark.parametrize("shape", [(120, 80), (40, 300)])
def test_row_blocks_do_not_change_results(monkeypatch, solver, shape):
    # tests use small matrices that fit in one block; force many blocks
    import spcp._stablepcp as sp
    L0, S0, M = make_data(*shape, r=3)
    a = StablePCP(sigma=0.1, solver=solver).fit(M)
    monkeypatch.setattr(sp, "_BLOCK", 7)          # rows of a few entries
    b = StablePCP(sigma=0.1, solver=solver).fit(M)
    assert a.n_iter_ == b.n_iter_
    assert np.allclose(a.L_, b.L_, rtol=0, atol=1e-10)
    assert np.allclose(a.S_, b.S_, rtol=0, atol=1e-10)
    assert np.allclose(a.noise_, M - a.L_ - a.S_, atol=1e-10)
