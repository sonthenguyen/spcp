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


def test_rejects_inf_and_all_missing():
    M = np.ones((5, 5)); M[0, 0] = np.inf
    with pytest.raises(ValueError, match="inf"):
        StablePCP(sigma=1).fit(M)
    with pytest.raises(ValueError, match="observed"):
        StablePCP(sigma=1).fit(np.full((5, 5), np.nan))


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
    a = StablePCP(sigma=0.1, tol=1e-9).fit(M)
    b = StablePCP(sigma=0.1, solver="apg", continuation=False, max_iter=20000, tol=1e-9).fit(M)
    assert b.converged_
    assert np.linalg.norm(b.L_ - a.L_) / np.linalg.norm(M) < 1e-6


def test_max_rank_solvers_agree():
    L0, S0, M = toy_data()
    M = M + np.random.default_rng(0).normal(0, 1, M.shape)
    a = StablePCP(sigma=1.0, max_rank=2, solver="bcd", tol=1e-9).fit(M)
    b = StablePCP(sigma=1.0, max_rank=2, solver="apg", tol=1e-9).fit(M)
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


# ----------------------------------------------------------------- debias
def test_debias_off_by_default_and_unchanged():
    import importlib.util
    L0, S0, M = make_data()
    m = StablePCP(sigma=0.1).fit(M)
    assert m.debias is False and m.debias_rank_ is None
    m2 = StablePCP(sigma=0.1, debias=False).fit(M)
    assert np.array_equal(m.L_, m2.L_) and np.array_equal(m.S_, m2.S_)


def test_debias_recovers_rank_and_improves_L():
    L0, S0, M = make_data()
    a = StablePCP(sigma=0.1).fit(M)
    b = StablePCP(sigma=0.1, debias=True).fit(M)
    assert b.debias_rank_ == 5 and b.rank_ == 5
    assert rel(b.L_, L0) < rel(a.L_, L0)
    assert np.allclose(b.noise_, M - b.L_ - b.S_)


def busy_area_counts(seed=3):
    # hourly counts for 30 areas with two rhythms; area 0 is far busier ("airport")
    rng = np.random.default_rng(seed)
    t = np.arange(1000)
    P = np.column_stack([1 + 0.8 * np.sin(2 * np.pi * t / 24), 1 + 0.6 * np.sin(2 * np.pi * t / 168)])
    W = rng.uniform(1, 8, (30, 2))
    W[0] = [120, 60]
    rate = P @ W.T
    C = rng.poisson(rate).astype(float)
    return 2 * np.sqrt(rate + 3 / 8), 2 * np.sqrt(C + 3 / 8)


def test_debias_removes_false_outliers_in_busy_column():
    X_true, X = busy_area_counts()
    a = StablePCP(sigma=1.0).fit(X)
    b = StablePCP(sigma=1.0, debias=True).fit(X)
    assert b.debias_rank_ == 2
    assert np.mean(b.S_[:, 0] != 0) < np.mean(a.S_[:, 0] != 0) - 0.05
    assert rel(b.L_, X_true) < rel(a.L_, X_true)


def test_debias_keeps_real_outliers():
    L0, S0, M = make_data()
    b = StablePCP(sigma=0.1, debias=True).fit(M)
    big = np.abs(S0) > 5
    assert np.mean(b.S_[big] != 0) > 0.95


def test_debias_respects_max_rank_and_validation():
    L0, S0, M = make_data()
    b = StablePCP(sigma=0.1, debias=True, max_rank=3, max_iter=3000).fit(M)
    assert b.rank_ <= 3
    for kw in [dict(debias_iter=0), dict(debias_rank_factor=-1), dict(debias_mask_factor=-1)]:
        with pytest.raises(ValueError):
            StablePCP(sigma=0.1, debias=True, **kw).fit(M)


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_debias_scale_invariant(scale):
    L0, S0, M = make_data(60, 40, 2)
    a = StablePCP(sigma=0.1, debias=True).fit(M)
    b = StablePCP(sigma=0.1 * scale, debias=True).fit(M * scale)
    assert rel(b.L_ / scale, a.L_) < 1e-8


def test_debias_mask_all_option_runs_and_differs():
    L0, S0, M = make_data()
    a = StablePCP(sigma=0.1, debias=True).fit(M)
    b = StablePCP(sigma=0.1, debias=True, debias_mask_factor=0).fit(M)
    assert b.debias_rank_ == a.debias_rank_ == 5
    assert rel(b.L_, L0) < 0.03 and not np.array_equal(a.L_, b.L_)


# ---------------------------------------------------------------- tol="auto", warm start
def _ws_data(seed=0, sig=1.0):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(300, 3)) @ rng.normal(size=(3, 80)) * 3
    S = np.where(rng.random((300, 80)) < 0.05, 20.0, 0.0)
    return L + S + rng.normal(0, sig, (300, 80))


def test_tol_auto_close_to_tight_solution():
    M = _ws_data()
    ref = StablePCP(sigma=1.0, tol=1e-10, max_iter=20000).fit(M)
    a = StablePCP(sigma=1.0, tol="auto").fit(M)
    assert a.converged_ and a.tol_ > 1e-7
    assert np.sqrt(np.mean((a.L_ - ref.L_) ** 2)) < 1e-2          # << sigma
    with pytest.raises(ValueError):
        StablePCP(sigma=1.0, tol="fast").fit(M)


def test_warm_start_same_answer_fewer_iterations():
    M = _ws_data()
    base = StablePCP(sigma=1.0, debias=True).fit(M)
    Mb = M.copy(); Mb[10:14, 5] += 8.0
    cold = StablePCP(sigma=1.0, debias=True).fit(Mb)
    warm = StablePCP(sigma=1.0, debias=True).fit(Mb, init=base.solution_)
    assert np.allclose(warm.L_, cold.L_, atol=1e-5)
    assert warm.n_iter_ <= cold.n_iter_
    with pytest.raises(ValueError):
        StablePCP(sigma=1.0).fit(Mb, init=(np.zeros((2, 2)), np.zeros((2, 2))))


def test_tol_scale_trades_precision_for_speed():
    M = _ws_data()
    ref = StablePCP(sigma=1.0, tol=1e-10, max_iter=20000).fit(M)
    base = StablePCP(sigma=1.0).fit(M)
    rough = StablePCP(sigma=1.0, tol_scale=100.0).fit(M)
    fine = StablePCP(sigma=1.0, tol_scale=0.01).fit(M)
    assert rough.tol_ == pytest.approx(100.0 * base.tol_) and fine.tol_ == pytest.approx(0.01 * base.tol_)
    assert rough.n_iter_ < base.n_iter_ < fine.n_iter_
    assert rough.converged_ and fine.converged_
    err = lambda m: np.sqrt(np.mean((m.L_ - ref.L_) ** 2))
    assert err(fine) < err(base) < err(rough) < 0.5            # the rough fit is still well below sigma


def test_tol_scale_default_floor_and_validation():
    M = _ws_data()
    a = StablePCP(sigma=1.0).fit(M)
    b = StablePCP(sigma=1.0, tol_scale=1.0).fit(M)
    assert np.array_equal(a.L_, b.L_) and a.n_iter_ == b.n_iter_
    assert StablePCP(sigma=1.0, tol_scale=1e-12, max_iter=50).fit(M).tol_ == 1e-7   # never below the floor
    c = StablePCP(sigma=1.0, tol=1e-5, tol_scale=100.0).fit(M)                      # no effect on a numeric tol
    assert c.tol_ == 1e-5
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="tol_scale"):
            StablePCP(sigma=1.0, tol_scale=bad).fit(M)


def test_rough_warm_started_refit_follows_changed_data():
    M = _ws_data()
    base = StablePCP(sigma=1.0).fit(M)
    Mb = M.copy(); Mb[10:14, 5] += 8.0
    exact = StablePCP(sigma=1.0).fit(Mb)
    quick = StablePCP(sigma=1.0, tol_scale=100.0).fit(Mb, init=base.solution_)
    assert quick.n_iter_ < exact.n_iter_
    assert np.sqrt(np.mean((quick.L_ - exact.L_) ** 2)) < 0.05


def test_solution_is_raw_fit_when_debiased():
    M = _ws_data()
    plain = StablePCP(sigma=1.0).fit(M)
    deb = StablePCP(sigma=1.0, debias=True).fit(M)
    assert np.allclose(deb.solution_[0], plain.L_)
    assert plain.solution_[0] is plain.L_


# ---------------------------------------------------------------- missing values
def _miss_data(seed=0, frac=0.3, outliers=0.0):
    rng = np.random.default_rng(seed)
    L0 = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 60)) * 3
    S0 = np.where(rng.random(L0.shape) < outliers, 25.0, 0.0)
    M = L0 + S0 + rng.normal(0, 1, L0.shape)
    miss = rng.random(L0.shape) < frac
    Mm = M.copy(); Mm[miss] = np.nan
    return L0, S0, M, Mm, miss


def test_missing_cells_are_filled_from_L():
    for outliers in (0.0, 0.05):
        L0, S0, M, Mm, miss = _miss_data(outliers=outliers)
        m = StablePCP(sigma=1.0, debias=True).fit(Mm)
        rmse_missing = np.sqrt(np.mean((m.L_[miss] - L0[miss]) ** 2))
        assert rmse_missing < 0.6                       # well below the noise (1.0)
        assert np.all(m.S_[miss] == 0) and np.all(np.isnan(m.noise_[miss]))
        assert np.all(np.isfinite(m.noise_[~miss]))
        if outliers:
            assert np.mean(m.S_[(S0 != 0) & ~miss] != 0) > 0.95


def test_missing_input_not_modified_and_no_nan_unchanged():
    L0, S0, M, Mm, miss = _miss_data()
    before = Mm.copy()
    StablePCP(sigma=1.0).fit(Mm)
    assert np.array_equal(np.isnan(Mm), np.isnan(before))
    a = StablePCP(sigma=1.0).fit(M)
    assert a.missing_ is None


def test_missing_apg_raises():
    L0, S0, M, Mm, miss = _miss_data()
    with pytest.raises(ValueError, match="solver"):
        StablePCP(sigma=1.0, solver="apg").fit(Mm)


def test_missing_values_work_with_a_warm_start():
    L0, S0, M, Mm, miss = _miss_data()
    cold = StablePCP(sigma=1.0).fit(Mm)
    warm = StablePCP(sigma=1.0).fit(Mm, init=cold.solution_)
    assert np.allclose(warm.L_, cold.L_, atol=1e-2) and warm.n_iter_ <= cold.n_iter_   # same to solver tolerance, far below sigma
    assert np.array_equal(warm.missing_, miss)


# ---------------------------------------------------------------- intervals: conformal
def _small_data(seed=0, m=60, n=50, r=2, sig=1.0):
    rng = np.random.default_rng(seed)
    L0 = rng.normal(size=(m, r)) @ rng.normal(size=(r, n)) * 3
    S0 = np.where(rng.random((m, n)) < 0.05, 20.0, 0.0)
    return L0, S0, L0 + S0 + rng.normal(0, sig, (m, n))


def test_conformal_outputs_and_coverage():
    rng = np.random.default_rng(0)
    m, n = 120, 40
    L0 = rng.normal(size=(m, 2)) @ rng.normal(size=(2, n)) * 3
    M = L0 + rng.normal(0, 1, (m, n))
    wins, truth = [], []
    for k in range(30):
        r0, j = 4 * k, k % n
        w = (slice(r0, r0 + 4), j)
        truth.append(M[w].sum())
        wins.append(w)
    Mp = M.copy()
    for w in wins:
        Mp[w] += 8.0
    mod = StablePCP(sigma=1.0, debias=True).fit(Mp)
    res = mod.conformal(wins, level=0.9, n_cal=120, per_refit=30, random_state=0)
    assert res["estimate"].shape == (30,) and len(res["cal_errors"]) == 120
    assert np.all(res["low"] <= res["high"])
    cov = np.mean((np.array(truth) >= res["low"]) & (np.array(truth) <= res["high"]))
    assert cov >= 0.75
    with pytest.raises(ValueError):
        mod.conformal(wins, level=2)


def test_conformal_n_groups():
    rng = np.random.default_rng(0)
    m, n = 160, 40
    M = np.abs(rng.normal(size=(m, 2)) @ rng.normal(size=(2, n))) * 5 + 10 + rng.normal(0, 1, (m, n))
    wins = [(slice(4 * k, 4 * k + 4), k % n) for k in range(30)]
    mod = StablePCP(sigma=1.0, debias=True).fit(M)
    r = mod.conformal(wins, n_cal=180, per_refit=30, n_groups=3, scale=lambda c: max(c, 1), random_state=0)
    assert r["low"].shape == (30,) and np.all(r["low"] <= r["high"])
    assert len(r["cal_estimates"]) == 180
    with pytest.raises(ValueError):
        mod.conformal(wins, n_cal=40, per_refit=30, n_groups=3, random_state=0)


def test_conformal_hidden_block():
    rng = np.random.default_rng(3)
    T, N = 160, 40
    M0 = rng.normal(size=(T, 2)) @ rng.normal(size=(2, N)) * 2 + rng.normal(0, 1, (T, N))
    Mm = M0.copy(); Mm[100:, 0] = np.nan                 # hide one unit's last 60 periods
    m = StablePCP(sigma=1.0, debias=True).fit(Mm)
    w = (slice(100, T), 0)
    r = m.conformal([w], candidates="same_rows", n_cal=60,
                    statistic=lambda V, ww: float(np.mean(V[ww])), random_state=0)
    assert r["low"][0] < r["estimate"][0] < r["high"][0]
    assert r["low"][0] <= M0[w].mean() <= r["high"][0]


def test_conformal_mixed_hidden_and_observed_windows_raise():
    L0, S0, M, Mm, miss = _miss_data()
    m = StablePCP(sigma=1.0).fit(Mm)
    r, c = np.argwhere(miss)[0]
    r2, c2 = np.argwhere(~miss)[0]
    with pytest.raises(ValueError, match="hidden"):
        m.conformal([(r, c), (r2, c2)])


def test_intervals_need_sigma_and_a_fit():
    L0, S0, M = _small_data()
    with pytest.raises(RuntimeError, match="not fitted"):
        StablePCP(sigma=1.0).conformal([(0, 0)])
    m = StablePCP(mu=10.0, lam=0.1, tol=1e-6).fit(M)            # mu given directly, no sigma
    with pytest.raises(ValueError, match="sigma"):
        m.conformal([(0, 0)])


# ---------------------------------------------------------------- sigma="auto" and sigma=0
@pytest.mark.parametrize("true", [0.05, 0.5, 3.0])
def test_sigma_auto_recovers_a_known_noise_level(true):
    L0, S0, M = make_data(m=150, n=120, r=4, sigma=true, seed=3)
    mod = StablePCP(sigma="auto", debias=True).fit(M)
    assert abs(mod.sigma_ / true - 1) < 0.12
    # as accurate as a fit that is told the true noise level
    oracle = StablePCP(sigma=true, debias=True).fit(M)
    assert mod.rank_ == oracle.rank_ == 4
    assert rel(mod.L_, L0) <= 1.1 * rel(oracle.L_, L0) + 0.01
    tried, measured = mod.sigma_path_[-1]
    assert tried == mod.sigma_ and abs(measured - tried) <= 0.02 * tried
    # a fixed sigma is reported as given, with no path
    fixed = StablePCP(sigma=true).fit(M)
    assert fixed.sigma_ == true and fixed.sigma_path_ is None
    assert StablePCP(mu=fixed.mu_, tol=1e-6).fit(M).sigma_ is None


def test_sigma_auto_gives_the_same_fit_as_passing_that_sigma():
    L0, S0, M = make_data(m=120, n=100, r=3, sigma=0.3, seed=5)
    auto = StablePCP(sigma="auto").fit(M)
    direct = StablePCP(sigma=auto.sigma_).fit(M)
    assert auto.rank_ == direct.rank_
    assert rel(auto.L_, direct.L_) < 1e-3 and rel(auto.S_, direct.S_) < 1e-3


def test_sigma_auto_does_not_depend_on_units():
    L0, S0, M = make_data(m=100, n=90, r=3, sigma=0.2, seed=7)
    a = StablePCP(sigma="auto").fit(M)
    b = StablePCP(sigma="auto").fit(1e4 * M)
    c = StablePCP(sigma="auto").fit(1e-5 * M)
    assert abs(b.sigma_ / (1e4 * a.sigma_) - 1) < 1e-3
    assert abs(c.sigma_ / (1e-5 * a.sigma_) - 1) < 1e-3
    assert rel(b.L_ / 1e4, a.L_) < 1e-3


def test_sigma_auto_with_missing_values():
    L0, S0, M = make_data(m=150, n=120, r=4, sigma=0.5, seed=9)
    Mm = M.copy()
    Mm[np.random.default_rng(0).random(M.shape) < 0.2] = np.nan
    mod = StablePCP(sigma="auto", debias=True).fit(Mm)
    assert abs(mod.sigma_ / 0.5 - 1) < 0.15
    assert np.all(np.isfinite(mod.L_)) and mod.rank_ == 4


def test_sigma_auto_is_estimated_once_and_reused_by_refits(monkeypatch):
    L0, S0, M = _small_data()
    mod = StablePCP(sigma="auto", debias=True).fit(M)
    assert abs(mod.sigma_ - 1.0) < 0.15
    same = StablePCP(sigma=mod.sigma_, debias=True).fit(M)
    w = [(slice(5, 8), 3), (slice(20, 23), 7)]
    ref_c = same.conformal(w, n_cal=40, random_state=0)

    def no_more(self, M):
        raise AssertionError("sigma was estimated again in a refit")
    monkeypatch.setattr(StablePCP, "_auto_sigma", no_more)
    c = mod.conformal(w, n_cal=40, random_state=0)
    assert mod.sigma == "auto"                       # the setting itself is kept
    np.testing.assert_allclose(c["low"], ref_c["low"], rtol=0.05, atol=0.05)
    np.testing.assert_allclose(c["high"], ref_c["high"], rtol=0.05, atol=0.05)


def test_sigma_auto_on_noise_free_data_uses_sigma_zero():
    L0, S0, M = make_data(m=80, n=80, r=3, sigma=0.0, seed=11)
    mod = StablePCP(sigma="auto").fit(M)
    assert mod.sigma_ == 0 and len(mod.sigma_path_) <= 3
    assert rel(mod.L_, L0) < 1e-4


def _video_like(seed=0, frames=60, h=20, w=30):
    """Integer frames: a fixed background, a brightness step half way, and a
    bright square moving across. Most pixels repeat exactly: no dense noise."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(40, 200, size=(h, w)).astype(float)
    V = np.repeat(bg[None], frames, 0)
    V[frames // 2:] += 10.0
    fg = np.zeros_like(V, bool)
    for f in range(frames):
        c = (2 * f) % (w - 5)
        V[f, 8:13, c:c + 5] = 250.0
        fg[f, 8:13, c:c + 5] = True
    return V.reshape(frames, -1), fg.reshape(frames, -1)


def test_sigma_auto_on_video_like_data_stops_fast_and_finds_the_foreground():
    M, fg = _video_like()
    mod = StablePCP(sigma="auto").fit(M)
    assert mod.sigma_ == 0 and len(mod.sigma_path_) <= 3
    found = np.abs(mod.S_) > 1.0
    assert found[fg].mean() > 0.9 and found[~fg].mean() < 0.01
    assert mod.rank_ <= 3


def test_sigma_zero_is_exact_and_handles_missing_values():
    L0, S0, M = make_data(m=100, n=90, r=3, sigma=0.0, seed=13)
    mod = StablePCP(sigma=0).fit(M)
    assert mod.converged_ and mod.mu_ == 0 and mod.debias_rank_ is None
    assert rel(mod.L_, L0) < 1e-4 and rel(mod.S_, S0) < 1e-3
    assert np.all(mod.noise_ == 0)
    Mm = M.copy()
    hide = np.random.default_rng(1).random(M.shape) < 0.2
    Mm[hide] = np.nan
    mm = StablePCP(sigma=0, debias=True).fit(Mm)
    assert rel(mm.L_[hide], L0[hide]) < 1e-3
    assert np.isnan(mm.noise_[hide]).all() and np.all(mm.S_[hide] == 0)
    # warm start reaches the same answer
    again = StablePCP(sigma=0).fit(M, init=mod.solution_)
    assert rel(again.L_, mod.L_) < 1e-4


def test_sigma_zero_intervals():
    L0, S0, M = make_data(m=60, n=40, r=2, sigma=0.0, seed=17)
    mod = StablePCP(sigma=0).fit(M)
    w = [(slice(5, 8), 3), (slice(20, 23), 7)]
    with pytest.raises(ValueError, match="units of sigma"):
        mod.conformal(w, n_cal=20, effect=(2, 5))
    r = mod.conformal(w, n_cal=40, random_state=0)
    assert np.all(np.isfinite(r["low"])) and np.all(r["low"] <= r["high"])


def test_sigma_auto_rejects_bad_input():
    with pytest.raises(ValueError):
        StablePCP(sigma="estimate").fit(np.ones((5, 5)))
    with pytest.raises(ValueError, match=">= 0"):
        StablePCP(sigma=-1.0).fit(np.ones((5, 5)))
    with pytest.raises(ValueError, match="no variation"):
        StablePCP(sigma="auto").fit(np.full((6, 6), 2.5))


def test_sigma_auto_one_dip_does_not_mean_noise_free(monkeypatch):
    """A single collapse of the measured noise is not enough: the search
    confirms one step lower, and real noise reappears there."""
    import spcp._stablepcp as sp
    calls = []

    def scripted(res):                    # true noise 0.5, except one odd low reading
        calls.append(1)
        return 0.01 if len(calls) == 2 else 0.5
    monkeypatch.setattr(sp, "_typical_noise", scripted)
    L0, S0, M = make_data(m=80, n=60, r=3, sigma=0.5, seed=21)
    mod = StablePCP(sigma="auto").fit(M)
    assert abs(mod.sigma_ / 0.5 - 1) <= 0.03
    tried = [a for a, b in mod.sigma_path_]
    assert abs(tried[2] - tried[1] / 5) < 1e-12  # the confirmation step


def _record_trials(monkeypatch):
    """Record every fit made inside the sigma='auto' search."""
    from spcp._stablepcp import _LOG_ABS_SD     # sigma of the measuring fits
    calls, orig = [], StablePCP.fit

    def spy(self, M, init=None):
        out = orig(self, M, init=init)
        if not isinstance(self.sigma, str) and self.sigma != _LOG_ABS_SD:
            calls.append(dict(sigma=self.sigma, init=init is not None,
                              mult=self.tol_scale,
                              converged=self.converged_))
        return out
    monkeypatch.setattr(StablePCP, "fit", spy)
    return calls


@pytest.mark.parametrize("true", [0.3, 1.0])
def test_sigma_auto_fast_search_agrees_with_full_precision_search(true):
    # rough trial rounds (tol="auto") and full-precision rounds (fixed tol)
    # must settle on the same sigma, within the search's own 2% rule
    L0, S0, M = make_data(m=120, n=100, r=3, sigma=true, seed=5)
    fast = StablePCP(sigma="auto").fit(M).sigma_
    full = StablePCP(sigma="auto", tol=1e-7).fit(M).sigma_
    assert abs(fast - full) <= 0.02 * full
    assert abs(fast - true) <= 0.1 * true


def test_sigma_auto_rounds_are_warm_started_and_accepted_at_full_precision(monkeypatch):
    L0, S0, M = make_data(m=120, n=100, r=3, sigma=0.5, seed=6)
    calls = _record_trials(monkeypatch)
    mod = StablePCP(sigma="auto").fit(M)
    trials = calls                              # the final fit is mod itself
    assert len(trials) >= 3
    assert not trials[0]["init"] and all(c["init"] for c in trials[1:])
    assert trials[0]["mult"] > 1.0              # search rounds are rough
    assert trials[-1]["mult"] == 1.0            # the accepted round is exact
    assert trials[-1]["sigma"] == mod.sigma_
    assert mod.converged_


def test_sigma_auto_on_larger_noise_free_data_converges_every_round(monkeypatch):
    # in noise-free data the search tries small sigmas, where a cold fit can
    # crawl; warm starts keep every round converging
    M, fg = _video_like(frames=90, h=40, w=60)
    calls = _record_trials(monkeypatch)
    mod = StablePCP(sigma="auto").fit(M)
    assert mod.sigma_ == 0 and len(mod.sigma_path_) <= 3
    assert all(c["converged"] for c in calls)
    found = np.abs(mod.S_) > 1.0
    assert found[fg].mean() > 0.9 and found[~fg].mean() < 0.01


# ---------------------------------------------------------------- accelerate=True
def test_accelerate_off_by_default():
    M = _ws_data()
    a = StablePCP(sigma=1.0).fit(M)
    b = StablePCP(sigma=1.0, accelerate=False).fit(M)
    assert np.array_equal(a.L_, b.L_) and a.n_iter_ == b.n_iter_ and b.accel_at_ is None


@pytest.mark.filterwarnings("ignore:.*classified as outliers")   # a deliberately slow setting
def test_accelerate_same_answer_in_fewer_steps_on_a_slow_fit():
    L0, S0, M = make_data(200, 200, 3, sigma=0.01)
    kw = dict(sigma=0.01, lam_scale=0.5)
    exact = StablePCP(tol=1e-11, max_iter=50000, **kw).fit(M)
    plain = StablePCP(**kw).fit(M)
    fast = StablePCP(accelerate=True, **kw).fit(M)
    assert fast.converged_ and fast.accel_at_ is not None
    assert fast.n_iter_ < plain.n_iter_ / 2
    err = lambda m: np.sqrt(np.mean((m.L_ - exact.L_) ** 2))
    assert err(fast) <= 2 * err(plain) and err(fast) < 1e-3 * 0.01       # far below the noise


def test_accelerate_with_missing_values():
    L0, S0, M, Mm, miss = _miss_data()
    exact = StablePCP(sigma=1.0, tol=1e-11, max_iter=50000).fit(Mm)
    plain = StablePCP(sigma=1.0).fit(Mm)
    fast = StablePCP(sigma=1.0, accelerate=True).fit(Mm)
    assert fast.converged_ and fast.n_iter_ < plain.n_iter_
    err = lambda m: np.sqrt(np.mean((m.L_ - exact.L_) ** 2))
    assert err(fast) <= 2 * err(plain)
    assert np.all(fast.S_[miss] == 0) and np.all(np.isnan(fast.noise_[miss]))


def test_accelerate_leaves_fits_that_are_about_to_stop_unchanged():
    M = _ws_data()
    base = StablePCP(sigma=1.0).fit(M)
    Mb = M.copy(); Mb[10:14, 5] += 8.0
    for X, init in ((M, None), (Mb, base.solution_)):          # a quick fit and a warm refit
        a = StablePCP(sigma=1.0).fit(X, init=init)
        b = StablePCP(sigma=1.0, accelerate=True).fit(X, init=init)
        assert b.accel_at_ is None
        assert np.array_equal(a.L_, b.L_) and a.n_iter_ == b.n_iter_


def test_accelerate_has_no_effect_with_apg_randomized_or_sigma_zero():
    M = make_data()[2]
    X0 = make_data(60, 40, 2, sigma=0.0, seed=17)[2]
    for X, kw in ((M, dict(sigma=0.1, solver="apg")),
                  (M, dict(sigma=0.1, svd="randomized", random_state=0)),
                  (X0, dict(sigma=0))):
        a = StablePCP(**kw).fit(X)
        b = StablePCP(accelerate=True, **kw).fit(X)
        assert b.accel_at_ is None
        assert np.array_equal(a.L_, b.L_) and a.n_iter_ == b.n_iter_


@pytest.mark.filterwarnings("ignore:.*classified as outliers")   # a deliberately slow setting
def test_accelerate_same_work_for_any_memory_layout():
    # the extrapolation must not depend on whether the input is row- or column-ordered
    L0, S0, M = make_data(200, 200, 3, sigma=0.01)
    kw = dict(sigma=0.01, lam_scale=0.5, accelerate=True)
    c = StablePCP(**kw).fit(np.ascontiguousarray(M))
    f = StablePCP(**kw).fit(np.asfortranarray(M))
    s = StablePCP(**kw).fit(np.asfortranarray(M)[:150])           # a column-ordered slice
    r = StablePCP(**kw).fit(np.ascontiguousarray(M[:150]))
    assert c.accel_at_ is not None and c.n_iter_ == f.n_iter_ and s.n_iter_ == r.n_iter_
    np.testing.assert_allclose(c.L_, f.L_, atol=1e-10)


def test_accelerate_is_used_by_the_sigma_search_and_conformal_refits(monkeypatch):
    from spcp._stablepcp import _LOG_ABS_SD         # sigma of the noise-measuring fits
    seen, orig = [], StablePCP.fit

    def spy(self, M, init=None):
        if self.sigma != _LOG_ABS_SD:
            seen.append(self.accelerate)
        return orig(self, M, init=init)
    monkeypatch.setattr(StablePCP, "fit", spy)
    M = _small_data()[2]
    m = StablePCP(sigma="auto", accelerate=True).fit(M)
    m.conformal([(slice(5, 8), 3)], n_cal=20, random_state=0)
    assert len(seen) > 3 and all(seen)

