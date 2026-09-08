"""Tests for evt_lab.

Same two kinds as in the RMT lab:

1. **Invariants** — statements with an exact right answer (closed-form PWM on the
   exponential, transform round-trips, Pareto tail integrals, the Kupiec and
   Christoffersen algebra, homogeneity of ES).
2. **Seeded statistical checks** — Monte Carlo against the theory the module
   implements (Hill recovers alpha, kappa recovers 2 - alpha, the GPD limit of
   Student-t exceedances, the extremal index of a GARCH, the invariance of the
   modified scale across thresholds). Fixed seed.

There is a third kind here that the RMT lab did not need: **falsification tests**.
Several functions exist in order to be shown wrong — the log-log OLS exponent, the
Gaussian ES on fat-tailed data, Kupiec's blindness to the size of an exception. We
assert that they fail.

Run with:  pytest -q
"""
import numpy as np
import pytest

from evt_lab import (
    sample_pareto, sample_student, sample_stable, sample_gpd, sample_gev,
    sample_lognormal, variance_preserving_mixture, garch_returns,
    garch_kesten_alpha, t_copula_uniforms,
    mad, mad_over_std, max_to_sum_ratio, kappa_mc, kappa_bootstrap,
    kurtosis_under_aggregation, survival_empirical, local_alpha,
    excess_conditional_ratio,
    block_maxima, fit_gev, gev_return_level, gumbel_norming,
    mean_excess, exceedances, fit_gpd, fit_gpd_pwm, gpd_profile_ci,
    threshold_stability, pot_var_es,
    hill, hill_plot, pickands, dedh_moment, loglog_ols_alpha,
    pareto_mle_alpha, alpha_hat_moments,
    pareto_mean_above, expected_max_pareto, hidden_mean_fraction,
    dual_transform, dual_inverse, shadow_mean, quantile_contribution,
    var_es_empirical, var_es_gaussian, var_es_student, var_es_ewma, var_es_gpd,
    es_var_ratio, alpha_from_es_var_ratio, es_sigma_ratio,
    extremal_index, decluster, stationary_bootstrap,
    drawdown, max_drawdown, growth_rate, kelly_fraction, kelly_empirical,
    barbell, leverage_for_es, simulate_wealth, ruin_probability,
    var_hits, kupiec_pof, christoffersen_independence, christoffersen_cc,
    acerbi_szekely_z2,
    sample_cov, sample_corr, eigh_desc, recompose, mp_edges, xi_clip,
    lw_shrinkage, xi_rie, clean_cov, min_var_weights, day_scale,
)

SEED = 20260907


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
#  1. Invariants — exact answers
# --------------------------------------------------------------------------- #
def test_gpd_pwm_is_exact_on_the_exponential():
    """The exponential is GPD(xi=0, beta): a0 = 1, a1 = 1/4, so PWM must return
    exactly (0, 1) in the population. Checked at a large n."""
    y = sample_gpd(0.0, 1.0, 400_000, np.random.default_rng(1))
    xi, beta = fit_gpd_pwm(y)
    assert xi == pytest.approx(0.0, abs=0.01)
    assert beta == pytest.approx(1.0, abs=0.01)


def test_gpd_survival_matches_its_own_sampler():
    """P(Y > y) = (1 + xi y/beta)^(-1/xi) — sampler and law must agree."""
    xi, beta = 0.4, 2.0
    y = sample_gpd(xi, beta, 400_000, np.random.default_rng(2))
    for level in (1.0, 5.0, 20.0):
        emp = np.mean(y > level)
        theo = (1.0 + xi * level / beta) ** (-1.0 / xi)
        assert emp == pytest.approx(theo, rel=0.05)


def test_pareto_mean_above_matches_numeric_integral():
    alpha, u = 2.5, 3.0
    x = sample_pareto(alpha, 2_000_000, np.random.default_rng(3))
    emp = np.mean(x * (x > u))
    assert emp == pytest.approx(pareto_mean_above(alpha, u), rel=0.03)


def test_hidden_fraction_reduces_to_the_closed_form():
    """The ratio at M = E[max of n] is exactly M^(1-alpha)."""
    for alpha in (1.5, 2.0, 3.0):
        m = expected_max_pareto(alpha, 10_000)
        assert hidden_mean_fraction(alpha, 10_000) == pytest.approx(
            m ** (1.0 - alpha), rel=1e-10)


def test_expected_max_grows_as_n_to_the_one_over_alpha():
    a = 3.0
    m1 = expected_max_pareto(a, 1_000)
    m2 = expected_max_pareto(a, 1_000_000)
    assert m2 / m1 == pytest.approx(1000.0 ** (1.0 / a), rel=0.01)


def test_dual_transform_round_trip():
    z = np.array([1.0001, 2.0, 7.5, 9.9])
    assert np.allclose(dual_inverse(dual_transform(z, 10.0, 1.0), 10.0, 1.0), z)


def test_dual_transform_rejects_out_of_range():
    with pytest.raises(ValueError):
        dual_transform(np.array([11.0]), 10.0, 1.0)


def test_es_var_ratio_round_trip():
    for a in (1.5, 2.0, 3.0, 8.0):
        assert alpha_from_es_var_ratio(es_var_ratio(a)) == pytest.approx(a, rel=1e-12)


def test_mad_over_std_is_sqrt_two_over_pi_for_a_gaussian():
    z = np.random.default_rng(4).standard_normal(500_000)
    assert mad_over_std(z) == pytest.approx(np.sqrt(2.0 / np.pi), abs=3e-3)


def test_gaussian_es_over_sigma_is_2665_at_one_percent():
    """Closed form: ES/sigma = phi(z_p)/p = 2.6652 at p = 1%. The number every
    fat-tailed ES in the notebook gets compared against."""
    z = np.random.default_rng(5).standard_normal(400_000)
    assert var_es_gaussian(z, 0.01)["es"] == pytest.approx(2.6652, abs=0.02)


def test_es_is_positively_homogeneous():
    """ES(c X) = c ES(X) — the property leverage_for_es relies on being exact."""
    z = np.abs(np.random.default_rng(6).standard_normal(20_000))
    a = var_es_empirical(z, 0.01)["es"]
    b = var_es_empirical(3.7 * z, 0.01)["es"]
    assert b == pytest.approx(3.7 * a, rel=1e-12)


def test_leverage_for_es_hits_its_target():
    r = -np.abs(np.random.default_rng(7).standard_normal(20_000)) * 0.01
    f = leverage_for_es(r, target_es=0.02, p=0.01)
    assert var_es_empirical(-f * r, 0.01)["es"] == pytest.approx(0.02, rel=1e-9)


def test_variance_preserving_mixture_keeps_variance_and_lifts_kurtosis():
    """At p = 1/2 the construction gives variance sigma^2 and kurtosis 3(1+a^2)
    exactly — the cleanest possible statement that the 2nd moment says nothing
    about the 4th."""
    a = 0.6
    x = variance_preserving_mixture(2_000_000, np.random.default_rng(8), 1.0, a, 0.5)
    assert x.var() == pytest.approx(1.0, rel=0.01)
    assert np.mean(x ** 4) / x.var() ** 2 == pytest.approx(3.0 * (1 + a ** 2), rel=0.03)


def test_max_to_sum_measures_usability_not_existence():
    """R_n(p) -> 0 iff E|X|^p < inf, but the rate is unbounded, so a high R is
    NOT evidence that the moment fails to exist.

    The lognormal is the counterexample that keeps this honest: sigma = 2 has
    every moment finite and still pins R_n(4) near one at n = 200,000, because
    X^4 = e^(8Z) has CV^2 = e^64 - 1. Drop to sigma = 0.5 — same family, same
    "all moments exist" — and it decays like a Gaussian. The scale parameter is
    doing the work, not the moment condition.
    """
    r = np.random.default_rng(9)
    thin = max_to_sum_ratio(r.standard_normal(200_000), 4.0)[-1]
    fat = max_to_sum_ratio(sample_pareto(2.5, 200_000, r), 4.0)[-1]
    ln_mild = max_to_sum_ratio(sample_lognormal(0.5, 200_000, r), 4.0)[-1]
    ln_wild = max_to_sum_ratio(sample_lognormal(2.0, 200_000, r), 4.0)[-1]
    assert thin < 0.01                       # E X^4 exists and is reachable
    assert fat > 0.10                        # E X^4 does not exist
    assert ln_mild < 0.05                    # exists and reachable
    assert ln_wild > 0.10                    # exists and NOT reachable — the point
    assert ln_wild > fat * 0.5               # as bad as a genuine infinite moment


def test_a_finite_fourth_moment_can_be_wholly_unmeasurable():
    """lognormal(2) has kurtosis 9,220,560 exactly. A large sample sees ~1% of it,
    so 'the moment exists' and 'the moment is estimable' are different claims."""
    s = 2.0
    true_kurt = np.exp(4 * s ** 2) + 2 * np.exp(3 * s ** 2) + 3 * np.exp(2 * s ** 2) - 3
    assert true_kurt == pytest.approx(9_220_560, rel=1e-4)
    x = sample_lognormal(s, 1_000_000, np.random.default_rng(12))
    measured = np.mean((x - x.mean()) ** 4) / x.var() ** 2
    assert measured < 0.10 * true_kurt       # off by more than an order of magnitude


def test_sample_kurtosis_cannot_exceed_the_number_of_observations():
    """b2 <= m. The ceiling that makes the naive aggregation test misleading:
    as k rises the block count m = n/k falls, and so does the largest kurtosis
    the estimator is capable of reporting."""
    for m in (10, 100, 1000):
        x = np.zeros(m); x[0] = 1e9
        b2 = np.mean((x - x.mean()) ** 4) / x.var() ** 2
        assert b2 <= m + 1e-6
        assert b2 > 0.8 * m                  # and the bound is essentially attained


def test_fixed_blocks_holds_the_ceiling_constant_across_lags():
    lags = [1, 5, 20]
    x = sample_student(3., 20 * 500, np.random.default_rng(13))
    v = kurtosis_under_aggregation(x, lags, blocks=500)
    assert np.all(np.isfinite(v))
    assert np.all(v <= 500 + 1e-6)
    # too few observations for the requested block count -> nan, never a silent lie
    assert np.isnan(kurtosis_under_aggregation(x, [250], blocks=500)[0])


def test_drawdown_and_max_drawdown_are_consistent():
    w = np.array([1.0, 1.5, 0.9, 1.2, 0.6])
    assert max_drawdown(w) == pytest.approx(0.6 / 1.5 - 1.0)
    assert drawdown(w).max() == pytest.approx(0.0)


def test_growth_rate_is_minus_infinity_on_a_wipeout():
    assert growth_rate(np.array([0.1, -1.0, 0.2]), 1.0) == -np.inf


def test_simulate_wealth_absorbs_at_the_barrier():
    w = simulate_wealth(np.array([-0.9, 5.0, 5.0]), f=1.0, w0=1.0, barrier=0.2)
    assert w[1] == pytest.approx(0.2)
    assert w[-1] == pytest.approx(0.2)       # no resurrection


def test_kupiec_is_zero_when_the_rate_is_exact():
    hits = np.zeros(1000, int)
    hits[:10] = 1
    out = kupiec_pof(hits, 0.01)
    assert out["lr"] == pytest.approx(0.0, abs=1e-9)
    assert out["pvalue"] == pytest.approx(1.0)


def test_christoffersen_independence_flags_a_perfect_cluster():
    """Ten exceptions in a row is the same count as ten spread out — and Kupiec
    cannot tell them apart. The independence test must."""
    n = 1000
    clustered = np.zeros(n, int); clustered[500:510] = 1
    spread = np.zeros(n, int); spread[::100] = 1
    assert kupiec_pof(clustered, 0.01)["hits"] == kupiec_pof(spread, 0.01)["hits"]
    assert christoffersen_independence(clustered)["pvalue"] < 1e-6
    assert christoffersen_independence(spread)["pvalue"] > 0.5


def test_conditional_coverage_is_the_sum_of_its_parts():
    hits = np.zeros(2000, int); hits[::90] = 1
    cc = christoffersen_cc(hits, 0.01)
    assert cc["lr"] == pytest.approx(kupiec_pof(hits, 0.01)["lr"]
                                     + christoffersen_independence(hits)["lr"])


def test_decluster_returns_one_peak_per_cluster():
    x = np.zeros(100)
    x[[10, 11, 12, 60, 61]] = [1.0, 3.0, 2.0, 5.0, 4.0]
    peaks, idx = decluster(x, 0.5, runs=5)
    assert np.array_equal(peaks, np.array([3.0, 5.0]))
    assert np.array_equal(idx, np.array([11, 60]))


def test_gumbel_norming_matches_the_textbook_constants():
    a, b = gumbel_norming(1_000_000)
    assert b == pytest.approx(4.7658, abs=1e-3)
    assert a == pytest.approx(1.0 / b, rel=1e-12)


def test_gev_return_level_inverts_the_gev_quantile():
    xi, mu, sigma, T = 0.2, 1.0, 2.0, 50.0
    z = gev_return_level(xi, mu, sigma, T)
    cdf = np.exp(-(1.0 + xi * (z - mu) / sigma) ** (-1.0 / xi))
    assert cdf == pytest.approx(1.0 - 1.0 / T, rel=1e-10)


def test_block_maxima_shape_and_values():
    x = np.arange(10, dtype=float)
    assert np.array_equal(block_maxima(x, 3), np.array([2.0, 5.0, 8.0]))


def test_exceedances_are_strictly_positive():
    x = np.array([1.0, 2.0, 3.0, 3.0])
    assert np.array_equal(exceedances(x, 2.0), np.array([1.0, 1.0]))


# --------------------------------------------------------------------------- #
#  2. Seeded statistical checks — against the theory implemented
# --------------------------------------------------------------------------- #
def test_hill_recovers_a_known_pareto_alpha(rng):
    for alpha in (1.5, 2.5, 3.0, 4.0):
        x = sample_pareto(alpha, 200_000, rng)
        assert hill(x, 20_000) == pytest.approx(alpha, rel=0.04)


def test_hill_standard_error_matches_alpha_over_sqrt_k():
    """se = alpha/sqrt(k) is asymptotic theory; check the spread of repeated
    estimates actually matches it."""
    ests = [hill(sample_pareto(3.0, 20_000, np.random.default_rng(s)), 1_000)
            for s in range(200)]
    assert np.std(ests) == pytest.approx(3.0 / np.sqrt(1_000), rel=0.25)


def test_pareto_mle_matches_hill_at_the_full_sample(rng):
    x = sample_pareto(2.2, 50_000, rng)
    a_mle, _ = pareto_mle_alpha(x, xm=1.0)
    assert a_mle == pytest.approx(hill(x, x.size - 1), rel=0.02)


def test_alpha_hat_is_biased_upward_and_the_formula_says_by_how_much():
    """E[alpha_hat] = alpha n/(n-1): the Pareto MLE systematically reports a
    *thinner* tail than the truth, which is the comforting direction."""
    n, alpha = 60, 3.0
    ests = [pareto_mle_alpha(sample_pareto(alpha, n, np.random.default_rng(s)),
                             xm=1.0)[0] for s in range(4000)]
    th = alpha_hat_moments(alpha, n)
    assert np.mean(ests) == pytest.approx(th["mean"], rel=0.03)
    assert np.median(ests) == pytest.approx(th["median"], rel=0.03)
    assert np.std(ests) == pytest.approx(th["sd"], rel=0.06)
    assert th["mean"] > alpha                       # the bias, stated


def test_loglog_ols_is_worse_than_hill(rng):
    """A falsification test: the most popular tail estimator in applied work is
    beaten by the MLE on data generated from the model it assumes."""
    err_ols, err_hill = [], []
    for s in range(60):
        x = sample_pareto(3.0, 5_000, np.random.default_rng(1000 + s))
        err_ols.append(abs(loglog_ols_alpha(x, 0.05) - 3.0))
        err_hill.append(abs(hill(x, 250) - 3.0))
    assert np.median(err_ols) > np.median(err_hill)


def test_student_t_exceedances_converge_to_a_gpd_with_xi_one_over_nu(rng):
    """Pickands–Balkema–de Haan in action: we never tell the fitter it is looking
    at a Student-t, and it recovers xi = 1/nu from the threshold data alone."""
    nu = 4.0
    x = sample_student(nu, 400_000, rng)
    u = float(np.quantile(x, 0.99))
    xi, _ = fit_gpd(x[x > u] - u)
    assert xi == pytest.approx(1.0 / nu, abs=0.06)


def test_gpd_mle_and_pwm_agree_on_a_large_sample(rng):
    y = sample_gpd(0.3, 2.0, 200_000, rng)
    xa, ba = fit_gpd(y, "mle")
    xb, bb = fit_gpd(y, "pwm")
    assert xa == pytest.approx(0.3, abs=0.02)
    assert xa == pytest.approx(xb, abs=0.02)
    assert ba == pytest.approx(bb, rel=0.05)


def test_modified_scale_is_flat_across_thresholds(rng):
    """beta* = beta - xi u is threshold-invariant when the GPD holds, and
    flatness is the whole justification for reading a stability plot.

    Two exact targets. GPD(xi, beta) data: exceedances over u are GPD(xi,
    beta + xi u), so beta* = beta at every threshold. Pareto(alpha) data:
    exceedances over u are GPD(1/alpha, u/alpha), so beta = xi*u and beta* = 0.
    """
    y = sample_gpd(0.3, 2.0, 300_000, rng)
    u, xi, bstar, n = threshold_stability(y, np.quantile(y, [0.80, 0.90, 0.95, 0.98]))
    assert np.all(np.abs(xi - 0.3) < 0.05)
    assert np.all(np.abs(bstar - 2.0) < 0.3)

    x = sample_pareto(3.0, 300_000, rng)
    u2, xi2, bs2, _ = threshold_stability(x, np.quantile(x, [0.90, 0.95, 0.98]))
    assert np.all(np.abs(xi2 - 1 / 3) < 0.05)
    assert np.all(np.abs(bs2) < 0.1 * u2)


def test_pot_var_es_matches_the_exact_pareto_answer(rng):
    """VaR = p^(-1/alpha), ES = alpha/(alpha-1) VaR. POT never sees these."""
    alpha, p = 3.0, 0.01
    x = sample_pareto(alpha, 200_000, rng)
    out = pot_var_es(x, float(np.quantile(x, 0.95)), p)
    assert out["var"] == pytest.approx(p ** (-1.0 / alpha), rel=0.05)
    assert out["es"] == pytest.approx(es_var_ratio(alpha) * p ** (-1.0 / alpha), rel=0.08)


def test_empirical_es_understates_the_truth_more_often_than_not():
    """The opening gambit of the simulation notebook, asserted: the historical ES
    is below the true ES in a clear majority of samples, because the sample
    cannot contain the days that make the true number."""
    alpha, p, n = 3.0, 0.01, 2_000
    true_es = es_var_ratio(alpha) * p ** (-1.0 / alpha)
    below = [var_es_empirical(sample_pareto(alpha, n, np.random.default_rng(s)),
                              p)["es"] < true_es for s in range(600)]
    assert np.mean(below) > 0.60


def test_gaussian_es_badly_understates_a_fat_tail(rng):
    """A falsification test: the Gaussian ES is not merely imprecise on t(3)
    data, it is wrong by a factor, and always in the same direction."""
    x = sample_student(3.0, 400_000, rng)
    g = var_es_gaussian(x, 0.001)["es"]
    e = var_es_empirical(x, 0.001)["es"]
    assert e > 1.5 * g


def test_kappa_recovers_two_minus_alpha_for_a_stable(rng):
    k = kappa_mc(lambda s, r: sample_stable(1.5, s, r), n=30, n0=1,
                 reps=40_000, rng=rng)
    assert k == pytest.approx(0.5, abs=0.08)


def test_kappa_of_a_gaussian_is_zero(rng):
    k = kappa_mc(lambda s, r: r.standard_normal(s), n=30, n0=1, reps=40_000, rng=rng)
    assert abs(k) < 0.03


def test_kappa_bootstrap_is_biased_toward_thin_tails_in_small_samples():
    """The bootstrap cannot invent observations the sample never contained, so
    empirical kappa understates — and the bias shrinks monotonically with sample
    size. For a stable(1.5) (true kappa = 0.5) the medians run roughly
    0.33 -> 0.43 -> 0.47 as n goes 300 -> 3,000 -> 100,000: always from below.

    This is the caveat every empirical kappa in the market notebook carries."""
    med = []
    for n in (300, 3_000, 100_000):
        ks = [kappa_bootstrap(sample_stable(1.5, n, np.random.default_rng(s)),
                              n=30, n0=1, reps=8_000,
                              rng=np.random.default_rng(s + 900)) for s in range(15)]
        med.append(float(np.median(ks)))
    assert med[0] < med[1] < med[2]            # monotone in n
    assert med[2] < 0.5 + 0.02                 # still approaching from below
    assert med[2] == pytest.approx(0.5, abs=0.10)


def test_kurtosis_under_aggregation_at_a_constant_block_count(rng):
    """Scored the honest way: the same number of blocks at every lag, so the
    estimator's ceiling does not move between columns.

    The Gaussian sits at its true value of 3 throughout. The t(3) — whose
    population kurtosis is *infinite* at every lag, since a sum of finitely many
    infinite-fourth-moment variables still has none — nonetheless descends,
    because what is being measured is the sample kurtosis of a bounded number of
    blocks. Descent is therefore evidence about the body filling in, not about a
    moment converging.
    """
    B, lags = 2000, [1, 30]
    thin = kurtosis_under_aggregation(rng.standard_normal(B * 30), lags, blocks=B)
    fat = kurtosis_under_aggregation(sample_student(3.0, B * 30, rng), lags, blocks=B)
    assert abs(thin[0] - 3.0) < 0.3 and abs(thin[1] - 3.0) < 0.5   # flat at the truth
    assert fat[0] > thin[0] * 2                                     # fat at k = 1
    assert fat[1] < fat[0]                                          # and descending
    assert fat[1] > 3.0                                             # but not yet Gaussian


def test_local_alpha_creeps_up_to_the_true_exponent():
    """The preasymptotic point in one assertion: a Student-t(3) has a local
    exponent near 1 in its body and reaches 3 only far out."""
    from scipy.stats import t as st
    g = np.exp(np.linspace(np.log(0.5), np.log(300.0), 500))
    xm, a = local_alpha(g, st.sf(g, 3), bins=None)
    assert float(np.interp(1.0, xm, a)) < 1.5
    assert float(np.interp(100.0, xm, a)) == pytest.approx(3.0, abs=0.1)


def test_excess_conditional_ratio_is_flat_for_pareto_and_decays_for_gaussian(rng):
    x = sample_pareto(3.0, 500_000, rng)
    r_pareto = excess_conditional_ratio(x, [2.0, 4.0, 8.0])
    assert np.all(np.abs(r_pareto - 1.5) < 0.12)
    z = np.abs(rng.standard_normal(500_000))
    r_gauss = excess_conditional_ratio(z, [1.0, 2.0, 3.0])
    assert r_gauss[0] > r_gauss[-1] and r_gauss[-1] < 1.25


def test_gev_recovers_the_domain_of_attraction_from_block_maxima(rng):
    """Fisher–Tippett: Pareto maxima must land in Fréchet (xi = 1/alpha > 0),
    Gaussian maxima in Gumbel (xi ~ 0)."""
    xi_p, _, _ = fit_gev(block_maxima(sample_pareto(3.0, 400_000, rng), 200))
    xi_g, _, _ = fit_gev(block_maxima(rng.standard_normal(400_000), 200))
    assert xi_p == pytest.approx(1.0 / 3.0, abs=0.06)
    assert abs(xi_g) < 0.10


def test_the_gaussian_maximum_converges_to_gumbel_appallingly_slowly(rng):
    """EVT's own preasymptotics. After norming, the Gaussian maximum is still not
    Gumbel at n = 1000 — the fitted shape is visibly negative, and the residual
    is the O(1/log n) the theory warns about."""
    n = 1000
    a, b = gumbel_norming(n)
    m = rng.standard_normal((6000, n)).max(axis=1)
    xi, _, _ = fit_gev((m - b) / a)
    assert xi < -0.02                          # not yet Gumbel
    assert xi > -0.25                          # but on its way


def test_mean_excess_is_linear_with_the_predicted_slope(rng):
    """e(u) = (beta + xi u)/(1 - xi): slope xi/(1-xi) = 0.5 at xi = 1/3."""
    x = sample_pareto(3.0, 400_000, rng)
    u, e, se, n = mean_excess(x, np.quantile(x, [0.80, 0.90, 0.95, 0.98]))
    slope = np.polyfit(u, e, 1)[0]
    assert slope == pytest.approx(0.5, rel=0.12)


def test_profile_interval_is_asymmetric_and_covers_the_truth(rng):
    """Not a Wald interval: the GPD likelihood in xi is skewed right, so the
    profile interval must be longer above xi_hat than below it."""
    y = sample_gpd(0.35, 1.0, 1_500, rng)
    xi, _ = fit_gpd(y)
    lo, hi = gpd_profile_ci(y, 0.95)
    assert lo < 0.35 < hi
    assert (hi - xi) > (xi - lo)


def test_extremal_index_is_one_for_iid_and_well_below_one_for_garch(rng):
    z = rng.standard_normal(100_000)
    assert extremal_index(np.abs(z), 2.5) == pytest.approx(1.0, abs=0.10)
    g = np.abs(garch_returns(100_000, rng, omega=1e-6, a=0.09, b=0.90))
    assert extremal_index(g, float(np.quantile(g, 0.98))) < 0.5


def test_garch_manufactures_a_power_law_from_gaussian_shocks(rng):
    """Kesten: clustered volatility is a *source* of tail index, not merely of
    dependence. The Hill estimate on simulated GARCH must land near the
    theoretical Kesten alpha, with thin-tailed innovations throughout."""
    a, b = 0.10, 0.88
    theo = garch_kesten_alpha(a, b, rng=np.random.default_rng(2))
    assert 2.0 < theo < 8.0
    x = np.abs(garch_returns(400_000, rng, omega=1e-6, a=a, b=b))
    assert hill(x, 4_000) == pytest.approx(theo, rel=0.35)


def test_t_copula_has_tail_dependence_at_zero_correlation(rng):
    """Zero correlation, strong joint extremes — the single fact behind
    'portfolios should never rely on correlation'."""
    q = 0.02
    u = t_copula_uniforms(400_000, 0.0, 3.0, rng)
    joint = np.mean((u[:, 0] < q) & (u[:, 1] < q)) / q
    assert joint > 4.0 * q                      # Gaussian copula would give ~q


def test_stationary_bootstrap_preserves_the_marginal(rng):
    x = sample_student(4.0, 20_000, rng)
    b = stationary_bootstrap(x, 20.0, 40, rng)
    assert b.shape == (40, 20_000)
    assert b.std() == pytest.approx(x.std(), rel=0.15)
    assert np.median(np.abs(b)) == pytest.approx(np.median(np.abs(x)), rel=0.05)


def test_shadow_mean_is_consistent_on_a_capped_pareto(rng):
    """Construct a variable that *is* Paretian in the dual, cap it, and check the
    shadow machinery recovers the mean the cap hides."""
    raw = sample_pareto(1.7, 40_000, rng)
    bounded = dual_inverse(raw, 80.0, 1.0)
    out = shadow_mean(bounded, 80.0, 1.0)
    assert out["alpha_dual"] == pytest.approx(1.7, rel=0.12)
    assert out["ratio"] == pytest.approx(1.0, abs=0.06)


def test_quantile_contribution_is_biased_downward_in_small_samples():
    """Taleb Ch. 16: the measured top-1% share rises with sample size, because
    small samples cannot contain the observations that would dominate it."""
    small = np.median([quantile_contribution(
        sample_pareto(1.5, 1_000, np.random.default_rng(s)), 0.01) for s in range(120)])
    large = np.median([quantile_contribution(
        sample_pareto(1.5, 100_000, np.random.default_rng(s)), 0.01) for s in range(12)])
    assert small < large


def test_kelly_is_capped_by_the_worst_observation_under_fat_tails(rng):
    """The Gaussian formula does not know the left tail exists; the empirical
    optimiser is bounded by it."""
    r = sample_student(2.5, 20_000, rng) * 0.01
    out = kelly_empirical(r, fmax=200.0)
    assert out["f_cap"] < 200.0
    assert out["f_star"] <= out["f_cap"] + 1e-9
    assert growth_rate(r, out["f_star"]) >= growth_rate(r, out["f_star"] * 2.0) - 1e-12


def test_ruin_probability_rises_with_leverage_and_clustering(rng):
    r = sample_student(3.0, 8_000, rng) * 0.01 + 3e-4
    lo = ruin_probability(r, 1.0, 750, reps=3000, barrier=0.5,
                          rng=np.random.default_rng(11))["p_ruin"]
    hi = ruin_probability(r, 4.0, 750, reps=3000, barrier=0.5,
                          rng=np.random.default_rng(11))["p_ruin"]
    assert hi > lo
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_barbell_truncates_the_tail_more_than_it_cuts_the_mean(rng):
    """20% risky keeps 20% of the edge and cuts ES by 80% """
    r = sample_student(3.0, 50_000, rng) * 0.01 + 5e-4
    b = barbell(r, 0.2)
    assert b.mean() == pytest.approx(0.2 * r.mean(), rel=1e-9)
    assert var_es_empirical(-b, 0.01)["es"] == pytest.approx(
        0.2 * var_es_empirical(-r, 0.01)["es"], rel=1e-9)


# --------------------------------------------------------------------------- #
#  3. Falsification — the estimators that must be caught failing
# --------------------------------------------------------------------------- #
def test_kupiec_passes_a_model_that_acerbi_rejects():
    """The headline of the backtesting block, run at the one tail probability
    where the demonstration is clean.

    For standardised t(3) the Gaussian VaR *understates* at p = 1% (2.33 vs 2.62)
    and *overstates* at p = 5% (1.64 vs 1.36); the two cross near **p = 2%**
    (2.054 vs 2.010), where the Gaussian gets the number of exceptions
    essentially right. Their size it does not: the true ES there is 3.157 against
    the Gaussian's 2.421, so the model is 23% short. Kupiec passes and Acerbi
    rejects — counting exceptions is not measuring them, which is exactly why the
    RMT lab's variance ranking told us nothing about solvency.

    The seeded counts are 11/12 and 11/12; the assertions leave headroom so the
    test is about the phenomenon and not about a scipy version.
    """
    n, p = 6_000, 0.02
    passes_count, rejects_size = 0, 0
    for s in range(12):
        r = np.random.default_rng(500 + s)
        z = r.standard_t(3, n) / np.sqrt(3.0)
        g = var_es_gaussian(z, p)
        hits = var_hits(z, g["var"])
        sim = r.standard_normal((1500, n)) * z.std() + z.mean()
        a = acerbi_szekely_z2(z, g["var"], g["es"], p, sim)
        passes_count += kupiec_pof(hits, p)["pvalue"] > 0.05
        rejects_size += (a["z2"] > 0.0) and (a["pvalue"] < 0.05)
    assert passes_count >= 9                               # the count is fine
    assert rejects_size >= 8                               # the size is not


def test_clustered_losses_pass_the_count_and_fail_independence(rng):
    g = garch_returns(30_000, rng, omega=1e-6, a=0.09, b=0.90, nu=6.0)
    losses = -g
    v = float(np.quantile(losses, 0.99))
    hits = var_hits(losses, v)
    assert kupiec_pof(hits, 0.01)["pvalue"] > 0.05
    assert christoffersen_independence(hits)["pvalue"] < 0.01


def test_lognormal_fools_the_hill_plot(rng):
    """A Hill horror plot: the lognormal has every moment, yet its Hill estimate
    drifts smoothly with k and never settles — so a 'plateau' is necessary
    evidence of a power law, never sufficient.
    """
    x = sample_lognormal(2.0, 200_000, rng)
    k, a, se = hill_plot(x, 200, 20_000)
    assert a[0] > a[-1]                                    # monotone drift, no plateau
    assert a[0] / a[-1] > 1.5
    pareto = sample_pareto(2.0, 200_000, rng)               # the control: flat
    _, ap, _ = hill_plot(pareto, 200, 20_000)
    assert abs(ap[0] / ap[-1] - 1.0) < 0.20


def test_es_sigma_ratio_separates_shapes_at_equal_variance(rng):
    """Same sigma, different ES: the reason the RMT lab's variance ranking
    scrambled."""
    z = rng.standard_normal(300_000)
    t3 = sample_student(3.0, 300_000, rng, standardize=True)
    assert es_sigma_ratio(z, 0.01) == pytest.approx(2.665, abs=0.10)
    assert es_sigma_ratio(t3, 0.01) > 3.4


def test_ewma_and_student_fits_return_sane_numbers(rng):
    z = np.abs(rng.standard_normal(20_000))
    for out in (var_es_ewma(z, 0.01), var_es_student(z, 0.01),
                var_es_gpd(z, 0.01, 0.95)):
        assert np.isfinite(out["var"]) and np.isfinite(out["es"])
        assert out["es"] > out["var"] > 0


def test_pickands_and_dedh_agree_with_hill_on_a_clean_pareto(rng):
    """Three independent estimators of the same quantity. When they disagree on
    real data, the disagreement is the finding — so they had better agree here."""
    x = sample_pareto(2.5, 200_000, rng)
    xi_true = 1.0 / 2.5
    assert pickands(x, 8_000) == pytest.approx(xi_true, abs=0.08)
    assert dedh_moment(x, 20_000) == pytest.approx(xi_true, abs=0.05)
    assert 1.0 / hill(x, 20_000) == pytest.approx(xi_true, abs=0.03)


# --------------------------------------------------------------------------- #
#  4. The covariance cleaners folded in from the RMT lab
# --------------------------------------------------------------------------- #
def test_cleaners_preserve_the_trace(rng):
    """Every eigenvalue map here redistributes variance, it does not create any."""
    x = rng.standard_normal((60, 200))
    e = sample_cov(x)
    for method in ("sample", "clip", "rie"):
        assert np.trace(clean_cov(x, method)) == pytest.approx(np.trace(e), rel=1e-8)


def test_mp_edges_and_clipping_are_consistent(rng):
    lo, hi = mp_edges(0.25, sigma2=3.0)
    assert (lo, hi) == pytest.approx((3.0 * 0.25, 3.0 * 2.25))
    lam = np.array([9.0, 5.0, 1.2, 0.9, 0.4])          # two above the q=0.25 edge
    xi = xi_clip(lam, 0.25)
    assert xi[0] == 9.0 and xi[1] == 5.0                # kept
    assert xi[2] == xi[3] == xi[4]                      # bulk flattened to one value
    assert xi.sum() == pytest.approx(lam.sum())


def test_eigh_desc_and_recompose_round_trip(rng):
    a = rng.standard_normal((25, 25)); a = a + a.T
    lam, u = eigh_desc(a)
    assert np.all(np.diff(lam) <= 1e-12)                # descending
    assert np.allclose(recompose(u, lam), a, atol=1e-10)


def test_cleaning_beats_the_raw_sample_out_of_sample(rng):
    """The result the previous lab was built on, kept alive here because the
    market notebook rebuilds those portfolios: at q = 1/2 every cleaner delivers
    less true risk than the raw sample covariance."""
    n, t = 100, 200
    c_true = np.eye(n)
    realized = {}
    for method in ("sample", "clip", "lw", "rie"):
        v = []
        for s in range(12):
            x = np.random.default_rng(500 + s).standard_normal((n, t))
            w = min_var_weights(clean_cov(x, method))
            v.append(w @ c_true @ w)
        realized[method] = float(np.mean(v))
    assert realized["sample"] > realized["rie"]
    assert realized["sample"] > realized["lw"]
    assert realized["sample"] > realized["clip"]


def test_min_var_weights_sum_to_one_and_lw_intensity_is_a_fraction(rng):
    x = rng.standard_normal((40, 120))
    assert min_var_weights(sample_cov(x)).sum() == pytest.approx(1.0)
    _, rho, mu = lw_shrinkage(x)
    assert 0.0 <= rho <= 1.0 and mu > 0


def test_sample_corr_has_unit_diagonal_and_day_scale_recovers_sigma(rng):
    x = rng.standard_normal((30, 400))
    assert np.allclose(np.diag(sample_corr(x)), 1.0)
    s = day_scale(3.0 * rng.standard_normal((400, 200)))    # 400 assets, 200 days
    assert s.shape == (1, 200)
    assert float(np.mean(s)) == pytest.approx(3.0, rel=0.02)


def test_module_exports_are_all_importable():
    import evt_lab
    missing = [n for n in evt_lab.__all__ if not hasattr(evt_lab, n)]
    assert missing == []
