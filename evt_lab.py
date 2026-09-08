"""

A laboratory for extreme values, preasymptotics, and ruin probabilities.

The previous lab (Quant-RMT-FatTails) ended on an uncomfortable note. Every score
in it was a second moment — minimum variance minimises variance, realized
volatility measures variance, the honesty ratio is a ratio of standard deviations
— and yet the market it was scoring has an ``R_4`` that never converges. When the
same P&L streams were re-scored on the tail, the ranking scrambled: RIE won on
volatility, Ledoit–Wolf on expected shortfall, clipping on drawdown. That lab
could say which estimator minimises variance. It could not say which one keeps
you solvent.

This module is the instrument set for the second question. Three things it does:

1. **Preasymptotics.** Before any limit theorem is invoked, we ask whether the
sample is anywhere near the limit. The kappa metric (how much slower than
   ``sqrt(n)`` your errors actually shrink), max-to-sum ratios (does the p-th
   moment exist *in this sample*), kurtosis under aggregation, MAD/STD, and the
   local tail exponent that shows where a power law starts pretending to be one.
2. **Extreme value theory.** Both classical routes — block maxima with the GEV
   (Fisher–Tippett–Gnedenko) and peaks over threshold with the GPD
   (Pickands–Balkema–de Haan) — plus the tail-index estimators (Hill, Pickands,
   and others), their *sampling distributions*, and the extremal-index machinery for 
   the fact that market extremes arrive in clusters.
3. **Survival.** Not going bust is the point here. Value-at-risk and expected shortfall under six 
   different assumptions, their backtests (Kupiec, Christoffersen, Acerbi–Székely). Then, we 
   build the sizing layer: growth- optimal fractions under fat tails, barbells, ES targeting,
   absorbing barriers, and the gap between the time average and the ensemble average.

Conventions. Samples are 1-D arrays unless stated. **Losses are positive**: a
return of -3% is a loss of 0.03, and every VaR/ES in this module is a positive
number quoted on the loss scale. Tail work is done on the *upper* tail of a
positive variable, so the left tail of returns is studied by passing ``-r``.
``p`` is always the tail probability (0.01), never the confidence level (0.99).
Tail exponents: ``alpha = 1 / xi``, so a cubic tail is ``alpha = 3``, ``xi = 1/3``.

numpy plus scipy (optimisation, special functions, and the GEV/GPD densities);
``rmt_lab.py`` is vendored alongside for the notebooks, not imported here.

Author: Marco Galoppo
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, special
from scipy.stats import chi2, genextreme, genpareto, norm, t as student_t

__all__ = [
    # generators
    "sample_pareto",
    "sample_student",
    "sample_lognormal",
    "sample_stable",
    "sample_gpd",
    "sample_gev",
    "variance_preserving_mixture",
    "stochvol_returns",
    "garch_returns",
    "garch_kesten_alpha",
    "t_copula_uniforms",
    # preasymptotics
    "mad",
    "mad_over_std",
    "max_to_sum_ratio",
    "kappa_mc",
    "kappa_bootstrap",
    "kurtosis_under_aggregation",
    "survival_empirical",
    "local_alpha",
    "excess_conditional_ratio",
    # block maxima
    "block_maxima",
    "fit_gev",
    "gev_return_level",
    "gumbel_norming",
    # peaks over threshold
    "mean_excess",
    "exceedances",
    "fit_gpd",
    "fit_gpd_pwm",
    "gpd_profile_ci",
    "threshold_stability",
    "pot_var_es",
    # tail index
    "hill",
    "hill_plot",
    "pickands",
    "dedh_moment",
    "loglog_ols_alpha",
    "pareto_mle_alpha",
    "alpha_hat_moments",
    # shadow moments / hidden tail
    "pareto_mean_above",
    "expected_max_pareto",
    "hidden_mean_fraction",
    "dual_transform",
    "dual_inverse",
    "shadow_mean",
    "quantile_contribution",
    # risk functionals
    "var_es_empirical",
    "var_es_gaussian",
    "var_es_student",
    "var_es_ewma",
    "var_es_gpd",
    "es_var_ratio",
    "alpha_from_es_var_ratio",
    "es_sigma_ratio",
    # clustering
    "extremal_index",
    "decluster",
    "stationary_bootstrap",
    # survival & sizing
    "drawdown",
    "max_drawdown",
    "growth_rate",
    "kelly_fraction",
    "kelly_empirical",
    "barbell",
    "leverage_for_es",
    "simulate_wealth",
    "ruin_probability",
    # backtests
    "var_hits",
    "kupiec_pof",
    "christoffersen_independence",
    "christoffersen_cc",
    "acerbi_szekely_z2",
    # covariance cleaning, carried over from the RMT lab
    "sample_cov",
    "sample_corr",
    "eigh_desc",
    "recompose",
    "mp_edges",
    "xi_clip",
    "lw_shrinkage",
    "xi_rie",
    "clean_cov",
    "min_var_weights",
    "day_scale",
]


# --------------------------------------------------------------------------- #
#  1. Generators — the distributions whose truth we know
# --------------------------------------------------------------------------- #
def sample_pareto(alpha: float, size, rng: np.random.Generator, xm: float = 1.0):
    """Pareto Type I: ``P(X > x) = (x/xm)^(-alpha)`` for ``x >= xm``.
    The mean exists iff alpha > 1, variance iff alpha > 2, kurtosis iff alpha > 4.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    u = rng.random(size)
    return xm * u ** (-1.0 / alpha)


def sample_student(nu: float, size, rng: np.random.Generator,
                   standardize: bool = False):
    """Student-t with ``nu`` degrees of freedom; tail exponent alpha = nu.
    One parameter moves continuously from the Gaussian (nu -> inf) through the Cauchy (nu = 1).
    standardize=True rescales to unit variance, which requires nu > 2 
    """
    x = rng.standard_t(nu, size)
    if standardize:
        if nu <= 2:
            raise ValueError("nu <= 2 has no variance to standardise to")
        x = x * np.sqrt((nu - 2.0) / nu)
    return x


def sample_lognormal(sigma: float, size, rng: np.random.Generator, mu: float = 0.0):
    """Lognormal. Not fat-tailed in the power-law sense (all moments exist), yet
    for large ``sigma`` it is preasymptotically indistinguishable from one.
    """
    return np.exp(mu + sigma * rng.standard_normal(size))


def sample_stable(alpha: float, size, rng: np.random.Generator):
    """Symmetric alpha-stable, unit scale (Chambers–Mallows–Stuck).
    Tail ``P(|X| > x) ~ C x^(-alpha)``; alpha = 2 is Gaussian, alpha < 2 has
    infinite variance, alpha <= 1 infinite mean. Carried over unchanged from
    ``rmt_lab.py`` so this module stands alone.
    """
    if not 0.0 < alpha <= 2.0:
        raise ValueError("alpha must be in (0, 2]")
    u = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size)
    w = rng.exponential(1.0, size)
    if abs(alpha - 1.0) < 1e-12:
        return np.tan(u)
    return (np.sin(alpha * u) / np.cos(u) ** (1.0 / alpha)
            * (np.cos((1.0 - alpha) * u) / w) ** ((1.0 - alpha) / alpha))


def sample_gpd(xi: float, beta: float, size, rng: np.random.Generator):
    """Generalised Pareto exceedances: ``P(Y > y) = (1 + xi*y/beta)^(-1/xi)``.
    xi > 0 is the heavy (Fréchet) case with alpha = 1/xi; xi = 0 the exponential
    limit; xi < 0 has a finite upper endpoint at -beta/xi. This is the limit law
    of threshold exceedances for essentially everything (Pickands–Balkema–de
    Haan).
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    u = rng.random(size)
    if abs(xi) < 1e-12:
        return -beta * np.log(u)
    return beta * (u ** (-xi) - 1.0) / xi


def sample_gev(xi: float, size, rng: np.random.Generator,
               mu: float = 0.0, sigma: float = 1.0):
    """Generalised extreme value draws: the limit law of block maxima.
    xi > 0 Fréchet (heavy), xi = 0 Gumbel (thin), xi < 0 Weibull (bounded).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    u = rng.random(size)
    y = -np.log(u)
    if abs(xi) < 1e-12:
        return mu - sigma * np.log(y)
    return mu + sigma * (y ** (-xi) - 1.0) / xi


def variance_preserving_mixture(size, rng: np.random.Generator,
                                sigma: float = 1.0, a: float = 0.5,
                                p: float = 0.5):
    """Taleb's variance-preserving heuristic (SCoFT 4.1.1): a Gaussian whose
    variance is itself a coin flip.

    With probability ``p`` the variance is ``sigma^2 (1 + a)``; otherwise
    ``sigma^2 (1 - a*p/(1-p))``. The unconditional variance is exactly
    ``sigma^2`` for any ``a``, but the kurtosis rises with ``a`` — at p = 1/2 it
    is exactly ``3(1 + a^2)``. 
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    v_hi = sigma ** 2 * (1.0 + a)
    v_lo = sigma ** 2 * (1.0 - a * p / (1.0 - p))
    if v_lo < 0:
        raise ValueError("a too large for this p: the low-variance state is negative")
    hi = rng.random(size) < p
    return rng.standard_normal(size) * np.sqrt(np.where(hi, v_hi, v_lo))


def stochvol_returns(t: int, rng: np.random.Generator, sigma: float = 1.0,
                     s: float = 0.5):
    """Lognormal stochastic volatility: ``x_t = sigma_t z_t``,
    ``sigma_t = sigma * exp(s w_t - s^2/2)`` with ``w_t`` iid standard normal.

    Fat-tailed, and *not* a power law: every moment exists. 
    """
    vol = sigma * np.exp(s * rng.standard_normal(t) - 0.5 * s ** 2)
    return vol * rng.standard_normal(t)


def garch_returns(t: int, rng: np.random.Generator, omega: float = 1e-6,
                  a: float = 0.09, b: float = 0.90, nu: Optional[float] = None,
                  burn: int = 1000):
    """GARCH(1,1) with Gaussian or standardised-t innovations.

    ``sigma_t^2 = omega + a * eps_{t-1}^2 + b * sigma_{t-1}^2``. Covariance
    stationary iff ``a + b < 1``. The point of having it here is *not* to fit
    GARCH to anything — it is that GARCH manufactures a genuine power-law tail
    out of thin-tailed innovations (Kesten), so clustered volatility is itself a
    source of tail index. See ``garch_kesten_alpha``.
    """
    if a + b >= 1.0:
        raise ValueError("a + b >= 1: not covariance stationary")
    n = t + burn
    x = np.empty(n)
    z = (rng.standard_normal(n) if nu is None
         else rng.standard_t(nu, n) * np.sqrt((nu - 2.0) / nu))
    s2 = omega / (1.0 - a - b)
    for i in range(n):
        x[i] = np.sqrt(s2) * z[i]
        s2 = omega + a * x[i] ** 2 + b * s2
    return x[burn:]


def garch_kesten_alpha(a: float, b: float, nu: Optional[float] = None,
                       reps: int = 200_000, rng: Optional[np.random.Generator] = None,
                       bracket: Tuple[float, float] = (0.2, 40.0)) -> float:
    """Tail index of a GARCH(1,1), from the Kesten condition
    ``E[(a z^2 + b)^(k/2)] = 1``, solved by Monte Carlo for ``k = alpha``.

    This is the fact that makes volatility clustering a *tail* phenomenon and not
    only a dependence phenomenon: feed a GARCH thin-tailed Gaussian shocks and it
    returns a power law. For the usual equity calibration (a ~ 0.09, b ~ 0.90)
    the answer lands close to the cubic law the data show.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    z = (rng.standard_normal(reps) if nu is None
         else rng.standard_t(nu, reps) * np.sqrt((nu - 2.0) / nu))
    w = a * z ** 2 + b

    def f(k):
        return np.mean(w ** (k / 2.0)) - 1.0

    lo, hi = bracket
    if f(lo) * f(hi) > 0:
        return float("nan")
    return float(optimize.brentq(f, lo, hi, xtol=1e-6))


def t_copula_uniforms(n: int, rho: float, nu: float, rng: np.random.Generator):
    """A pair of uniforms with a t-copula: ``(n, 2)`` array in ``[0, 1]^2``.

    Zero correlation is not zero tail dependence. For the t-copula the lower and
    upper tail-dependence coefficients are both
    ``2 * T_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))``, which is strictly positive
    even at ``rho = 0``.
    """
    g = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    w = rng.chisquare(nu, n)
    x = g * np.sqrt(nu / w)[:, None]
    return student_t.cdf(x, nu)


# --------------------------------------------------------------------------- #
#  2. Preasymptotics — is this sample anywhere near any limit?
# --------------------------------------------------------------------------- #
def mad(x) -> float:
    """Mean absolute deviation from the mean, an L1 scale.

    A first-moment object: the influence of an extreme observation is linear
    rather than quadratic. It exists whenever the mean does, which is more than
    can be said for the standard deviation.
    """
    x = np.asarray(x, float).ravel()
    return float(np.mean(np.abs(x - x.mean())))


def mad_over_std(x) -> float:
    """MAD / STD. Exactly ``sqrt(2/pi) ~ 0.7979`` for a Gaussian; falls toward
    zero as tails fatten, because STD is inflated by the extremes and MAD is not.

    The cheapest fat-tail diagnostic in existence. It is bounded and needs one
    moment, so unlike kurtosis it always returns a number that means something.
    """
    x = np.asarray(x, float).ravel()
    s = x.std()
    return float(mad(x) / s) if s > 0 else float("nan")


def max_to_sum_ratio(a, p: float = 4.0) -> np.ndarray:
    """Running ``R_n(p) = max_i |a_i|^p / sum_i |a_i|^p`` over the first n points.

    By the strong law, ``R_n(p) -> 0`` if and only if ``E|X|^p < inf``.

    **Read that as a usability test, not an existence test.** The "only if" is an
    asymptotic statement and says nothing about the rate, which can be
    arbitrarily slow.

    So a stubbornly high R tells you the p-th moment is a fiction *of your
    sample*, and every statistic resting on it — kurtosis, GARCH fits,
    Tracy–Widom edge arguments, Sharpe ratios — is measuring one observation.
    It does not tell you whether the moment exists in the population, and in a
    finite sample the two cases are essentially indistinguishable.

    Carried over from ``rmt_lab.py``.
    """
    z = np.abs(np.ravel(np.asarray(a, float))) ** p
    return np.maximum.accumulate(z) / np.cumsum(z)


def kappa_mc(sample_fn: Callable, n: int, n0: int = 1, reps: int = 20_000,
             rng: Optional[np.random.Generator] = None) -> float:
    """Preasymptotic ``kappa(n0, n) = 2 - ln(n/n0) / ln(M(n)/M(n0))``, where
    ``M(m)`` is the mean absolute deviation of an m-sum of iid draws from
    ``sample_fn(size, rng)``.

    kappa = 0 is Gaussian-speed aggregation (``M ~ sqrt(n)``); an alpha-stable
    gives ``kappa = 2 - alpha`` exactly. Read it as: how much slower than
    ``sqrt(n)`` your errors actually shrink at *this* sample size, and therefore
    how much more data the CLT is quietly charging you.

    Carried over from ``rmt_lab.py``.
    """
    rng = np.random.default_rng() if rng is None else rng

    def mad_of_sum(m: int) -> float:
        s = sample_fn((reps, m), rng).sum(axis=1)
        return float(np.mean(np.abs(s - s.mean())))

    return 2.0 - np.log(n / n0) / np.log(mad_of_sum(n) / mad_of_sum(n0))


def kappa_bootstrap(x, n: int, n0: int = 1, reps: int = 20_000,
                    rng: Optional[np.random.Generator] = None) -> float:
    """Empirical kappa: the same statistic with the sample resampled in place of
    a known generator.

    Sums of ``n`` draws taken with replacement from ``x``. Bootstrapping from a fat-tailed sample 
    cannot invent the observations the sample never contained, so this is biased *toward
    thin tails* — it is a lower bound on how bad things are, not an estimate.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, float).ravel()

    def mad_of_sum(m: int) -> float:
        s = rng.choice(x, size=(reps, m), replace=True).sum(axis=1)
        return float(np.mean(np.abs(s - s.mean())))

    return 2.0 - np.log(n / n0) / np.log(mad_of_sum(n) / mad_of_sum(n0))


def kurtosis_under_aggregation(x, lags: Sequence[int],
                               blocks: Optional[int] = None) -> np.ndarray:
    """Sample kurtosis of non-overlapping ``k``-period sums, for each k in lags.

    Under a finite fourth moment the CLT drives excess kurtosis down like ``1/k``:
    aggregate enough and everything becomes Gaussian. The *shape* of this curve is the test.
    """
    x = np.asarray(x, float).ravel()
    out = []
    for k in lags:
        m = len(x) // k if blocks is None else int(blocks)
        if m < 4 or m * k > len(x):
            out.append(np.nan)
            continue
        s = x[: m * k].reshape(m, k).sum(axis=1)
        sd = s.std()
        out.append(np.mean((s - s.mean()) ** 4) / sd ** 4 if sd > 0 else np.nan)
    return np.asarray(out, float)


def survival_empirical(x) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical survival function as ``(x_sorted_ascending, S)`` with
    ``S = 1 - i/n`` — i.e. ``S(x_(i)) = (n - i)/n``.

    Plotted on log-log axes this is the Zipf plot: a power-law tail is a straight
    line of slope ``-alpha``. 
    """
    x = np.sort(np.asarray(x, float).ravel())
    n = x.size
    s = 1.0 - np.arange(1, n + 1) / n
    return x[:-1], s[:-1]


def local_alpha(x_grid, survival, bins: Optional[int] = 40
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Local (running) tail exponent ``-d log S / d log x`` by finite differences.

    Feed it either an exact survival function or the output of
    ``survival_empirical``. This is what makes the *preasymptotics* visible: a
    Student-t(3) does not have a local exponent of 3 anywhere in its body — the
    exponent creeps up from roughly 1 and only reaches 3 far out in the tail, past
    the Karamata point where the slowly varying part has finished varying. So
    "the exponent is 3" is a statement about a region, and any estimate that
    silently averages over the body is measuring the wrong thing.

    ``bins`` differences over a log-spaced grid instead of over consecutive
    points: the raw pointwise version is a ratio of two noisy differences and is badly
    biased upward in the far tail, where the gaps between order statistics are themselves exponential. 
    Pass ``bins=None`` only when ``survival`` is an exact function.
    Returns ``(x_mid, alpha_local)``.
    """
    x = np.asarray(x_grid, float).ravel()
    s = np.asarray(survival, float).ravel()
    keep = (x > 0) & (s > 0)
    lx, ls = np.log(x[keep]), np.log(s[keep])
    order = np.argsort(lx)
    lx, ls = lx[order], ls[order]
    if bins is not None and lx.size > bins:
        grid = np.linspace(lx[0], lx[-1], bins + 1)
        ls = np.interp(grid, lx, ls)
        lx = grid
    with np.errstate(divide="ignore", invalid="ignore"):
        a = -np.diff(ls) / np.diff(lx)
    return np.exp(0.5 * (lx[1:] + lx[:-1])), a


def excess_conditional_ratio(x, thresholds) -> np.ndarray:
    """``E[X | X > K] / K`` at each threshold K.

    For a thin-tailed variable this ratio decays to 1: conditioning on an
    exceedance tells you the value is *just* above K. For a Pareto tail it is
    constant at ``alpha/(alpha-1)``, forever — "how bad, given that it is bad" has
    no scale.
    """
    x = np.asarray(x, float).ravel()
    out = []
    for k in np.atleast_1d(thresholds):
        tail = x[x > k]
        out.append(tail.mean() / k if tail.size >= 5 and k > 0 else np.nan)
    return np.asarray(out, float)


# --------------------------------------------------------------------------- #
#  3. Extreme value theory I — block maxima and the GEV
# --------------------------------------------------------------------------- #
def block_maxima(x, block: int, drop_partial: bool = True) -> np.ndarray:
    """Maxima of consecutive non-overlapping blocks of length ``block``."""
    x = np.asarray(x, float).ravel()
    m = len(x) // block
    out = x[: m * block].reshape(m, block).max(axis=1)
    if not drop_partial and len(x) % block:
        out = np.append(out, x[m * block:].max())
    return out


def fit_gev(m, method: str = "mle") -> Tuple[float, float, float]:
    """Fit the generalised extreme value law to block maxima. Returns
    ``(xi, mu, sigma)`` in the *statistical* convention ``xi > 0 = heavy``.

    Fisher–Tippett–Gnedenko: if block maxima converge to anything at all after
    affine rescaling, they converge to this three-parameter family, and the sign
    of xi selects the domain of attraction. method ∈ {"mle", "pwm"}; PWM
    (Hosking) is more stable in small samples and is used as the MLE's start.

    Note scipy's sign convention differs: ``genextreme(c=-xi)``.
    """
    m = np.asarray(m, float).ravel()
    if m.size < 8:
        raise ValueError("need at least 8 block maxima")
    xi0, mu0, sig0 = _gev_pwm(m)
    if method == "pwm":
        return xi0, mu0, sig0
    if method != "mle":
        raise ValueError(f"unknown method {method!r}")
    try:
        c, loc, scale = genextreme.fit(m, -xi0, loc=mu0, scale=sig0)
    except Exception:                                   # pragma: no cover
        return xi0, mu0, sig0
    if not np.isfinite([c, loc, scale]).all() or scale <= 0:
        return xi0, mu0, sig0
    return float(-c), float(loc), float(scale)


def _gev_pwm(m: np.ndarray) -> Tuple[float, float, float]:
    """Hosking's probability-weighted-moment estimator for the GEV."""
    y = np.sort(m)
    n = y.size
    f = (np.arange(1, n + 1) - 0.35) / n
    b0 = y.mean()
    b1 = np.mean(f * y)
    b2 = np.mean(f ** 2 * y)
    denom = 3.0 * b2 - b0
    if abs(denom) < 1e-14:
        return 0.0, float(b0), float(np.std(y) + 1e-12)
    c = (2.0 * b1 - b0) / denom - np.log(2.0) / np.log(3.0)
    k = 7.8590 * c + 2.9554 * c ** 2                     # Hosking's k = -xi
    if abs(k) < 1e-6:
        sigma = (2.0 * b1 - b0) / np.log(2.0)
        mu = b0 - sigma * np.euler_gamma
        return 0.0, float(mu), float(max(sigma, 1e-12))
    g = special.gamma(1.0 + k)
    sigma = (2.0 * b1 - b0) * k / (g * (1.0 - 2.0 ** (-k)))
    mu = b0 + sigma * (g - 1.0) / k
    return float(-k), float(mu), float(max(sigma, 1e-12))


def gev_return_level(xi: float, mu: float, sigma: float, period: float) -> float:
    """The level exceeded by one block maximum every ``period`` blocks:
    ``z = mu - (sigma/xi) [1 - (-ln(1 - 1/T))^(-xi)]``.
    """
    if period <= 1:
        raise ValueError("period must exceed 1")
    y = -np.log(1.0 - 1.0 / period)
    if abs(xi) < 1e-12:
        return float(mu - sigma * np.log(y))
    return float(mu - (sigma / xi) * (1.0 - y ** (-xi)))


def gumbel_norming(n: int) -> Tuple[float, float]:
    """Classical norming constants for the maximum of n iid standard normals:
    ``b_n = sqrt(2 ln n) - (ln ln n + ln 4pi) / (2 sqrt(2 ln n))``, ``a_n = 1/b_n``.

    The Gaussian *is* in the Gumbel domain — and gets there at rate ``1/ln n``,
    which is the slowest useful rate in statistics.
    """
    if n < 3:
        raise ValueError("n must be at least 3")
    ln = np.log(n)
    b = np.sqrt(2.0 * ln) - (np.log(ln) + np.log(4.0 * np.pi)) / (2.0 * np.sqrt(2.0 * ln))
    return float(1.0 / b), float(b)


# --------------------------------------------------------------------------- #
#  4. Extreme value theory II — peaks over threshold and the GPD
# --------------------------------------------------------------------------- #
def exceedances(x, u: float) -> np.ndarray:
    """Threshold exceedances ``x - u`` for ``x > u``, strictly positive."""
    x = np.asarray(x, float).ravel()
    return x[x > u] - u


def mean_excess(x, thresholds=None, min_count: int = 10):
    """Mean excess function ``e(u) = E[X - u | X > u]`` on a threshold grid.

    Returns ``(u, e, se, count)``. For a GPD tail ``e(u) = (beta + xi*u)/(1 - xi)``
    is **linear in u** with slope ``xi/(1-xi)``. Upward slope means a heavy tail,
    flat means exponential, downwardnmeans a finite endpoint. Infinite mean (xi >= 1) means 
    the function does notexist and the plot is drawing noise.
    """
    x = np.asarray(x, float).ravel()
    if thresholds is None:
        thresholds = np.quantile(x, np.linspace(0.50, 0.995, 60))
    u_out, e_out, se_out, n_out = [], [], [], []
    for u in np.atleast_1d(thresholds):
        y = x[x > u] - u
        if y.size < min_count:
            continue
        u_out.append(u)
        e_out.append(y.mean())
        se_out.append(y.std(ddof=1) / np.sqrt(y.size))
        n_out.append(y.size)
    return (np.asarray(u_out), np.asarray(e_out),
            np.asarray(se_out), np.asarray(n_out, int))


def fit_gpd_pwm(y) -> Tuple[float, float]:
    """Probability-weighted moments for the GPD (Hosking & Wallis 1987).

    ``xi = 2 - a0/(a0 - 2 a1)``, ``beta = 2 a0 a1/(a0 - 2 a1)`` with
    ``a0 = mean(y)`` and ``a1 = mean(y (1 - F))``. Closed form, no optimiser,
    unbiased-ish for ``xi < 0.5`` — and therefore also the honest warning that it
    degrades exactly in the regime (xi -> 1/2, alpha -> 2) markets live in.
    """
    y = np.sort(np.asarray(y, float).ravel())
    n = y.size
    if n < 5:
        raise ValueError("need at least 5 exceedances")
    f = (np.arange(1, n + 1) - 0.35) / n
    a0 = y.mean()
    a1 = np.mean(y * (1.0 - f))
    d = a0 - 2.0 * a1
    if abs(d) < 1e-14:
        return 0.0, float(a0)
    xi = 2.0 - a0 / d
    beta = 2.0 * a0 * a1 / d
    if beta <= 0:
        return 0.0, float(a0)
    return float(xi), float(beta)


def _gpd_nll(params, y) -> float:
    xi, beta = params
    if beta <= 0:
        return np.inf
    z = 1.0 + xi * y / beta
    if xi < 0 and np.any(z <= 0):
        return np.inf
    if abs(xi) < 1e-10:
        return float(y.size * np.log(beta) + y.sum() / beta)
    if np.any(z <= 0):
        return np.inf
    return float(y.size * np.log(beta) + (1.0 + 1.0 / xi) * np.log(z).sum())


def fit_gpd(y, method: str = "mle") -> Tuple[float, float]:
    """Fit the GPD to exceedances. Returns ``(xi, beta)``; ``alpha = 1/xi``.

    Pickands–Balkema–de Haan: for essentially any distribution in a domain of
    attraction, the conditional excess distribution over a high enough threshold
    converges to this two-parameter family. That is the licence for fitting a
    parametric tail to a sample whose parent we do not know. However, "high enough"
    is doing all of the work, which is why ``threshold_stability`` exists.

    method ∈ {"mle", "pwm"}; MLE starts from the PWM fit and falls back to it.
    """
    y = np.asarray(y, float).ravel()
    y = y[y > 0]
    xi0, beta0 = fit_gpd_pwm(y)
    if method == "pwm":
        return xi0, beta0
    if method != "mle":
        raise ValueError(f"unknown method {method!r}")
    res = optimize.minimize(_gpd_nll, np.array([xi0, beta0]), args=(y,),
                            method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    if not res.success or not np.isfinite(res.fun) or res.x[1] <= 0:
        return xi0, beta0
    return float(res.x[0]), float(res.x[1])


def gpd_profile_ci(y, level: float = 0.95, grid: int = 200,
                   span: float = 0.6) -> Tuple[float, float]:
    """Profile-likelihood interval for ``xi``: the set where the deviance
    ``2 (l_max - l_profile(xi))`` stays below the chi-square(1) critical value.

    Deliberately *not* the Wald interval ``xi_hat +/- z*se``. The GPD likelihood
    in xi is markedly asymmetric — long to the right — so the symmetric interval
    is too narrow exactly where being too narrow costs you, which is on the heavy
    side. 
    """
    y = np.asarray(y, float).ravel()
    y = y[y > 0]
    xi_hat, beta_hat = fit_gpd(y, "mle")
    l_max = -_gpd_nll((xi_hat, beta_hat), y)
    crit = chi2.ppf(level, 1)
    xis = np.linspace(xi_hat - span, xi_hat + span, grid)
    ok = []
    for xi in xis:
        r = optimize.minimize_scalar(
            lambda b: _gpd_nll((xi, b), y),
            bounds=(1e-8, 20.0 * max(beta_hat, 1e-8)), method="bounded")
        if 2.0 * (l_max + r.fun) <= crit:
            ok.append(xi)
    if not ok:
        return xi_hat, xi_hat
    return float(min(ok)), float(max(ok))


def threshold_stability(x, thresholds=None, min_count: int = 30,
                        method: str = "mle"):
    """Fit the GPD at every threshold and report ``xi`` and the *modified* scale
    ``beta* = beta - xi*u``, which is threshold-invariant if the model holds.

    Returns ``(u, xi, beta_star, count)``. Both curves should be flat above the
    threshold where the GPD approximation becomes valid; below it they drift
    (bias from the body), above it they get noisy (too few points). 
    """
    x = np.asarray(x, float).ravel()
    if thresholds is None:
        thresholds = np.quantile(x, np.linspace(0.80, 0.995, 40))
    u_out, xi_out, bs_out, n_out = [], [], [], []
    for u in np.atleast_1d(thresholds):
        y = x[x > u] - u
        if y.size < min_count:
            continue
        try:
            xi, beta = fit_gpd(y, method)
        except ValueError:
            continue
        u_out.append(u)
        xi_out.append(xi)
        bs_out.append(beta - xi * u)
        n_out.append(y.size)
    return (np.asarray(u_out), np.asarray(xi_out),
            np.asarray(bs_out), np.asarray(n_out, int))


def pot_var_es(x, u: float, p: float = 0.01, method: str = "mle") -> dict:
    """Peaks-over-threshold VaR and ES at tail probability ``p`` (McNeil–Frey).

        VaR_p = u + (beta/xi) [ ((n/N_u) p)^(-xi) - 1 ]
        ES_p  = VaR_p/(1 - xi) + (beta - xi*u)/(1 - xi)

    ``x`` is a **loss** series (positive = bad), so VaR and ES come back positive.
    The virtue over the empirical quantile is that ``p`` may sit beyond the data:
    the GPD extrapolates on a principled law rather than on the largest thing
    that happened to occur. The vice is that ES is finite only for ``xi < 1`` and
    the whole construction inherits every bit of the error in ``xi``.
    """
    x = np.asarray(x, float).ravel()
    n = x.size
    y = x[x > u] - u
    nu = y.size
    if nu < 10:
        raise ValueError(f"only {nu} exceedances above u={u:.4g}; raise the sample or lower u")
    xi, beta = fit_gpd(y, method)
    var = u + (beta / xi) * ((n / nu * p) ** (-xi) - 1.0) if abs(xi) > 1e-10 \
        else u + beta * np.log(nu / (n * p))
    es = (var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)) if xi < 1.0 else np.inf
    return {"var": float(var), "es": float(es), "xi": float(xi),
            "beta": float(beta), "n_exc": int(nu), "u": float(u),
            "alpha": float(1.0 / xi) if xi > 0 else np.inf}


# --------------------------------------------------------------------------- #
#  5. Tail-index estimators, and how wrong each one is
# --------------------------------------------------------------------------- #
def hill(x, k: int) -> float:
    """Hill estimator of ``alpha`` from the top ``k`` order statistics:
    ``alpha_hat = k / sum_{i<=k} (ln X_(i) - ln X_(k+1))``.

    The MLE for a Pareto tail above the (k+1)-th order statistic. Consistent,
    asymptotically normal (when varianc exist) with ``se(alpha_hat) = alpha/sqrt(k)``  bi.
    It is biased whenever the tail is only *approximately* Pareto, which is always. Positive
    data only.
    """
    x = np.sort(np.asarray(x, float).ravel())[::-1]
    if k < 2 or k >= x.size:
        raise ValueError("need 2 <= k < n")
    if x[k] <= 0:
        raise ValueError("Hill needs positive data above the threshold")
    h = np.mean(np.log(x[:k])) - np.log(x[k])
    return float(1.0 / h) if h > 0 else np.inf


def hill_plot(x, kmin: int = 10, kmax: Optional[int] = None):
    """Hill estimates for every ``k`` — the Hill plot. Returns
    ``(k, alpha_hat, se)`` with ``se = alpha_hat/sqrt(k)``.
    """
    x = np.sort(np.asarray(x, float).ravel())[::-1]
    n = x.size
    kmax = min(n - 1, kmax if kmax is not None else n // 4)
    ks = np.arange(kmin, kmax + 1)
    lx = np.log(x[x > 0])
    if lx.size < kmax + 1:
        raise ValueError("not enough positive observations")
    csum = np.cumsum(lx)
    h = csum[ks - 1] / ks - lx[ks]
    with np.errstate(divide="ignore"):
        a = np.where(h > 0, 1.0 / h, np.inf)
    return ks, a, a / np.sqrt(ks)


def pickands(x, k: int) -> float:
    """Pickands estimator of ``xi``:
    ``(1/ln 2) ln[(X_(k) - X_(2k)) / (X_(2k) - X_(4k))]``.

    Works for any sign of xi and needs no positivity — its price is variance,
    since it throws away all but three order statistics.
    """
    x = np.sort(np.asarray(x, float).ravel())[::-1]
    if 4 * k > x.size:
        raise ValueError("need 4k <= n")
    num = x[k - 1] - x[2 * k - 1]
    den = x[2 * k - 1] - x[4 * k - 1]
    if den <= 0 or num <= 0:
        return np.nan
    return float(np.log(num / den) / np.log(2.0))


def dedh_moment(x, k: int) -> float:
    """Dekkers–Einmahl–de Haan moment estimator of ``xi``:
    ``M1 + 1 - 0.5 / (1 - M1^2/M2)`` with ``M_j`` the j-th log-excess moment.

    A Hill that has been repaired to admit ``xi <= 0``. Positive data only.
    """
    x = np.sort(np.asarray(x, float).ravel())[::-1]
    if k < 2 or k >= x.size or x[k] <= 0:
        raise ValueError("need 2 <= k < n and positive data")
    d = np.log(x[:k]) - np.log(x[k])
    m1, m2 = d.mean(), np.mean(d ** 2)
    if m2 <= 0:
        return np.nan
    return float(m1 + 1.0 - 0.5 / (1.0 - m1 ** 2 / m2))


def loglog_ols_alpha(x, tail_frac: float = 0.05) -> float:
    """Slope of an OLS line through the top ``tail_frac`` of the log-log survival
    plot. **Included so the notebook can demonstrate that it is wrong.**

    The points on a Zipf plot are neither independent nor homoskedastic — they
    are a monotone transform of the same order statistics, so the residuals are
    strongly correlated and the extreme points, the informative ones, carry the
    least weight. The estimator is biased, its standard error is meaningless, and
    it remains the most widely used tail-exponent method in applied work. Compare
    it against ``hill`` on a sample whose alpha you set yourself.
    """
    xs, s = survival_empirical(x)
    keep = (xs > 0) & (s > 0) & (s <= tail_frac)
    if keep.sum() < 5:
        raise ValueError("not enough tail points; raise tail_frac")
    b = np.polyfit(np.log(xs[keep]), np.log(s[keep]), 1)
    return float(-b[0])


def pareto_mle_alpha(x, xm: Optional[float] = None) -> Tuple[float, float]:
    """Exact Pareto MLE ``alpha_hat = n / sum ln(x_i/xm)`` above ``xm`` (default:
    the sample minimum). Returns ``(alpha_hat, xm)``.

    Worth having in closed form because its sampling distribution is also closed
    form — see ``alpha_hat_moments``.
    """
    x = np.asarray(x, float).ravel()
    x = x[x > 0]
    xm = float(x.min()) if xm is None else float(xm)
    y = x[x >= xm]
    s = np.log(y / xm).sum()
    if s <= 0:
        return np.inf, xm
    return float(y.size / s), xm


def alpha_hat_moments(alpha: float, n: int) -> dict:
    """Exact sampling distribution of the Pareto MLE (Taleb, Appendix E).

    ``sum ln(X_i/xm) ~ Gamma(n, 1/alpha)``, so ``alpha_hat = n / that`` is
    **inverse-gamma** with shape ``n`` and scale ``n*alpha``:

        E[alpha_hat]  = alpha * n/(n-1)          (biased upward: too thin)
        median        = n*alpha / InvGammaMed
        SD[alpha_hat] = alpha * n / ((n-1) sqrt(n-2))

    Returns mean, median, sd, and a 90% interval.
    """
    if n < 3:
        raise ValueError("n must exceed 2 for a finite variance")
    from scipy.stats import invgamma
    d = invgamma(a=n, scale=n * alpha)
    return {"mean": float(alpha * n / (n - 1.0)),
            "median": float(d.median()),
            "sd": float(alpha * n / ((n - 1.0) * np.sqrt(n - 2.0))),
            "q05": float(d.ppf(0.05)),
            "q95": float(d.ppf(0.95))}


# --------------------------------------------------------------------------- #
#  6. Shadow moments — the part of the distribution you have never seen
# --------------------------------------------------------------------------- #
def pareto_mean_above(alpha: float, u: float, xm: float = 1.0) -> float:
    """``E[X 1{X > u}]`` for a Pareto(alpha, xm) tail:
    ``alpha/(alpha-1) * u^(1-alpha) * xm^alpha``, for ``u >= xm``, ``alpha > 1``."""
    if alpha <= 1:
        return np.inf
    return float(alpha / (alpha - 1.0) * u ** (1.0 - alpha) * xm ** alpha)


def expected_max_pareto(alpha: float, n: int, xm: float = 1.0) -> float:
    """``E[max of n iid Pareto] = xm * n * B(n, 1 - 1/alpha)``.

    Grows like ``n^(1/alpha)``: for alpha = 3 the record of 10,000 observations
    is only about twice the record of 1,000. Which is precisely why the largest
    thing in your sample is a poor guide to the largest thing that can happen.
    """
    if alpha <= 1:
        return np.inf
    return float(xm * n * special.beta(n, 1.0 - 1.0 / alpha))


def hidden_mean_fraction(alpha: float, n: int) -> float:
    """Share of the true mean contributed by the region **beyond the expected
    maximum of n observations** — Taleb's invisible tail (SCoFT 9.2).

    For a Pareto with ``xm = 1`` the mean is ``alpha/(alpha-1)``, the part above
    a level M is ``pareto_mean_above``, and the ratio evaluated at
    ``M = E[max of n]`` reduces to ``M^(1-alpha)``.

    Ten thousand observations buy less than you would think: the hidden share is
    0.1% at alpha = 3, 2.8% at alpha = 1.5, and **34% at alpha = 1.1**. The
    empirical average is therefore not an estimate of the mean — it is an
    estimate of the mean *of what showed up*, and it is biased downward by
    construction, always in the reassuring direction. Note also that this is the
    mildest version of the problem: expected shortfall lives entirely in the
    region concerned, so its hidden share is far larger than the mean's at every
    alpha. "The empirical distribution is not empirical."
    """
    if alpha <= 1:
        return 1.0
    m = expected_max_pareto(alpha, n)
    return float(pareto_mean_above(alpha, m) / (alpha / (alpha - 1.0)))


def dual_transform(z, h: float, l: float = 1.0) -> np.ndarray:
    """Map a variable bounded above by ``h`` onto ``[l, inf)``:
    ``phi(z) = l - h ln((h - z)/(h - l))`` (Cirillo & Taleb 2016).

    The trick behind shadow moments. Real quantities have hard upper bounds — a
    stock cannot fall more than 100%, a war cannot kill more than the population
    — so their empirical tail must bend down near the bound, and a naive fit
    reads that bending as a thin tail. Fit the *unbounded* dual instead, where a
    genuine power law can express itself, then map back.
    """
    z = np.asarray(z, float)
    if h <= l:
        raise ValueError("need h > l")
    if np.any(z >= h) or np.any(z < l):
        raise ValueError("z must lie in [l, h)")
    return l - h * np.log((h - z) / (h - l))


def dual_inverse(y, h: float, l: float = 1.0) -> np.ndarray:
    """Inverse of ``dual_transform``: ``z = h - (h - l) exp(-(y - l)/h)``."""
    y = np.asarray(y, float)
    return h - (h - l) * np.exp(-(y - l) / h)


def shadow_mean(x, h: float, l: float = 1.0, k: Optional[int] = None) -> dict:
    """Shadow (population) mean of a bounded-above sample.

    Fit a Pareto tail to the dual variable, then integrate the *implied* parent
    back through the inverse transform by Monte Carlo. Returns the shadow mean,
    the plain sample mean, their ratio, and the fitted dual alpha.

    The ratio is the number to look at. Above 1 it says the sample mean is an
    underestimate — the observed period was lucky — and by how much. 
    """
    x = np.asarray(x, float).ravel()
    x = x[(x >= l) & (x < h)]
    if x.size < 20:
        raise ValueError("need at least 20 in-range observations")
    y = dual_transform(x, h, l)
    k = max(10, y.size // 5) if k is None else k
    k = min(k, y.size - 2)
    alpha = hill(y, k)
    ys = np.sort(y)[::-1]
    u = ys[k]                                           # Pareto scale for the tail
    rng = np.random.default_rng(0)
    draws = u * rng.random(400_000) ** (-1.0 / alpha)
    tail_mean = float(np.mean(dual_inverse(draws, h, l)))
    frac = k / y.size
    body_mean = float(np.mean(x[y <= u])) if np.any(y <= u) else 0.0
    sm = frac * tail_mean + (1.0 - frac) * body_mean
    return {"shadow_mean": sm, "sample_mean": float(x.mean()),
            "ratio": float(sm / x.mean()) if x.mean() != 0 else np.nan,
            "alpha_dual": float(alpha), "k": int(k)}


def quantile_contribution(x, p: float = 0.01) -> float:
    """Share of the total carried by the top ``p`` fraction of observations.

    Taleb Chapter 16: this statistic is *biased downward* under fat tails and the
    bias grows as the tail fattens, because the sample cannot contain the
    observations that would dominate the true share. So a measured "the worst 1%
    of days carry 31% of the variance" is a floor, not an estimate — and any
    inequality or concentration measure computed this way understates.
    """
    x = np.abs(np.asarray(x, float).ravel())
    n = x.size
    k = max(1, int(np.ceil(p * n)))
    tot = x.sum()
    return float(np.sort(x)[::-1][:k].sum() / tot) if tot > 0 else np.nan


# --------------------------------------------------------------------------- #
#  7. Risk functionals — six ways to be wrong about the same number
# --------------------------------------------------------------------------- #
def var_es_empirical(losses, p: float = 0.01) -> dict:
    """Historical VaR and ES: the empirical quantile and the mean beyond it.

    Assumption-free, and hostage to the sample: ES at p = 1% on 500 days is the
    average of five numbers. It cannot exceed the worst observed loss, so it
    assigns probability zero to anything worse than what has already happened.
    """
    z = np.asarray(losses, float).ravel()
    v = float(np.quantile(z, 1.0 - p))
    tail = z[z >= v]
    return {"var": v, "es": float(tail.mean()) if tail.size else v,
            "n_tail": int(tail.size)}


def var_es_gaussian(losses, p: float = 0.01) -> dict:
    """Gaussian VaR/ES from the sample mean and standard deviation.
    ``ES = mu + sigma * phi(z_p)/p``."""
    z = np.asarray(losses, float).ravel()
    m, s = z.mean(), z.std(ddof=1)
    zp = norm.ppf(1.0 - p)
    return {"var": float(m + s * zp), "es": float(m + s * norm.pdf(zp) / p)}


def var_es_student(losses, p: float = 0.01, nu: Optional[float] = None) -> dict:
    """Student-t VaR/ES with ``nu`` fitted by MLE unless supplied.

    Better than Gaussian and still a *global* fit: it makes one distribution
    answer for the body and the tail at once, so the body — where nearly all the
    data are — drives the estimate of the tail. POT exists to avoid exactly this.
    """
    z = np.asarray(losses, float).ravel()
    if nu is None:
        nu, loc, scale = student_t.fit(z)
    else:
        loc, scale = student_t.fit(z, f0=nu)[1:]
    nu = float(nu)
    tp = student_t.ppf(1.0 - p, nu)
    var = loc + scale * tp
    es_std = (student_t.pdf(tp, nu) / p) * (nu + tp ** 2) / (nu - 1.0) if nu > 1 else np.inf
    return {"var": float(var), "es": float(loc + scale * es_std),
            "nu": nu, "loc": float(loc), "scale": float(scale)}


def var_es_ewma(losses, p: float = 0.01, lam: float = 0.94,
                dist: str = "gaussian", nu: float = 5.0) -> dict:
    """RiskMetrics-style EWMA volatility with a Gaussian or t quantile.

    The industry default. It adapts to volatility clustering, which is real and
    matters; it says nothing about the conditional *shape*, which is what kills
    you. Returns the one-step-ahead forecast from the end of the sample.
    """
    z = np.asarray(losses, float).ravel()
    w = (1.0 - lam) * lam ** np.arange(z.size)[::-1]
    w /= w.sum()
    m = float(w @ z)
    s = float(np.sqrt(w @ (z - m) ** 2))
    if dist == "gaussian":
        zp = norm.ppf(1.0 - p)
        return {"var": m + s * zp, "es": m + s * norm.pdf(zp) / p, "sigma": s}
    tp = student_t.ppf(1.0 - p, nu)
    scale = s / np.sqrt(nu / (nu - 2.0))
    es_std = (student_t.pdf(tp, nu) / p) * (nu + tp ** 2) / (nu - 1.0)
    return {"var": m + scale * tp, "es": m + scale * es_std, "sigma": s}


def var_es_gpd(losses, p: float = 0.01, u_quantile: float = 0.95) -> dict:
    """POT VaR/ES with the threshold set at a quantile of the loss sample.
    Thin wrapper over ``pot_var_es`` for the estimator shoot-out."""
    z = np.asarray(losses, float).ravel()
    return pot_var_es(z, float(np.quantile(z, u_quantile)), p)


def es_var_ratio(alpha: float) -> float:
    """``ES_p / VaR_p = alpha/(alpha-1)`` for a Pareto tail — independent of p.

    A pure shape number: 1.5 for alpha = 3, 2.0 for alpha = 2, and unbounded as
    alpha -> 1. Under a Gaussian it is about 1.15 at p = 1% and shrinks with p,
    which is the whole difference in one line.
    """
    return float(alpha / (alpha - 1.0)) if alpha > 1 else np.inf


def alpha_from_es_var_ratio(ratio: float) -> float:
    """Invert ``es_var_ratio``: ``alpha = r/(r-1)``. Gives a tail exponent from
    two quantities every risk system already reports — a free sanity check that
    needs no tail fitting at all."""
    if ratio <= 1:
        return np.inf
    return float(ratio / (ratio - 1.0))


def es_sigma_ratio(losses, p: float = 0.01) -> float:
    """``ES_p / sigma`` — the diagnostic that ended the RMT lab.

    Roughly 2.67 for a Gaussian at p = 1%. Larger means the same volatility is
    buying you a worse tail, which is exactly what minimising a second moment on
    fat-tailed data tends to do: the variance goes down and what remains is more
    concentrated in the part that ruins you.
    """
    z = np.asarray(losses, float).ravel()
    s = z.std(ddof=1)
    return float(var_es_empirical(z, p)["es"] / s) if s > 0 else np.nan


# --------------------------------------------------------------------------- #
#  8. Clustering — extremes do not arrive alone
# --------------------------------------------------------------------------- #
def extremal_index(x, u: float) -> float:
    """Ferro–Segers intervals estimator of the extremal index ``theta``.

    ``theta`` in (0, 1] is the reciprocal of the mean cluster size: ``theta = 1``
    means extremes arrive independently, ``theta = 0.4`` means a bad day is
    typically one of two-and-a-half. It rescales everything downstream — the
    effective number of independent extremes is ``theta * N_u``, so confidence
    intervals computed as though exceedances were iid are too narrow by roughly
    ``1/sqrt(theta)``, and a return level computed the same way is optimistic.
    """
    x = np.asarray(x, float).ravel()
    idx = np.flatnonzero(x > u)
    if idx.size < 3:
        return np.nan
    s = np.diff(idx).astype(float)
    n = idx.size
    if s.max() <= 2:
        th = 2.0 * s.sum() ** 2 / ((n - 1) * np.sum(s ** 2))
    else:
        th = 2.0 * np.sum(s - 1.0) ** 2 / ((n - 1) * np.sum((s - 1.0) * (s - 2.0)))
    return float(min(1.0, max(0.0, th)))


def decluster(x, u: float, runs: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Runs declustering: keep one maximum per cluster, where clusters are
    separated by at least ``runs`` consecutive observations below ``u``.

    Returns ``(cluster_maxima, their_indices)``. The standard repair before
    fitting a GPD to dependent data.
    """
    x = np.asarray(x, float).ravel()
    idx = np.flatnonzero(x > u)
    if idx.size == 0:
        return np.array([]), np.array([], int)
    breaks = np.flatnonzero(np.diff(idx) > runs)
    groups = np.split(idx, breaks + 1)
    peaks = np.array([g[np.argmax(x[g])] for g in groups], int)
    return x[peaks], peaks


def stationary_bootstrap(x, block_mean: float, reps: int, rng: np.random.Generator,
                         size: Optional[int] = None) -> np.ndarray:
    """Politis–Romano stationary bootstrap: geometric blocks, wrap-around.
    Returns ``(reps, size)``.

    Resampling iid destroys volatility clustering, which makes every bootstrap
    confidence interval on a tail statistic too narrow. Geometric block lengths
    with mean ``block_mean`` keep the clustering roughly intact while remaining
    stationary, which the fixed-block bootstrap does not.
    """
    x = np.asarray(x, float).ravel()
    n = x.size
    size = n if size is None else size
    p = 1.0 / block_mean
    start = rng.integers(0, n, size=(reps, size))
    newblock = rng.random((reps, size)) < p
    newblock[:, 0] = True
    idx = np.empty((reps, size), dtype=np.int64)
    idx[:, 0] = start[:, 0]
    for j in range(1, size):
        idx[:, j] = np.where(newblock[:, j], start[:, j], (idx[:, j - 1] + 1) % n)
    return x[idx]


# --------------------------------------------------------------------------- #
#  9. Survival & sizing — the only part that decides anything
# --------------------------------------------------------------------------- #
def drawdown(wealth) -> np.ndarray:
    """Fractional drawdown path ``w/cummax(w) - 1`` (values <= 0)."""
    w = np.asarray(wealth, float).ravel()
    return w / np.maximum.accumulate(w) - 1.0


def max_drawdown(wealth) -> float:
    """Worst fractional peak-to-trough decline (a negative number)."""
    return float(drawdown(wealth).min())


def growth_rate(returns, f: float = 1.0) -> float:
    """Time-average (compound) growth rate ``mean log(1 + f r)`` — the thing an
    individual trajectory actually earns.

    Not ``log(1 + f * mean(r))``, which is the ensemble average and belongs to
    nobody. Returns ``-inf`` if the position is ever wiped out, because it is.
    """
    r = np.asarray(returns, float).ravel()
    z = 1.0 + f * r
    if np.any(z <= 0):
        return -np.inf
    return float(np.mean(np.log(z)))


def kelly_fraction(mu: float, sigma2: float) -> float:
    """Gaussian Kelly ``f* = mu / sigma^2``.

    The formula everyone quotes, valid only when the log-growth is well
    approximated by ``f mu - f^2 sigma^2/2``. That approximation is a second-order
    Taylor expansion, so it is a *thin-tail* formula. Compare with ``kelly_empirical``
    on the same data and the gap is the size of the mistake.
    """
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    return float(mu / sigma2)


def kelly_empirical(returns, fmax: float = 10.0, grid: int = 2000) -> dict:
    """Growth-optimal fraction by direct maximisation of ``mean log(1 + f r)``
    over the sample, with the no-ruin constraint ``1 + f * min(r) > 0``.

    Returns f*, the growth at f*, and the Gaussian-Kelly answer for comparison.
    Under a fat left tail the constraint usually binds long before the derivative
    vanishes: the cap is set by the worst thing in your sample, and the worst thing in your 
    sample is not the worst thing there is.
    """
    r = np.asarray(returns, float).ravel()
    rmin = r.min()
    cap = fmax if rmin >= 0 else min(fmax, -1.0 / rmin * (1.0 - 1e-9))
    fs = np.linspace(0.0, cap, grid)
    g = np.array([growth_rate(r, f) for f in fs])
    i = int(np.nanargmax(g))
    return {"f_star": float(fs[i]), "growth": float(g[i]),
            "f_gaussian": kelly_fraction(r.mean(), r.var(ddof=1)),
            "f_cap": float(cap), "grid_f": fs, "grid_growth": g}


def barbell(returns, w_risky: float, safe_rate: float = 0.0) -> np.ndarray:
    """Barbell allocation: ``w_risky`` in the risky stream, the rest in cash.

    Taleb & Geman: under a left-tail constraint the maximum-entropy allocation is
    not a smoothly diversified middle but a *bimodal* one, i.e., a floor you cannot
    lose past, plus a small aggressive sleeve. Truncating the loss at the
    portfolio level is not the same as reducing volatility, and it is the
    difference between a bad year and no more years.
    """
    if not 0.0 <= w_risky <= 1.0:
        raise ValueError("w_risky must be in [0, 1]")
    r = np.asarray(returns, float).ravel()
    return w_risky * r + (1.0 - w_risky) * safe_rate


def leverage_for_es(returns, target_es: float, p: float = 0.01,
                    method: str = "empirical", **kw) -> float:
    """Leverage that puts the portfolio's ES at ``target_es`` (both positive, on
    the loss scale). ES is positively homogeneous, so this is exact:
    ``f = target_es / ES_1``.

    The tail analogue of volatility targeting, and the sizing rule the market
    notebook races against fixed leverage. Which ES you use — empirical, GPD, or
    shadow — is the entire question, and the three disagree by a factor that
    grows as the tail fattens.
    """
    r = np.asarray(returns, float).ravel()
    losses = -r
    es1 = (var_es_empirical(losses, p)["es"] if method == "empirical"
           else var_es_gpd(losses, p, **kw)["es"])
    if not np.isfinite(es1) or es1 <= 0:
        return 0.0
    return float(target_es / es1)


def simulate_wealth(returns, f: float = 1.0, w0: float = 1.0,
                    barrier: float = 0.0) -> np.ndarray:
    """Compound a return stream at leverage ``f`` with an **absorbing barrier**.

    Once wealth touches ``barrier`` it stays there forever: no recovery, no
    averaging over the branches where you survived. This is the mechanism the
    absorbing-barrier lab was about, and the reason expected returns are the
    wrong object — the expectation integrates over paths that a ruined trader
    does not get to occupy.
    """
    r = np.asarray(returns, float).ravel()
    w = np.empty(r.size + 1)
    w[0] = w0
    dead = False
    for i, ri in enumerate(r):
        if dead:
            w[i + 1] = barrier
            continue
        nxt = w[i] * (1.0 + f * ri)
        if nxt <= barrier:
            nxt, dead = barrier, True
        w[i + 1] = nxt
    return w


def ruin_probability(returns, f: float, horizon: int, reps: int = 5000,
                     barrier: float = 0.5, rng: Optional[np.random.Generator] = None,
                     block_mean: Optional[float] = None) -> dict:
    """Probability of touching ``barrier`` (as a fraction of initial wealth)
    within ``horizon`` steps, at leverage ``f``, by resampling the return stream.

    Set ``block_mean`` to resample in blocks (stationary bootstrap) rather than
    iid — under clustering the iid version materially understates ruin, since
    ruin is caused by bad days *arriving together*. Returns the ruin probability,
    the median terminal wealth among survivors, and the median max drawdown.
    """
    rng = np.random.default_rng() if rng is None else rng
    r = np.asarray(returns, float).ravel()
    paths = (rng.choice(r, size=(reps, horizon), replace=True)
             if block_mean is None
             else stationary_bootstrap(r, block_mean, reps, rng, size=horizon))
    gross = np.maximum(1.0 + f * paths, 0.0)            # a wipeout stays wiped out
    w = np.cumprod(gross, axis=1)
    ruined = np.minimum.accumulate(w, axis=1)[:, -1] <= barrier
    peak = np.maximum.accumulate(w, axis=1)
    dd = (w / np.where(peak > 0, peak, 1.0) - 1.0).min(axis=1)   # wiped out -> -1
    surv = w[~ruined, -1]
    return {"p_ruin": float(ruined.mean()),
            "median_terminal": float(np.median(surv)) if surv.size else 0.0,
            "median_max_dd": float(np.median(dd))}


# --------------------------------------------------------------------------- #
#  10. Backtests — is the risk model calibrated, or merely plausible?
# --------------------------------------------------------------------------- #
def var_hits(losses, var) -> np.ndarray:
    """Exception indicator ``1{loss > VaR}``; ``var`` scalar or per-observation."""
    z = np.asarray(losses, float).ravel()
    v = np.broadcast_to(np.asarray(var, float), z.shape)
    return (z > v).astype(int)


def kupiec_pof(hits, p: float) -> dict:
    """Kupiec proportion-of-failures test: are there the right *number* of VaR
    exceptions? Likelihood ratio, chi-square(1).

    Weak on purpose. It cannot see whether the exceptions clustered, and — the
    limitation that matters here — it says nothing about how *large* they were.
    A model can pass Kupiec while losing you everything on its one exception.
    """
    h = np.asarray(hits, int).ravel()
    n, x = h.size, int(h.sum())
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        lr = -2.0 * n * np.log(p)
    else:
        ph = x / n
        lr = -2.0 * ((n - x) * np.log((1 - p) / (1 - ph)) + x * np.log(p / ph))
    return {"lr": float(lr), "pvalue": float(chi2.sf(lr, 1)),
            "n": n, "hits": x, "rate": float(x / n), "expected": float(p)}


def christoffersen_independence(hits) -> dict:
    """Christoffersen's independence test: are exceptions serially independent,
    or do they arrive in clusters? Likelihood ratio on the transition counts.

    The test that catches volatility clustering a static VaR ignores. Failing it
    means your exceptions come in runs, which is how three bad days become a
    margin call rather than three inconveniences.
    """
    h = np.asarray(hits, int).ravel()
    a, b = h[:-1], h[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    n = n00 + n01 + n10 + n11
    if n01 + n11 == 0 or n == 0:
        return {"lr": 0.0, "pvalue": 1.0, "n00": n00, "n01": n01,
                "n10": n10, "n11": n11}
    pi = (n01 + n11) / n
    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0

    def ll(p0, p1):
        out = 0.0
        if n00: out += n00 * np.log(max(1 - p0, 1e-300))
        if n01: out += n01 * np.log(max(p0, 1e-300))
        if n10: out += n10 * np.log(max(1 - p1, 1e-300))
        if n11: out += n11 * np.log(max(p1, 1e-300))
        return out

    lr = -2.0 * (ll(pi, pi) - ll(p01, p11))
    return {"lr": float(lr), "pvalue": float(chi2.sf(lr, 1)),
            "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def christoffersen_cc(hits, p: float) -> dict:
    """Conditional coverage: Kupiec + independence, chi-square(2)."""
    a = kupiec_pof(hits, p)
    b = christoffersen_independence(hits)
    lr = a["lr"] + b["lr"]
    return {"lr": float(lr), "pvalue": float(chi2.sf(lr, 2)),
            "lr_pof": a["lr"], "lr_ind": b["lr"], "rate": a["rate"]}


def acerbi_szekely_z2(losses, var, es, p: float,
                      sim_losses: Optional[np.ndarray] = None) -> dict:
    """Acerbi–Székely Z2 test of **expected shortfall**, not just VaR:

        Z2 = sum_t [ loss_t * 1{loss_t > VaR_t} ] / (N p ES_t)  -  1

    Zero under a correct model; positive means realised tail losses exceeded what
    the model promised. This is the test that matters for our question, because
    Kupiec only counts exceptions and ES is about their size — and the RMT lab's
    closing puzzle was precisely that the estimator with the fewest large days
    had the worst ones.

    Pass ``sim_losses`` of shape ``(reps, N)`` drawn from the predictive model to
    get a Monte Carlo p-value; without it only the statistic is returned.
    """
    z = np.asarray(losses, float).ravel()
    v = np.broadcast_to(np.asarray(var, float), z.shape)
    e = np.broadcast_to(np.asarray(es, float), z.shape)
    n = z.size
    z2 = float(np.sum(z * (z > v) / e) / (n * p) - 1.0)
    out = {"z2": z2, "n": n, "hits": int(np.sum(z > v))}
    if sim_losses is not None:
        s = np.asarray(sim_losses, float)
        null = np.sum(s * (s > v) / e, axis=1) / (n * p) - 1.0
        out["pvalue"] = float(np.mean(null >= z2))
    return out


# --------------------------------------------------------------------------- #
#  11. Covariance cleaning — carried over from the RMT lab
# --------------------------------------------------------------------------- #
# Part 5 of the market notebook has to rebuild the five portfolios the previous
# lab scored (sample / clipping / Ledoit-Wolf / RIE / 1-N) so it can re-score
# them on the tail. Rather than vendor the whole of ``rmt_lab.py``, the minimum
# needed to reconstruct those P&L streams lives here, unchanged in behaviour.
# The spectral machinery this lab does *not* need — Wigner, Marchenko-Pastur
# densities, the BBP spike formulas, eigenvector overlaps, Porter-Thomas — stays
# where it belongs, in Quant-RMT-FatTails.
#
# Convention, inherited: returns matrices are ``(N, T)`` — assets in rows, time
# in columns — and eigenvalues are sorted DESCENDING. Note this is the opposite
# orientation to the 1-D loss samples the rest of this module works with.
# --------------------------------------------------------------------------- #
def sample_cov(x: np.ndarray, demean: bool = False) -> np.ndarray:
    """E = X X^T / T, for X of shape (N, T)."""
    x = np.asarray(x, float)
    if demean:
        x = x - x.mean(axis=1, keepdims=True)
    return x @ x.T / x.shape[1]


def sample_corr(x: np.ndarray, demean: bool = False) -> np.ndarray:
    """Sample correlation matrix: E rescaled to unit diagonal."""
    e = sample_cov(x, demean=demean)
    d = 1.0 / np.sqrt(np.diag(e))
    return e * np.outer(d, d)


def eigh_desc(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and matching eigenvectors of a symmetric m."""
    lam, u = np.linalg.eigh(m)
    return lam[::-1], u[:, ::-1]


def recompose(u: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Rebuild U diag(xi) U^T. Every cleaner here keeps the sample eigenvectors
    and surgically replaces the eigenvalues."""
    return (u * xi[None, :]) @ u.T


def mp_edges(q: float, sigma2: float = 1.0) -> Tuple[float, float]:
    """Support edges of the Marchenko-Pastur law: sigma2 * (1 -/+ sqrt(q))^2.

    The null hypothesis of correlation-matrix mining: with N assets, T
    observations and q = N/T, the eigenvalues of a sample covariance of
    *uncorrelated* data fill this band. Anything inside it is noise.
    """
    return sigma2 * (1.0 - np.sqrt(q)) ** 2, sigma2 * (1.0 + np.sqrt(q)) ** 2


def xi_clip(lam: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Eigenvalue clipping (Laloux, Cizeau, Bouchaud & Potters 1999).

    Keep the eigenvalues above the MP edge — they might be signal — and replace
    everything below by their common average. Descending input.
    """
    lam = np.asarray(lam, float)
    _, lp = mp_edges(q, sigma2)
    xi = lam.copy()
    bulk = lam <= lp
    if bulk.any():
        xi[bulk] = lam[bulk].mean()
    return xi


def lw_shrinkage(x: np.ndarray, demean: bool = False) -> Tuple[np.ndarray, float, float]:
    """Ledoit-Wolf (2004) linear shrinkage toward mu*I, intensity from the data.

    Returns ``(Sigma_lw, rho, mu)`` with ``Sigma = rho*mu*I + (1-rho)*E``. Same
    eigenvectors as E, eigenvalues squeezed toward mu along one straight line —
    the optimal *linear* eigenvalue map.
    """
    x = np.asarray(x, float)
    n, t = x.shape
    if demean:
        x = x - x.mean(axis=1, keepdims=True)
    e = x @ x.T / t
    mu = np.trace(e) / n
    d2 = np.sum((e - mu * np.eye(n)) ** 2) / n
    s2 = np.sum(x * x, axis=0)
    cross = np.sum(x * (e @ x), axis=0)
    b2 = (np.sum(s2 ** 2) - 2.0 * np.sum(cross) + t * np.sum(e ** 2)) / (n * t ** 2)
    b2 = min(b2, d2)
    rho = b2 / d2 if d2 > 0 else 1.0
    sigma = rho * mu * np.eye(n) + (1.0 - rho) * e
    return sigma, rho, mu


def xi_rie(lam: np.ndarray, q: float, eta: Optional[float] = None,
           preserve_trace: bool = True) -> np.ndarray:
    """Rotationally Invariant Estimator (Ledoit-Peche 2011; Bun, Bouchaud &
    Potters 2017): nonlinear shrinkage read off the observed spectrum alone.

        xi_i = lambda_i / | 1 - q + q * z_i * g(z_i) |^2 ,   z_i = lambda_i - i*eta

    with g the Stieltjes transform of the sample spectrum evaluated just below
    the real axis (eta ~ N^(-1/2), the resolution at which N eigenvalues can be
    told apart), self-term excluded. Each eigenvalue is shrunk by exactly the
    amount of its own unreliability. Valid for q < 1.
    """
    lam = np.asarray(lam, float)
    n = lam.size
    if eta is None:
        eta = 1.0 / np.sqrt(n)
    z = lam - 1j * eta
    inv = 1.0 / (z[:, None] - lam[None, :])
    np.fill_diagonal(inv, 0.0)
    g = inv.sum(axis=1) / (n - 1)
    xi = lam / np.abs(1.0 - q + q * z * g) ** 2
    if preserve_trace and xi.sum() > 0:
        xi *= lam.sum() / xi.sum()
    return xi


def clean_cov(x: np.ndarray, method: str = "rie", sigma2: float = 1.0) -> np.ndarray:
    """One-stop covariance estimate from returns ``x`` of shape (N, T).

    method in {"sample", "clip", "lw", "rie"}. The eigen-based methods keep the
    sample eigenvectors and replace the eigenvalues; the previous lab's "oracle"
    is dropped here because on real data there is no truth to consult.
    """
    x = np.asarray(x, float)
    n, t = x.shape
    q = n / t
    e = sample_cov(x)
    if method == "sample":
        return e
    if method == "lw":
        return lw_shrinkage(x)[0]
    lam, u = eigh_desc(e)
    if method == "clip":
        xi = xi_clip(lam, q, sigma2=sigma2)
    elif method == "rie":
        xi = xi_rie(lam, q)
    else:
        raise ValueError(f"unknown method {method!r}")
    return recompose(u, xi)


def min_var_weights(sigma: np.ndarray) -> np.ndarray:
    """Fully-invested minimum-variance weights  w ∝ Sigma^{-1} 1."""
    n = sigma.shape[0]
    w = np.linalg.solve(sigma, np.ones(n))
    return w / w.sum()


def day_scale(x, robust: bool = True) -> np.ndarray:
    """Per-day cross-sectional scale, shape ``(1, T)``: divide by it to strip a
    common volatility mode.

    Each day's scale comes from the N *simultaneous* observations of that day —
    across the cross-section, never through time — so nothing here needs the
    tail-dominated time series of squares that breaks GARCH estimation.
    robust=True uses sqrt(pi/2) * MAD, a first-moment object whose influence is
    linear in an extreme rather than quadratic. The constant makes it consistent
    for sigma on Gaussian cross-sections; when the cross-section is itself
    fat-tailed the constant is off, so anything with an absolute ruler attached
    (clipping's MP edge, believed vols) needs re-matching or trace renormalising.
    """
    x = np.asarray(x, float)
    if robust:
        mean = np.mean(x, axis=0, keepdims=True)
        return np.sqrt(np.pi / 2.0) * np.mean(np.abs(x - mean), axis=0, keepdims=True)
    return x.std(axis=0, keepdims=True)
