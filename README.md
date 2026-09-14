# Surviving the tail — an extreme-value laboratory that would rather not go bust

[![tests](https://github.com/MarcoGaloppo/EVT-FatTails-PreAsymptotics/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcoGaloppo/EVT-FatTails-PreAsymptotics/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

In our previous project, [Quant-RMT-FatTails](https://github.com/MarcoGaloppo/Quant-RMT-FatTails), we built 
the whole RMT apparatus and, on S&P data, cleaned SCM five different ways and scored them out of sample. 
We found that RIE won. Then we re-scored the very same P&L streams on the *tail* instead of on the variance.
This simple change destroyed the ranking. Indeed, we saw that whilst RIE won on volatility, it was Ledoit–Wolf
who won on expected shortfall (ES), and finally clipping turned up as the winner with respect to maximum drawdown. 
Additionally, RIE had the *worst* ES-to-volatility ratio of the lot. To wit, the harder one squeezes the second moment 
the more of what remains gets concentrated into the tail. This can be a problem. 

Those results were simply a *measurement* problem. Indeed, [Quant-RMT-FatTails](https://github.com/MarcoGaloppo/Quant-RMT-FatTails) 
was built around RMT and SCM. As such, every score effectively probed the second moment of a distribution. And this in a 
market where the max-to-sum ratio of the alleged fourth-moment, $R_4$, never converges and one day owns 12% of the fourth-moment sum.

Hence this lab. We want to measure the tail properly, discover how badly that measurement can be done, and then act anyway. 
In one line: **we learned how to see structure, we now want to learn how to survive it.**

This is particularly important because, whilst most people look at themselves like the special case and argue "that will not happen to me", 
we should always keep in mind the words of *Publilius Syrus* as reported by the big boss himself *lucio Anneo Seneca*#

### **"Cuivis potest accidere quod cuiquam potest." — "What can happen to someone, it can happen to anyone."**

## The set-up

- `evt_lab.py` — It is our toolkit, in eleven blocks: generators whose truth we know (Pareto,
  Student-t, stable, GPD/GEV, variance-preserving mixtures, stochastic volatility, GARCH,
  t-copulas), the preasymptotic diagnostics (κ, max-to-sum ratios, kurtosis under aggregation,
  MAD/STD, the *local* tail exponent), both classical EVT routes (block maxima with the GEV,
  peaks-over-threshold with the GPD), the tail-index estimators and their sampling distributions
  (Hill, Pickands, DEdH, the Pareto MLE, and the log-log regression), shadow moments for what the sample 
  never contained, six ways of computing VaR and ES, the extremal-index machinery for the fact 
  that extremes arrive in clusters, the survival layer (growth-optimal sizing, barbells, ES targeting, 
  absorbing barriers, ruin), and the necessary backtests, and finally the covariance cleaners carried 
  over from the previous lab. numpy plus scipy, and nothing else.
- `narrative_evt_simulations.ipynb` — This is our laboratory. Synthetic data only, where α, ES and the
  return level are all *set by us*, so every estimator can be scored against a given truth.
- `narrative_evt_market_data.ipynb` — the market. Same S&P panel and cross-asset futures as the
  RMT lab, plus a long index history for the sample sizes EVT actually needs, and the same five
  cleaned portfolios. This time, however, re-scored on the tail.
- `tests/test_evt_lab.py` — 71 checks, run by CI on every push. Three kinds this time. Invariants
  with exact answers (PWM is exactly `(0, 1)` on the exponential; `β* = β` for GPD data and `β* = 0`
  for Pareto data; the Kupiec and Christoffersen algebra). Seeded statistical checks against the
  theory implemented (Hill recovers α, κ recovers `2 − α`, Student-t exceedances converge to a GPD
  with `ξ = 1/ν`). And **falsification tests**, which assert that the deliberately bad
  estimators fail. 

## Simulation notebook

**Part 0** is the opening gambit, and mirrors the RMT simulation lab's SCM lie. There, the in-sample volatility
understated the truth. Here, the *empirical* expected shortfall understates the true ES in a clear majority 
of samples. Note that it cannot do otherwise: historical ES is bounded above by the worst thing that has
already happened, so it assigns probability zero to everything worse.

**Part 1** asks how much data you actually need. We use κ, max-to-sum ratios, kurtosis under aggregation.
Essentially we look at the law of medium numbers: the CLT is not wrong (of course), it is *slow*, and at
α = 3 the sample size that makes it usable is not the one we usually have. 

**Part 2** asks where the tail begins. The local exponent `−d log S / d log x` creeps up from about 1
in the body of a Student-t(3) and only reaches 3 far out, past the Karamata point. Thus, "the exponent
is 3" is a statement about a *region*. On the other hand "the data look normal" is a statement about 
sample size.

**Part 3** is EVT proper, route one: block maxima, Fisher–Tippett–Gnedenko, the three domains of
attraction. We show that, of course, the Gaussian *is* in the Gumbel domain but it gets there at 
rate `1/ln n`. To wit, the slowest useful rate in statistics. At n = 1000 the normalised maximum 
is still visibly not Gumbel. Power laws converge to Fréchet fast. **EVT's own asymptotics are sharpest exactly where the tool is needed and worst where it is not**.

**Part 4** is route two: peaks over threshold, Pickands–Balkema–de Haan, the GPD. Here, we look at 
mean-excess plots, the modified scale, threshold selection as a bias–variance dial with no objective setting. 
We also take a swing at Hill plots. 

**Part 5** is where we admit we do not know α. The Pareto MLE's sampling distribution is
inverse-gamma in closed form, so the bias is computable. In particular, we have `E[α̂] = α n/(n−1)`, 
and therefore the sampling is biased toward reporting a *thinner* tail than the truth. This problem
compounds. Indeed, the elasticity `d ln ES / d ln α` is −2.05, so a 10% error in α is a 20% error in ES.
We also look at reporting intervals for the tail exponent (i.e., Wald and likelihood intervals).

**Part 6** is the hidden tail. How much of the true mean lives beyond the largest observation you
will ever see in n draws? Closed form, and the answer is `M^(1−α)` at `M = E[max]`: at n = 10,000
that is 0.1% for α = 3, 2.8% for α = 1.5, and **34% for α = 1.1**. But most importantly the expected 
shortfall lives *entirely* in the region concerned, and its hidden share is 27–45% at n = 250 for every 
α from 1.5 to 4. 

**Part 7** is the shoot-out, the analogue of the RMT lab's cleaning race. Six estimators — empirical,
Gaussian, Student-t, EWMA-normal, EWMA-t, POT-GPD — forecasting one day ahead on a rolling window. This
is done across four data-generating processes, scored by Kupiec, Christoffersen and Acerbi–Székely.
"How often was it wrong" and "how wrong was it" turn out to be different questions with different answers.

**Part 8** is survival, and the checklist the market notebook executes. Three streams with *identical*
mean and standard deviation, differing only in shape, and put through the sizing rule that can only see
mean and standard deviation. Absorbing barriers, ruin against growth, and the time average an actual
trajectory earns versus the ensemble average.

## Headline results of the simulation laboratory

1. **Same σ, wildly different tail.** ES₉₉/σ runs from 2.66 (Gaussian) to 6.45 (Pareto(3)) at
   identical unit variance, and ES₉₉.₉/σ from 3.36 to 15.86. Volatility is a *scale*; it fixes how
   wide a distribution is and says nothing about its shape. 
2. **The historical ES is a liar.** On Pareto(3) at n = 250 the median estimate is 82.5% of the truth 
   and lands below it 74% of the time; at n = 2000 it is still low 59% of the time.
3. **κ is not a constant of a distribution, it is a function of sample size.** To wit t(3) runs 0.297 →
   0.195 → 0.155 as the aggregation goes 2 → 30 → 100. To learn what 1000 Gaussian observations
   teach, a t(3) needs ~1,900, a Pareto(3) ~3,700, and a **lognormal(2) needs 168,000**. In other words,
   fat-tailedness for inference is not the same thing as an infinite moment.
4. **R₄ is a usability test, not an existence test.** The lognormal(2) has *every* moment finite and
   still shows `R(4) = 0.543` — worse than the genuinely infinite-moment t(3) at 0.314 — because
   the sum stops being owned by its maximum only once `ln n ≫ 64`. Its true kurtosis is 9,220,560 and 
   ten million observations measure 1.6% of it. Drop to lognormal(0.5) — same family — and `R(4) = 0.0075`. 
   In a finite sample "infinite" and "finite but unreachable" are indistinguishable, and for every practical 
   purpose they are the same thing.
5. **A Student-t(3) has a local exponent of 1.06 at x = 1 and reaches 3 only past x ≈ 4.6**, where
   the survival probability is 0.96%. To wit, one observation in 104 even qualifies as being in the region
   where "α = 3" is true. For a t(6) it is one in 4,158. "Is this a power law?" is the wrong
   question; "over what range, and do I have data there?" is the right one.
6. **EVT's own asymptotics are slow where the tail is thin.** Fitted ξ for the normalised Gaussian
   maximum is linear in 1/ln n with R² = 0.97, and still −0.023 at n = 10⁶. With blocks of 250 days
   the Gaussian reads as *Weibull*, which is false. A Pareto is at its limit by n = 100.
7. **With eighty years of daily data, best practice gives you anything.** Fitting a GPD at the 99.5th
   percentile of 20,000 t(3) draws, the middle half of 200 independent draws lands between α̂ = 2.68
   and 5.0, with 39% coming out above 4 and 19% below 2.5. The *median* is fine (3.32–3.40). 
8. **A third of ES₉₉ is structurally invisible at one year of data.** For a Pareto tail, the share of
   the true ES beyond the expected maximum of n draws is 27–45% at n = 250 for *every* α from 1.5 to
   4, and still 6.4% at n = 2500. No estimator can recover it. There is nothing there to recover.
9. **Counting exceptions is not measuring them.** A Gaussian model on regime-switching data keeps a
   respectable exception count (1.3% against a 1.0% target) while delivering **31% less shortfall
   than it promised**. Acerbi–Székely catches this.
10. **No risk estimator sweeps the shoot-out.** Student-t wins on |Z₂| where one global shape fits
    (t(3), GARCH) and POT-GPD wins where the tail is a different animal from the body (the regime
    mixture). POT is the robust default *because* it assumes least, and pays for that in variance.
11. **Fat tails do not bleed a portfolio to death faster — they kill it instantly.** At matched μ and
    σ, the ruin probability of Gaussian, t(3) and GARCH streams is identical to three digits (0.477,
    0.471, 0.472 at 3×). However, the leverage at which *one day* ends you is 21.7× for the Gaussian 
    and **5.2× for the t(3)**.
12. **Growth per unit ruin:** full Kelly 0.35, half Kelly 1.28, unlevered 2.71, barbell 194,
    ES-target 385. The ES-target rule independently chooses to *deleverage* to about 0.41×. 

## Market notebook

**Part 0** is the data layer, inherited from [Quant-RMT-FatTails](https://github.com/MarcoGaloppo/Quant-RMT-FatTails): 
~220 liquid S&P names tagged by sector, the cross-asset futures panel, log returns by default, per-window completeness
instead of forward-filling, and **no winsorising anywhere, ever**. Plus a long index history, because a GPD fit on 2,500
days rests on 50 exceedances and that would hurt us bad.

**Part 1** diagnoses the market using preasymptotics machinery: kurtosis under aggregation, empirical κ, 
`R₄` instability, the MS plot, records and extrema, right-versus-left tail asymmetry. The RMT lab asserted 
α ≈ 3 in passing. Here it gets tested, with error bars.

**Part 2** runs EVT on the index. POT and GPD on the left tail, threshold stability, block maxima as
an independent cross-check, GPD versus empirical versus Gaussian ES, and bootstrap intervals that
respect dependence.

**Part 3** confronts the fact that tails are not iid. Volatility clustering, the extremal index,
declustering, and the effective sample size that follows. Here, we have a direct continuity with 
the previous lab: day-standardising halved `R₄` and no more — does it change α?

**Part 4** asks where fat-tailedness lives on the panel, given that the conditional shocks are fat.
Thus, **which directions are the fat ones, and does any standard construction avoid them?**

**Part 5** rebuilds the five cleaned portfolios of the previous lab, re-scored on the tail: GPD
fits, tail α, ES₉₉ three ways, ES/σ, drawdown, and each estimator's own risk forecast put through
Kupiec, Christoffersen and Acerbi–Székely. **Does covariance cleaning reduce tail risk, or only variance?**

**Part 6** is the endgame. We take the variance winner and the tail winner and size them five ways — 
fixed leverage, volatility targeting, ES targeting, barbell, α-haircut fractional Kelly — then we score
on compound growth (the time average, not the ensemble one), maximum drawdown, ruin frequency and ES, through 
2008, 2020 and 2022. We also look at the full market.

**Part 7** draws the conclusions: what we can say, what we cannot, and what the industry already
does about it.

## Headline results of the market laboratory

1. **Not one of 216 S&P names is thin.** Every single one is fat-tailed, and pooling them does not help.
   Indeed, we find that *averaging two hundred stocks leaves α exactly where it was* and makes κ worse 
   by two thirds relative to the median stock.
2. **κ measured on your own data is only ever a floor.** The bootstrap resamples days that happened, so
   it cannot see the days that did not. The 16-year window gives κ(1,30) = 0.181. But the same index over 98
   years gives 0.214. 
3. **The cross-section knows something the time series does not.** If we divide each day by the *cross-sectional*
   dispersion of that day's 199 returns, the median name's α moves 3.05 → 3.94, its R₄ 0.201 → 0.146, its
   MAD/STD 0.685 → 0.746. Names with no fourth moment fall from 198/199 to 105/199.
4. **The 1987 test, and the flexible method loses.** Let us say one fits the 2010– window, extrapolate a factor
   of two past its own worst day, and predict how often a −22.9% day arrives. Hill (k=100) on 16 years says every 
   144 years, 1.46× the observed 1-in-98. GEV on annual maxima says 79 years, i.e., 0.81×. However, POT-GPD
   says 344 to 741 years, *wrong by 3.5× to 7.5×*, because the extra shape parameter spends 
   itself describing the shoulders.
5. **The interval on ξ is honest and the interval on α is a joke.** The raw GPD fit gives ξ = 0.172 with a
   profile interval of [0.042, 0.344]. Invert it: **α ∈ [2.90, 23.95]**. The same data are consistent with 
   "no fourth moment" and with "Gaussian for every practical purpose".
6. **Extremes arrive in convoys.** The extremal index at the 99th percentile is θ = 0.181, so the 247
   exceedances up there are worth about **45 independent ones**. 1929–33, 1987 and 2008 are not 247 pieces
   of evidence about the tail. They are only a few dozen episodes.
7. **Volatility filtering removes the clustering and does not remove the fat.** EWMA filtering takes θ easily
   close to 1 — i.e., clusters of five and a half days become extremes arriving one at a time — and the
   residual α is 3.70. Across λ ∈ {0.90, 0.94, 0.97, 0.99} and windows of 22, 66 and 252 days it stays in
   [3.51, 3.72]. *The fourth moment does not exist even after conditioning on volatility.*
8. **Kesten cannot settle it, but direct measurement can.** A GARCH with perfectly thin shocks can reach the
   observed α, so the theorem alone excludes nothing. Filter to θ = 0.994 — clustering gone as completely as
   the data permit — and the residuals still have κ = 0.084 and α = 3.72. That is a measurement of the
   conditional law, not an inference from an assumed one.
9. **Idiosyncratic risk is exactly as fat as everything else.** Market component α = 3.05, residual α = 3.01,
   with 66% of a typical name's variance idiosyncratic. There is no thin half to diversify into.
10. **Dependence does essentially all the work.** Keep every marginal exactly as it is and shuffle the
    dependence away, and the equal-weight portfolio's α goes **2.34 → 7.46** with a near-Gaussian shape.
    Diversification fails not just because the marginals are fat but because the extremes arrive *together*.
11. **Covariance cleaning reduces variance and makes the tail worse.** The previous lab reproduced to the
    third decimal (RIE 0.1213 against 0.121, sample 0.1345 against 0.135). But ES₉₉/σ *rises* 3.91 → 4.27
    as cleaning gets more aggressive, α falls 2.87 → 2.20, and ξ rises 0.329 → 0.459. *The volatility ranking                                         and the tail ranking are close to inverted across all six portfolios.*
12. **The 99% VaR breaks out of sample, and it is SCM's fault, not the thin-tailed assumption.** A Gaussian VaR₉₉ 
    on the believed variance breaches at 1.9–9.1% against a 1% target. Hand it the realised variance — a
    counterfactual, not a forecast — and that falls to 1.20–1.58%. What is left over is a genuine shape
    failure of a different size: even with the variance exactly right the Gaussian under-promises the
    shortfall by *34–48%*, and its observed-to-promised exception ratio runs 1.29 at p = 1%, 1.89 at
    0.5% and *5.35 at 0.1%*. A standardised t(3) can close the gap, and closes it best for the portfolio
    whose measured α is nearest 3.
13. **Position size is an exponent, not a multiplier.** Pr(ruin) ∝ f^α, fitted rather than assumed: slopes
    of log Pr against log f come out at 2.20–3.13 against Hill estimates of 2.20–3.09. At f = 5 the chance
    of a single day wiping you out is 3.9×10⁻⁵ under the GPD and *3.1×10⁻¹⁵¹ under a Gaussian of identical σ*. 
    Same variance, incomparable consequence. And the size decision moves maximum drawdown about *seven times more* 
    than the choice among all six covariance estimators does.
14. **Ninety-eight years: every rule that let an estimate pick a large number was destroyed.** Kelly,
    empirical Kelly and *half* Kelly were all pinned at their ceiling on 19 October 1987 — any leverage
    above 1/0.229 = 4.37 dies on that one day. What lived was bounded — volatility targeting kept 90% of 
    buy-and-hold's growth for half its drawdown, and the barbell never lost more than 5.7% in a day since 1928.
15. **α is knowable as a shape and not as a level, which is why the cap cannot live inside the model.**
    ξ from a rolling 504-day window ranges −0.56 to +0.54 with a *median of 0.057* — i.e., the median two-year
    window concludes equity losses are essentially exponential. Extrapolating to a ruin-level probability
    multiplies the threshold by (1/p)^ξ: 3.5 at ξ = 0.10, *145 at ξ = 0.40*. An error of 0.3 in ξ moves
    the leverage you are permitted by a factor of forty. You cannot compute your way to a safe position
    size, so you must choose one you can afford to be wrong about.

## Running it

    pip install -r requirements.txt
    jupyter lab narrative_evt_simulations.ipynb      # synthetic, seeded, ~2.5 min end to end
    jupyter lab narrative_evt_market_data.ipynb      # real data, ~30 s once cached

    pytest tests/ -q                                 # 71 checks, ~5 s

Most of the simulation notebook's runtime is in two cells: the rolling shoot-out of Part 7 (~55 s)
and the κ estimates of Part 1 (~35 s). Everything else is seconds.

The simulation notebook is fully seeded and needs no network. The market notebook downloads prices
once via `yfinance` into `data/` (gitignored) and runs offline from the cache afterwards; `pyarrow`
is worth installing so the cache is parquet rather than a pandas-version-locked pickle. `evt_lab.py`
itself needs only numpy and scipy.

## On the use of AI

`evt_lab.py` and both notebooks were written with Claude (Anthropic) used as a pair programmer
over multiple sessions. I chose the questions, the structure, and the standard of evidence whilst 
employing Claude to write a good part of the implementation and some of the prose.

Every numerical claim in the notebooks is produced by the code in this repository and was checked
against its output. Several of Claude's results were wrong and were corrected, among them a
look-ahead bias in the Part 5 variance decomposition for example.

## Main references

### Extreme value theory

- Fisher & Tippett (1928), Math. Proc. Cambridge Philos. Soc. 24, 180-190 — *Limiting forms of the frequency distribution of the largest or smallest member of a sample*
- Gnedenko (1943), Ann. Math. 44, 423-453 — *Sur la distribution limite du terme maximum d'une série aléatoire*
- Balkema & de Haan (1974), Ann. Probab. 2, 792-804 — *Residual life time at great age*
- Pickands (1975), Ann. Statist. 3, 119-131 — *Statistical inference using extreme order statistics* 
- Hill (1975), Ann. Statist. 3, 1163-1174 — *A simple general approach to inference about the tail of a distribution*
- Hosking, Wallis, and Wood (1985), Technometrics 27, 251-261 — *Estimation of the generalized extreme-value distribution by the method of probability-weighted moments*
- Hosking & Wallis (1987), Technometrics 29, 339-349 — *Parameter and quantile estimation for the generalized Pareto distribution* 
- Smith (1987), Ann. Statist. 15, 1174-1207 — *Estimating tails of probability distributions*
- Dekkers, Einmahl, and de Haan (1989), Ann. Statist. 17, 1833-1855 — *A moment estimator for the index of an extreme-value distribution*
- Davison & Smith (1990), J. R. Stat. Soc. B 52, 393-442 — *Models for exceedances over high thresholds* (the paper that made POT practical)
- Embrechts, Klüppelberg, and Mikosch (1997) — *Modelling Extremal Events for Insurance and Finance* (i.e., the review this lab shadows)
- Coles (2001) — *An Introduction to Statistical Modeling of Extreme Values*
- de Haan & Ferreira (2006) — *Extreme Value Theory: An Introduction*

### Threshold choice

- Resnick (1997), ASTIN Bulletin 27, 139-151 — *Discussion of the Danish data on large fire insurance losses* (the original Hill horror plots)
- Drees, de Haan, and Resnick (2000), Ann. Statist. 28, 254-274 — *How to make a Hill plot*
- Danielsson, de Haan, Peng, and de Vries (2001), J. Multivar. Anal. 76, 226-248 — *Using a bootstrap method to choose the sample fraction in tail index estimation*
- Huisman, Koedijk, Kool, and Palm (2001), J. Bus. Econ. Statist. 19, 208-216 — *Tail-index estimates in small samples*
- Goldstein, Morris, and Yen (2004), Eur. Phys. J. B 41, 255-258 — *Problems with fitting to the power-law distribution*
- Clauset, Shalizi, and Newman (2009), SIAM Review 51, 661-703 — *Power-law distributions in empirical data* (the standing case against log-log regression)

### The Taleb programme

- Taleb (2025) — *Statistical Consequences of Fat Tails* (Chapters 8, 9, 10, 16, 17 and 30 are this lab's spine)
- Taleb (2009), Int. J. Forecasting 25, 744-759 — *Errors, robustness, and the fourth quadrant*
- Taleb (2019), Int. J. Forecasting 35, 677-686 — *How much data do you need? An operational, pre-asymptotic metric for fat-tailedness* 
- Cirillo & Taleb (2016), Physica A 452, 29-45 — *On the statistical properties and tail risk of violent conflicts*
- Cirillo & Taleb (2020), Nature Physics 16, 606-613 — *Tail risk of contagious diseases*
- Geman, Geman, and Taleb (2015), Entropy 17, 3724-3737 — *Tail risk constraints and maximum entropy* 
- Taleb, Bar-Yam, and Cirillo (2022), Int. J. Forecasting 38, 413-422 — *On single point forecasts for fat-tailed variables*

### Fat tails in financial data

- Mandelbrot (1963), J. Business 36, 394-419 — *The variation of certain speculative prices*
- Fama (1965), J. Business 38, 34-105 — *The behavior of stock-market prices*
- Longin (1996), J. Business 69, 383-408 — *The asymptotic distribution of extreme stock market returns*
- Gopikrishnan, Meyer, Amaral, and Stanley (1998), Eur. Phys. J. B 3, 139-140 — *Inverse cubic law for the distribution of stock price variations*
- Plerou, Gopikrishnan, Amaral, Meyer, and Stanley (1999), PRE 60, 6519-6529 — *Scaling of the distribution of price fluctuations of individual companies*
- Cont (2001), Quantitative Finance 1, 223-236 — *Empirical properties of asset returns: stylized facts and statistical issues*
- Gabaix, Gopikrishnan, Plerou, and Stanley (2003), Nature 423, 267-270 — *A theory of power-law distributions in financial market fluctuations*
- Bouchaud & Potters (2003) — *Theory of Financial Risk and Derivative Pricing*

### Clustering

- Leadbetter (1983), Z. Wahrscheinlichkeitstheor. Verw. Geb. 65, 291-306 — *Extremes and local dependence in stationary sequences* (the extremal index)
- Engle (1982), Econometrica 50, 987-1008 — *Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation*
- Bollerslev (1986), J. Econometrics 31, 307-327 — *Generalized autoregressive conditional heteroskedasticity*
- Kesten (1973), Acta Math. 131, 207-248 — *Random difference equations and renewal theory for products of random matrices*
- de Haan, Resnick, Rootzén, and de Vries (1989), Stoch. Process. Appl. 32, 213-224 — *Extremal behaviour of solutions to a stochastic difference equation with applications to ARCH processes*
- Mikosch & Stărică (2000), Ann. Statist. 28, 1427-1451 — *Limit theory for the sample autocorrelations and extremes of a GARCH(1,1) process*
- Ferro & Segers (2003), J. R. Stat. Soc. B 65, 545-556 — *Inference for clusters of extreme values* (the intervals estimator implemented here)
- Politis & Romano (1994), JASA 89, 1303-1313 — *The stationary bootstrap*

### Risk measures and their backtests

- Artzner, Delbaen, Eber, and Heath (1999), Math. Finance 9, 203-228 — *Coherent measures of risk* 
- Acerbi & Tasche (2002), J. Banking Finance 26, 1487-1503 — *On the coherence of expected shortfall*
- Kupiec (1995), J. Derivatives 3, 73-84 — *Techniques for verifying the accuracy of risk measurement models*
- Christoffersen (1998), Int. Econ. Rev. 39, 841-862 — *Evaluating interval forecasts*
- McNeil & Frey (2000), J. Empirical Finance 7, 271-300 — *Estimation of tail-related risk measures for heteroscedastic financial time series* (the POT VaR/ES formulas)
- Gneiting (2011), JASA 106, 746-762 — *Making and evaluating point forecasts* 
- Acerbi & Székely (2014), Risk 27(11), 76-81 — *Back-testing expected shortfall* 
- Fissler & Ziegel (2016), Ann. Statist. 44, 1680-1707 — *Higher order elicitability and Osband's principle*
- Danielsson, James, Valenzuela, and Zer (2016), J. Banking Finance 69, S114-S144 — *Model risk of risk models*
- Basel Committee on Banking Supervision (2019), BCBS d457 — *Minimum capital requirements for market risk* 

### Dependence in the tail

- Embrechts, McNeil, and Straumann (2002), in *Risk Management: Value at Risk and Beyond*, CUP, 176-223 — *Correlation and dependence in risk management: properties and pitfalls*
- Longin & Solnik (2001), J. Finance 56, 649-676 — *Extreme correlation of international equity markets*
- Poon, Rockinger, and Tawn (2004), Rev. Financ. Stud. 17, 581-610 — *Extreme value dependence in financial markets: diagnostics, models, and financial implications*
- Demarta & McNeil (2005), Int. Stat. Rev. 73, 111-129 — *The t copula and related copulas*
- Chicheportiche & Bouchaud (2012), IJTAF 15, 1250019 — *The joint distribution of stock returns is not elliptical*

### Growth, ruin, and not being removed from the sample

- Kelly (1956), Bell Syst. Tech. J. 35, 917-926 — *A new interpretation of information rate*
- Thorp (1971), in *Proc. Business and Economics Section, ASA*, 215-224 — *Portfolio choice and the Kelly criterion*
- MacLean, Thorp, and Ziemba, eds. (2011) — *The Kelly Capital Growth Investment Criterion: Theory and Practice*
- Peters & Klein (2013), PRL 110, 100603 — *Ergodicity breaking in geometric Brownian motion*
- Peters & Gell-Mann (2016), Chaos 26, 023103 — *Evaluating gambles using dynamics*
- Peters (2019), Nature Physics 15, 1216-1221 — *The ergodicity problem in economics*

### Carried over from the RMT lab

- Laloux, Cizeau, Bouchaud, and Potters (1999), PRL 83, 1467 — *Noise dressing of financial correlation matrices*
- Ledoit & Wolf (2004), JMVA 88, 365-411 — *A well-conditioned estimator for large-dimensional covariance matrices*
- Ledoit & Péché (2011), Probab. Theory Relat. Fields 151, 233-264 — *Eigenvectors of some large sample covariance matrix ensembles*
- Bun, Bouchaud, and Potters (2017), Phys. Rep. 666, 1-109 — *Cleaning large correlation matrices: tools from RMT*

Author: Marco Galoppo
