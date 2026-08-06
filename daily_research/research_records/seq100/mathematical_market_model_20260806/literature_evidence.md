# Literature Evidence Used in the Research Contract

These references justify mathematical definitions and evaluation procedures.
They do not establish that any feature, model, or A-share strategy is
profitable.

## Information and no-arbitrage boundaries

- Harrison, J. M. and Kreps, D. M. (1979), “Martingales and arbitrage in
  multiperiod securities markets,” *Journal of Economic Theory*.
  [DOI: 10.1016/0022-0531(79)90043-7](https://doi.org/10.1016/0022-0531(79)90043-7).
  The martingale/no-arbitrage result depends on a pricing measure and does not
  specify a physical expected-return forecast.
- Hansen, L. P. and Richard, S. F. (1987), “The role of conditioning
  information in deducing testable restrictions implied by dynamic asset
  pricing models,” *Econometrica*.
  [DOI: 10.2307/1913601](https://doi.org/10.2307/1913601). The conditioning
  information set is part of the model contract; it cannot be silently
  replaced by a convenient feature vector.

## Factor structure and dimension reduction

- Chamberlain, G. and Rothschild, M. (1983), “Arbitrage, factor structure, and
  mean-variance analysis on large asset markets,” *Econometrica*.
  [DOI: 10.2307/1912275](https://doi.org/10.2307/1912275). Approximate factor
  structure motivates separating common market/industry risk from idiosyncratic
  stock information; it does not imply a particular number of factors.
- Bai, J. and Ng, S. (2002), “Determining the number of factors in approximate
  factor models,” *Econometrica*.
  [DOI: 10.1111/1468-0262.00273](https://doi.org/10.1111/1468-0262.00273).
  Factor dimension is an empirical selection problem, not a PC1 convention.

## Volatility and realized variation

- Engle, R. F. (1982), “Autoregressive conditional heteroskedasticity with
  estimates of the variance of United Kingdom inflation,” *Econometrica*.
  [DOI: 10.2307/1912773](https://doi.org/10.2307/1912773).
- Bollerslev, T. (1987), “A conditionally heteroskedastic time series model
  for speculative prices and rates of return,” *Review of Economics and
  Statistics*.
  [DOI: 10.2307/1925546](https://doi.org/10.2307/1925546). These papers motivate
  conditional variance and heavy-tail challengers, not a guaranteed GARCH law.
- Corsi, F. (2009), “A simple approximate long-memory model of realized
  volatility,” *Journal of Financial Econometrics*.
  [DOI: 10.1093/jjfinec/nbp001](https://doi.org/10.1093/jjfinec/nbp001). HAR
  aggregation is a parsimonious long-memory challenger.
- Gatheral, J., Jaisson, T. and Rosenbaum, M. (2018), “Volatility is rough,”
  *Quantitative Finance*.
  [DOI: 10.1080/14697688.2017.1393551](https://doi.org/10.1080/14697688.2017.1393551).
  Rough-volatility evidence motivates testing memory beyond a short AR/GARCH
  recursion; it does not identify the correct A-share state.
- Creal, D., Koopman, S. J. and Lucas, A. (2013), “Generalized autoregressive
  score models with applications,” *Journal of Applied Econometrics*.
  [DOI: 10.1002/jae.1279](https://doi.org/10.1002/jae.1279). Score-driven state
  updates are a later alternative to the fixed VAR kernel.
- Andersen, T. G., Bollerslev, T., Diebold, F. X. and Labys, P. (2003),
  “Modeling and forecasting realized volatility,” *Econometrica*.
  [DOI: 10.1111/1468-0262.00418](https://doi.org/10.1111/1468-0262.00418).
  Quadratic variation provides the mathematical basis for realized-volatility
  features, subject to market microstructure and aggregation limits.

## Direct, iterated, and joint forecasts

- Marcellino, M., Stock, J. H. and Watson, M. W. (2006), “A comparison of
  direct and iterated multistep AR methods for forecasting macroeconomic time
  series,” *Journal of Econometrics*.
  [DOI: 10.1016/j.jeconom.2005.07.020](https://doi.org/10.1016/j.jeconom.2005.07.020).
  Direct and iterated multi-step forecasts have different finite-sample error
  trade-offs; neither is mathematically dominant.
- Gneiting, T. and Raftery, A. E. (2007), “Strictly proper scoring rules,
  prediction, and estimation,” *JASA*.
  [DOI: 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
  Proper scores make the true conditional law the expected optimum and avoid
  choosing an arbitrary event threshold as the forecast object.
- Scheuerer, M. and Hamill, T. M. (2015), “Variogram-based proper scoring
  rules for probabilistic forecasts of multivariate quantities,” *Monthly
  Weather Review*.
  [DOI: 10.1175/MWR-D-14-00269.1](https://doi.org/10.1175/MWR-D-14-00269.1).
  Dependence-sensitive scores complement marginal log score and CRPS.

## Tail risk, dependence, and selection

- Koenker, R. and Bassett, G. (1978), “Regression quantiles,” *Econometrica*.
  [DOI: 10.2307/1913643](https://doi.org/10.2307/1913643).
- Chernozhukov, V., Fernandez-Val, I. and Galichon, A. (2010), “Quantile and
  probability curves without crossing,” *Econometrica*.
  [DOI: 10.3982/ECTA7880](https://doi.org/10.3982/ECTA7880).
- Fissler, T. and Ziegel, J. F. (2016), “Higher order elicitability and
  Osband's principle,” *Annals of Statistics*.
  [DOI: 10.1214/16-AOS1439](https://doi.org/10.1214/16-AOS1439). Quantiles,
  VaR, and ES require coherent scoring rather than isolated threshold losses.
- Newey, W. K. and West, K. D. (1987), “A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix,”
  *Econometrica*.
  [DOI: 10.2307/1913610](https://doi.org/10.2307/1913610).
- Kunsch, H. R. (1989), “The jackknife and the bootstrap for general
  stationary observations,” *Annals of Statistics*.
  [DOI: 10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265).
- White, H. (2000), “A reality check for data snooping,” *Econometrica*.
  [DOI: 10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152).
- Hansen, P. R. (2005), “A test for superior predictive ability,” *JBES*.
  [DOI: 10.1198/073500105000000063](https://doi.org/10.1198/073500105000000063).
  HAC, block bootstrap, and family-level correction are required because the
  same dates and candidate family are reused.

## Decision layer

- Kelly, J. L. (1956), “A new interpretation of information rate,” *Bell
  System Technical Journal*.
  [DOI: 10.1002/j.1538-7305.1956.tb03809.x](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x).
  Expected log wealth motivates a utility layer, not a supervised event label
  or permission to use full Kelly under parameter uncertainty.
- Rockafellar, R. T. and Uryasev, S. (2000), “Optimization of conditional
  value-at-risk,” *Journal of Risk*.
  [DOI: 10.21314/JOR.2000.038](https://doi.org/10.21314/JOR.2000.038).
- Elmachtoub, A. N. and Grigas, P. (2022), “Smart ‘predict, then optimize’,”
  *Management Science*.
  [DOI: 10.1287/mnsc.2020.3922](https://doi.org/10.1287/mnsc.2020.3922).
  Decision-focused losses are a later challenger after the optimization and
  execution contract is frozen.

