# Mathematical foundations literature evidence

This file records the external literature used to choose mathematical objects
and validation rules. It is supporting evidence for the report, not a claim
that any paper proves a profitable A-share strategy.

## Forecast evaluation and tail risk

1. [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437), Tilmann Gneiting and Adrian E. Raftery, 2007, *Journal of the American Statistical Association*, 102(477), 359-378. A strictly proper score is minimized (or maximized, depending on sign convention) in expectation by the true predictive distribution. This supports evaluating a conditional distribution rather than selecting an arbitrary event threshold.
2. [Higher order elicitability and Osband's principle](https://doi.org/10.1214/16-AOS1439), Tobias Fissler and Johanna F. Ziegel, 2016, *The Annals of Statistics*, 44, 1680-1707. Value at Risk and Expected Shortfall are jointly elicitable under regularity conditions; ES alone is not the right object for a single strictly consistent forecast score.
3. [Comparing Density Forecasts Using Threshold- and Quantile-Weighted Scoring Rules](https://doi.org/10.1198/jbes.2010.08110), Tilmann Gneiting and Ranjan Ranjan, 2011, *Journal of Business & Economic Statistics*. Tail-weighted scoring can emphasize decision-relevant regions while retaining a principled forecast-evaluation framework; it does not make a particular 8% threshold theoretically privileged.
4. [Regression Quantiles](https://doi.org/10.2307/1913643), Roger Koenker and Gilbert Bassett Jr., 1978, *Econometrica*, 46(1), 33-50. Pinball-loss minimization identifies conditional quantiles and therefore provides a principled alternative to predicting only a single threshold event.
5. [Quantile and Probability Curves Without Crossing](https://doi.org/10.3982/ECTA7880), Victor Chernozhukov, Ivan Fernandez-Val, and Alfred Galichon, 2010, *Econometrica*, 78(3), 1093-1125. Monotone rearrangement repairs crossing quantile curves and can improve estimation; separate quantile heads should not be treated as a coherent distribution without this check.

## Risk-sensitive decisions and stopping

6. [Optimization of conditional value-at-risk](https://doi.org/10.21314/JOR.2000.038), R. Tyrrell Rockafellar and Stanislav Uryasev, 2000, *The Journal of Risk*, 2(3), 21-41. CVaR has the Rockafellar-Uryasev auxiliary-variable representation used in constrained or penalized tail-risk optimization.
7. [A New Interpretation of Information Rate](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x), J. L. Kelly Jr., 1956, *Bell System Technical Journal*, 35(4), 917-926. Long-run wealth growth leads to expected log wealth under repeated favorable bets. In the project this is a decision-layer utility, not a supervised label and not permission to use full Kelly under parameter uncertainty.
8. [Optimal Stopping Methods for Investment Decisions: A Literature Review](https://doi.org/10.3390/ijfs10040096), Zhenya Liu and Yuhao Mu, 2022, *International Journal of Financial Studies*. Optimal stopping formalizes exit timing as a value problem under a specified state process and utility; a fixed timeout is one restricted policy, not a universal law.

## Dependence, data snooping, and model comparison

9. [A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix](https://doi.org/10.2307/1913610), Whitney K. Newey and Kenneth D. West, 1987, *Econometrica*, 55(3), 703-708. HAC standard errors are appropriate for serially dependent date-level loss differences when the bandwidth and asymptotic approximation are defensible.
10. [The Jackknife and the Bootstrap for General Stationary Observations](https://doi.org/10.1214/aos/1176347265), Hans R. Kunsch, 1989, *The Annals of Statistics*, 17(3). Block bootstrap preserves local time dependence; the report uses moving blocks as a sensitivity interval, not as a guarantee of stationarity.
11. [Tests of Conditional Predictive Ability](https://doi.org/10.1111/j.1468-0262.2006.00718.x), Raffaella Giacomini and Halbert White, 2006, *Econometrica*, 74(6), 1545-1578. Forecast comparison can test conditional rather than only unconditional predictive performance; regime instruments must be specified before inspecting the winning model.
12. [A Reality Check for Data Snooping](https://doi.org/10.1111/1468-0262.00152), Halbert White, 2000, *Econometrica*, 68(5), 1097-1126. A family-level bootstrap test is needed when a best strategy is chosen after trying many alternatives.
13. [A Test for Superior Predictive Ability](https://doi.org/10.1198/073500105000000063), Peter R. Hansen, 2005, *Journal of Business & Economic Statistics*, 23(4), 365-380. SPA reduces the power loss of the Reality Check from poor or irrelevant alternatives while still addressing data snooping.
14. [The probability of backtest overfitting](https://doi.org/10.21314/JCF.2016.322), David H. Bailey, Jonathan Borwein, Marcos Lopez de Prado, and Qiji Zhu, 2016, *The Journal of Computational Finance*. PBO is a diagnostic for selection instability across backtest partitions; it is meaningful only when the partitioning units are sufficiently independent.
15. [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality](https://doi.org/10.3905/JPM.2014.40.5.094), David H. Bailey and Marcos Lopez de Prado, 2014, *The Journal of Portfolio Management*, 40(5), 94-107. DSR adjusts a selected Sharpe estimate for the number of trials and non-normal returns; it is not a substitute for a correct time-series split or execution model.
16. [. . . and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059), Campbell R. Harvey, Yan Liu, and Heqing Zhu, 2015, *The Review of Financial Studies*. The paper's multiple-testing argument motivates reporting the research family and not treating every positive cross-sectional result as independent evidence.
17. [Thousands of Alpha Tests](https://doi.org/10.1093/rfs/hhaa111), Stefano Giglio, Yuan Liao, and Dacheng Xiu, 2021, *The Review of Financial Studies*, 34(7), 3456-3496. Large-scale asset-pricing tests require explicit control of false discoveries and dependence.

## Prediction followed by optimization

18. [Smart "Predict, then Optimize"](https://doi.org/10.1287/mnsc.2020.3922), Adam N. Elmachtoub and Paul Grigas, 2022, *Management Science*, 68(1), 9-26. Decision-focused losses can target downstream regret, but they require a fully specified optimization problem. For this project they are a later challenger after a properly scored distributional forecast, because execution is intentionally a separate study.

## Retrieval notes and limitations

- Scite retrieval was attempted on 2026-08-05 but returned its monthly usage-limit error. The citations above were cross-checked through DOI/Crossref metadata and Consensus/SciSpace abstracts where available; the report does not claim a Scite Smart-Citation review.
- Koenker-Bassett, Chernozhukov-Fernandez-Val-Galichon, and Giacomini-White were additionally checked against Crossref on 2026-08-05.
- These papers establish definitions, scoring consistency, risk representations, or inference procedures. None establishes that a particular feature set, take-profit level, or A-share strategy will earn money.
- The report uses the DOI links as canonical identifiers. Citation counts and access status are intentionally not used as evidence for a mathematical claim.
