# Empirical Literature Review: Five-Minute Data and Executable Equity Alpha

Date: 2026-08-06

This review was conducted with Sider Scholar (OpenAlex metadata and paper
discovery) and SciSpace (semantic search plus method/dataset/results/limitations
columns). Scite full-text verification was unavailable after its monthly MCP
quota was exhausted. Claims below are therefore separated into: (a) findings
visible in the paper abstract or bibliographic record, and (b) details that must
be checked in the article or online appendix before reproduction. A positive
forecast score is never treated as a profit claim.

## Evidence matrix

| Paper | Data and target | Method | Reported result | Boundary for this project |
|---|---|---|---|---|
| Liu, Patton, Sheppard (2015), [Does Anything Beat 5-Minute RV?](https://doi.org/10.1016/j.jeconom.2015.02.008) | 11 years, 31 assets, five asset classes; nearly 400 realized-measure estimators and volatility forecasts | Ranking and forecast-comparison procedures | Little evidence that sophisticated measures beat 5-minute RV; low-frequency truncated RV often forecasts better | Five-minute RV is a strong baseline, not a reason to add complex features automatically; this is volatility, not alpha |
| Huddleston, Liu, Stentoft (2021), [Intraday Market Predictability](https://doi.org/10.1093/jjfinec/nbab007) | Five-minute US equity market returns; lagged returns of index constituents | Regularized linear, tree-based and ensemble models | Abstract reports Sharpe 0.98 after transaction costs; predictability varies by time of day, volatility, illiquidity and market structure | Strongest close analogue for five-minute return prediction, but market, constituent data, execution rules and cost schedule are not the repository contract; verify the full paper before copying the number |
| Aleti, Bollerslev, Siggaard (2025), [Intraday Market Return Predictability Culled from the Factor Zoo](https://doi.org/10.1287/mnsc.2023.01657) | Lagged high-frequency cross-sectional factor returns; liquid ETFs as trading vehicles | Regularized predictive regressions with continuous/jump separation | Positive out-of-sample Sharpe and alpha after costs; performance concentrates in uncertainty, tail-risk and liquidity regimes | Supports cross-sectional state conditioning and regime tests; factor-zoo breadth makes selection correction and capacity essential |
| Arismendi-Zambrano, De Genaro, Alexandre (2023), [Intraday Returns Forecasting in Brazil](https://doi.org/10.2139/ssrn.4557985) | Five-minute-ahead Ibovespa returns; lagged returns of all index constituents | Ridge, LASSO, elastic net, PCR, PLS; volatility-selected portfolios | Ridge is reported as the best cost-adjusted strategy, especially among volatile stocks | SSRN working paper rather than a settled peer-reviewed result; useful design challenger, not confirmation |
| Kolm, Turiel, Westray (2023), [Deep Order Flow Imbalance](https://doi.org/10.1111/mafi.12413) | Event-level limit-order-book data for 115 Nasdaq stocks | Neural nets on stationary order-flow inputs; LSTM comparisons | Order flow beats raw book states; effective horizon about two average price changes | Requires trades/quotes/queue information absent from five-minute OHLCV; cannot be claimed as reproducible here |
| Reboredo, Matias, Garcia-Rubio (2012), [Nonlinearity in Forecasting High-Frequency Stock Returns](https://doi.org/10.1007/S10614-011-9288-5) | S&P 500 intraday observations; 5, 10, 30 and 60 minute horizons | Random walk/AR, Markov switching, ANN and SVM | Weak statistical evidence, but nonlinear rules look better under selected economic criteria | Illustrates why economic tests must be pre-specified and cost-aware; does not establish stable net alpha |
| Narayan, Narayan, Westerlund (2015), [Order Imbalances and Chinese Returns](https://doi.org/10.1016/j.pacfin.2015.07.003) | Chinese intraday data, 2008-2010; 1-minute through 90-minute horizons | Panel predictive regression handling persistent/endogenous imbalance and cross-sectional dependence | Order imbalance predicts returns and strategy profits persist intraday | Signal is order-flow based and finer than OHLCV; abstract does not establish the repository's cost and T+1 execution contract |
| Chu, Qiu (2021), [First-Half-Hour Order Imbalance](https://doi.org/10.1002/ijfe.1960) | Chinese stocks; first 30-minute order imbalance | In-sample and out-of-sample predictive regressions and portfolios | FOIB outperforms daily imbalance and is economically significant | Requires signed order-flow/large-trade information; transaction-cost details must be independently verified |
| Kong, Zhu, Azencott (2021), [Predicting Intraday Jumps in Shenzhen](https://doi.org/10.1002/FOR.2721) | Level-2 data, 1,271 Shenzhen stocks, 5-minute intervals; next 5-minute jump arrival/direction | Liquidity and technical features; random forest and other ML | Initial evidence of jump and direction predictability | Level-2 inputs are absent from five-minute OHLCV; a next-bar signal is not directly executable as a new long position under T+1 |
| [Exploring Intraday Returns in China](https://doi.org/10.54691/bcpbm.v30i.2524) (2022) | Half-hour Chinese returns; momentum/reversal tests | Correlation, regressions and trading rules | Intraday momentum appears before costs but excess return disappears after transaction costs | Direct warning that a plausible intraday pattern is not a net strategy; venue and cost specification are lower-confidence |
| [Daily Return Distribution Forecast with Intraday Information in China](https://doi.org/10.1080/1331677X.2022.2107554) (2022) | Chinese daily returns augmented by intraday returns/realized volatility | EGARCH, EGARCH-X, realGARCH, NP and realNP; log score and CRPS combinations | realGARCH is relatively best; intraday data describe part of the daily distribution but add less than 1% information to a one-step distribution forecast in the abstract | Supports distributional rather than point targets; the small incremental mean-information claim must be checked against the exact sample and cost design |
| [A Unique T+1 Trading Rule in China](https://doi.org/10.1016/j.jbankfin.2011.09.002) (2012) | Theory plus B-share evidence comparing T+1 and T+0 | Dynamic price-manipulation model | T+1 reduces volume and volatility and changes trend-chaser welfare | The legal constraint changes the action space and target; it is not a nuisance to be omitted from labels |
| [Machine-Learning in the Chinese Stock Market](https://doi.org/10.2139/SSRN.3754339) (2020) | Daily/monthly A-share returns, quarterly fundamentals and bond yield | Gu-Kelly-Xiu-style ML asset-pricing design | Liquidity is the dominant predictor; abstract reports economically significant OOS results after costs | Important China-specific factor prior, but it is not five-minute evidence and is an SSRN working paper |
| Gu, Kelly, Xiu (2020), [Empirical Asset Pricing via Machine Learning](https://doi.org/10.1093/rfs/hhaa009) | Monthly cross-section of stocks with firm characteristics and macro variables | Regularization, trees and neural networks | Nonlinear interactions improve risk-premium measurement; momentum, liquidity and volatility recur as dominant signals | Establishes a cross-sectional modeling template, not a high-frequency execution result |
| McLean, Pontiff (2016), [Does Academic Research Destroy Stock Return Predictability?](https://doi.org/10.1111/jofi.12365) | 97 published cross-sectional predictors, with out-of-sample and post-publication periods | Portfolio return attenuation analysis | Returns are 26% lower OOS and 58% lower post-publication (abstract estimates) | Any discovered factor must be treated as a decaying, crowded hypothesis and re-tested in a genuinely later period |
| [The Term Structure of Machine Learning Alpha](https://doi.org/10.3905/jfds.2023.1.135) (2023) | Stock-return ML strategies trained at different forward horizons | Horizon-specific ML plus low-turnover portfolio construction | One-month net alpha is close to zero after 2004; longer horizons and slower signals recover positive net returns | Directly motivates D2-D20/low-turnover targets rather than five-minute turnover |
| Lalwani, Meshram, Jindal (2025), [Empirical Asset Pricing via ML: Role of Research Design Choices](https://doi.org/10.1111/eufm.70033) | 5,376 portfolios across model and design choices | Research-design sensitivity analysis | Net results vary widely; nonstandard errors can be up to five times standard errors; only about one-third remain positive after costs (abstract) | Hyperparameters, filters, portfolio rules and costs are part of the statistical experiment and must be counted |
| Bailey et al. (2016), [The Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322) | Generic investment backtests | Combinatorially symmetric cross-validation (CSCV) | Provides a framework for estimating the probability that the selected backtest is overfit | The 3,456-account surface is a multiple-testing family, not 3,456 independent confirmations |
| Bailey, Lopez de Prado (2014), [The Deflated Sharpe Ratio](https://doi.org/10.3905/JPM.2014.40.5.094) | Multiple strategy trials with non-normal returns | Sharpe adjustment for selection and non-normality | Adjusts reported Sharpe for the number of trials and higher moments | Useful diagnostic alongside White RC/Hansen SPA; it does not replace an untouched confirmation period |

## What the literature actually supports

1. Five-minute data contain information. The most reproducible information is
   conditional scale, jumps, intraday seasonality, liquidity state and some
   cross-sectional lag structure. The evidence does not imply a universal
   positive conditional mean.
2. Fine microstructure signals are not interchangeable with five-minute bars.
   Order imbalance, order-flow imbalance, queue imbalance, spread and depth
   papers require event-level trades/quotes or level-2 data. OHLCV-derived
   proxies should be labeled as proxies and tested against a null, not renamed
   as order flow.
3. Net profitability is horizon- and market-structure-dependent. Results that
   survive costs are concentrated in liquid instruments, low-turnover or
   special regimes in the strongest studies. At least one China intraday study
   finds that the apparent momentum premium vanishes after costs.
4. Volatility forecasting is a more defensible first objective than direction.
   HAR, signed jumps, semivariance, regime switching and simple realized
   measures are mandatory challengers before deep learning. Several studies
   report that simple or regime-switching models beat more complex ML models.
5. Academic success is not a transferable prior. Publication, parameter
   search, structural change and crowding reduce expected future returns. The
   full model/feature/policy family, not only the winning cell, determines the
   relevant inference problem.

## Implication for the repository

The closest valid research object is:

`Q_t = P(Y_{i,t,h}^{net}, C_{i,t,h}, A_{i,t,h} | F_t)`

where `Y` is a close-to-next-open or next-open-to-legal-exit return under T+1,
`C` is state-dependent execution cost, and `A` records whether the path is
buyable, sellable, suspended or limit-censored. A five-minute feature vector is
an input to this conditional law, not the law itself.

The next study should compare (inside nested chronological folds):

- unconditional and seasonal baselines;
- EWMA/GARCH/HAR and signed-jump volatility laws;
- direct horizon-specific conditional distributions;
- an iterated one-step kernel;
- a joint multi-horizon path law;
- a regularized cross-sectional residual-return model as a mean challenger.

The decision gate is a selection-adjusted lower confidence bound on net utility,
not accuracy, rank IC, or gross backtest return:

`LCB(Delta U_net) > 0`,

with cash as an admissible action. Until this gate passes on an unused
confirmation period, the correct conclusion is “conditional predictability
found” rather than “profitable strategy found.”

## Reproducibility cautions

- SciSpace's added columns are discovery summaries; they are not substitutes
  for reading tables, appendices and code.
- Search results include working papers and low-quality venues. Peer-reviewed
  status, DOI, sample period, cost schedule and survivorship treatment must be
  checked before using a result as a design constraint.
- Scite full-text/citation classification could not be run in this session due
  to the account's monthly MCP quota. No claim here relies on an unobserved
  Scite passage.
