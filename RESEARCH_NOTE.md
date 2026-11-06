# Initial experiment: discrete log-contract variance hedging

Alex Danakis · October 19, 2026 · preliminary computational note

## Question and scope

How much replication error remains when the continuous-price variance hedge is traded only at discrete dates? How do stochastic volatility, jumps, fixed daily payoff sampling, and trading fees change the answer?

This note studies a classical benchmark. The full Carr–Lee–Lorig hedge for jump claims and the barrier strategy have not been implemented. These computations verify known identities and build infrastructure for a focused follow-up.

## Experiment

20,000 independent paths per model, one-year maturity, zero interest, seven nested trading schedules from 12 to 2,016 rebalances/year. All schedules and fees within a model use common paths. The continuous cases have matching expected quadratic variation .04: constant volatility .20, and independent daily switching between .10 and sqrt(.07). A third case adds bounded independent Poisson jumps. Fees are 0, 1, or 5 basis points of traded stock notional. The ideal log-contract payoff is held exactly; finite option strikes and option spreads are omitted.

The hedge payoff is H_n = -2 log(S_T/S0) + 2 sum_i(S_(i+1)/S_i-1). Primary error is H_n-Q, with Q the exact log-price quadratic variation. Daily realized variance is a separate target whose observation schedule stays fixed as trading frequency changes.

## Initial results

With zero fees and constant volatility, increasing trading from 252 to 2,016 times/year reduces RMSE from **.003552 to .001254** in integrated variance units. That is a factor of 2.83, consistent with the square-root-of-frequency rate. The exact constant-volatility formula is sqrt(4n[exp(.04/n)-1-.04/n]); all seven simulated RMSEs agree within five estimated standard errors.

Under independent switching volatility, the corresponding RMSEs are **.004498 and .001574**. Volatility dynamics affect finite-frequency error even though the continuous-price replication identity remains valid.

For the jump model, the highest-frequency mean error is **-.001316**. The residual predicted by the jump version of Itô's formula is sum_j{2[exp(J_j)-1-J_j]-J_j^2}; its sample mean is **-.001319**. This is evidence that the simulator captures the known missing jump term in the continuous-price hedge. It is not evidence that the paper's jump hedge fails.

Trading fees can offset the benefits of frequent rebalancing. In the constant-volatility sample, at 5 basis points, moving from 252 to 2,016 trades/year raises RMSE from **.005702 to .009234**. These fees are stylized and include initial and terminal stock trades. This comparison does not determine an optimal market execution schedule.

## Validation and interpretation limits

The baseline passes 58 economic/statistical checks: expected quadratic variation, martingale means, known jump residuals, corrected residual means, exact constant-volatility RMSE, and average fee monotonicity. Code also checks algebraic hedge and turnover identities. Means have Monte Carlo confidence intervals in the CSV output. RMSE standard errors are delta-method approximations. Quantile confidence intervals and paired strategy-comparison intervals remain to be added.

The simulations use risk-neutral synthetic prices. Terminal replication residuals exclude a separately booked initial premium and are not a full trading P&L. Jumps add risk to the third model, so its raw RMSE is not directly comparable to the continuous cases as a pure test of robustness. The internal grid samples these particular processes exactly at relevant dates; this is not a Heston Euler simulation.

## Next research decision

Reproduce one explicit jump-hedging proposition from Carr–Lee–Lorig using the two-point bounded jump distribution already implemented. Track every option/power claim, stock holding, cash balance, and initial value. Validate that strategy under its stated assumptions before introducing finite strikes or execution delays. Ask Professor Lorig whether that claim/strategy is an appropriate starting point and which practical friction would make a useful narrow extension.

Before asserting novelty, review existing work on discrete variance hedging, jump corrections, and discretely sampled variance swaps. A useful next note should compare the baseline with the actual paper strategy, with uncertainty on paired differences and a theoretical explanation of the observed scaling.

## Sources

- Carr and Lee, *Robust Replication of Volatility Derivatives*: https://math.uchicago.edu/~rl/rrvd.pdf
- Carr, Lee, Lorig, *Robust Replication of Volatility and Hybrid Derivatives on Jump Diffusions*: https://arxiv.org/abs/2107.00554
- Carr, Lee, Lorig, *Robust replication of barrier-style claims on price and volatility*: https://arxiv.org/abs/1508.00632

Numerical values above come from the delivered `results/summary.csv` and `results/diagnostics.csv`, seed 20261019. Full setup and caveats are in README.md.
