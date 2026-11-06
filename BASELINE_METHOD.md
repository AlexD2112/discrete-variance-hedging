# Original log-contract baseline: methods and initial research plan

This document describes `simulate.py` only. The separate Carr–Lee–Lorig implementation has now been added; see `README.md`, `CLL_METHOD.md`, and `CLL_RESULTS.md` for current project status. References below to the unimplemented paper hedge describe the original baseline stage.

Alex Danakis — initial research scaffold, October 19, 2026.

## The first research question

How does the distribution of terminal replication error change with trading frequency, volatility dynamics, payoff sampling, and proportional trading costs?

This first implementation studies the classical log-contract variance hedge. It is a prerequisite and diagnostic benchmark, not an implementation of the full Carr–Lee–Lorig jump or barrier strategies and not a claim of new research.

## Run it

Requires Python 3.10+ and NumPy. From the folder containing this file:

```powershell
python -m pip install -r requirements.txt
python simulate.py --paths 20000 --out results
```

The delivered `results/` contains an actual 20,000-path run per model, not example output. `summary.csv` includes means, mean standard errors, approximate 95% confidence intervals for means, root mean square errors (RMSE), approximate RMSE standard errors, and quantiles. Confidence intervals concern Monte Carlo uncertainty, not model uncertainty. Tail quantile confidence intervals are not yet implemented. `diagnostics.csv` reports payoff sampling differences, terminal price ratios, quadratic variation, and jump residuals. `metadata.json` records the environment and seed; `validation.json` compares the constant-volatility simulation against exact mean and RMSE formulas. `convergence.svg` plots zero-cost error against frequency.

Run the independent output checks with `python validate_results.py results`. These check martingale means, expected quadratic variation, the jump residual formula, corrected residual means, analytic constant-volatility RMSE, and monotonicity of average fees. The delivered baseline passes all 58 checks. `grid_check/` contains a second 20,000-path run with 4,032 internal steps and seed 20261020; it is a separate Monte Carlo sample.

## Upload to GitHub

This entire folder is the project root. Upload its contents, including `simulate.py`, `validate_results.py`, `requirements.txt`, `.gitignore`, this README, and the result folders, to a new GitHub repository. Suggested name: `discrete-variance-hedging`. Suggested description: "Monte Carlo experiments on discrete log-contract variance hedging, stochastic volatility, jump residuals, and transaction costs."

The code runs locally and needs no API keys or accounts. The included results are synthetic. No repository has been created or uploaded automatically. Choose a license before inviting others to reuse the code; no license is selected on your behalf.

For a second seed and a finer internal grid:

```powershell
python simulate.py --paths 20000 --seed 20261020 --out results_second_seed
python simulate.py --paths 20000 --fine-steps 4032 --out results_finer_grid
```

Use the same batch size to reproduce a run bit for bit in the same NumPy environment. The generator consumes draws by batch, so changing batch size changes the realized sample. A comparison of trading frequencies within a run uses the same simulated paths. Separate seeds, models, or fine-grid runs are not paired samples.

## Mathematical setup

Set maturity T=1 year, S0=100, zero rates and dividends. Define X=log(S/S0). For a positive continuous asset with dS/S = sigma dW, Itô's formula gives

    Q = integral_0^T sigma_t^2 dt
      = -2 log(S_T/S0) + integral_0^T (2/S_t) dS_t.

Hold an ideal European log payoff L_T=-2 log(S_T/S0) and rebalance the stock position theta_i=2/S_ti on n equal intervals. Terminal hedge value is

    H_n = -2 log(S_T/S0) + 2 sum_i (S_t(i+1)/S_ti - 1).

We measure E_n = H_n - payoff - stock trading costs. Positive means excess terminal assets relative to the liability. This is a payoff replication residual, not a complete dealer P&L or a trading profit. We do not separately estimate the initial option premium; in the ideal continuous case it equals the fair Q claim price. Funding, market option quotes, margin, and executable option spreads are absent.

The ideal log claim may be represented by a continuum of options plus a linear position:

    -2 log(S_T/S0) = -2(S_T-S0)/S0
      + integral_0^S0 2/K^2 (K-S_T)^+ dK
      + integral_S0^infinity 2/K^2 (S_T-K)^+ dK.

We evaluate the exact log payoff directly, rather than approximating this option strip. Thus option strike truncation, strike spacing, option transaction costs, and the implementation of the static linear position are not simulated. Stock fees concern the dynamic 2/S leg only.

For constant sigma=.20 and zero fees, E[E_n]=0 and

    RMSE(E_n) = sqrt(4 n [exp(.04/n)-1-.04/n])
              ~ sqrt(2) * .04 / sqrt(n).

The code checks agreement with this exact formula within five estimated Monte Carlo standard errors, in addition to algebraic flat-path, turnover, and single-jump checks. These stochastic checks are diagnostics, not proofs.

## Models

1. Constant volatility: sigma=.20; exact lognormal increments.
2. Independent switching volatility: sigma=.10 or sqrt(.07), initially equally likely, switching with probability .02 at each subsequent daily boundary. Each state persists throughout its day. Expected integrated variance is .04, matching the constant case. Conditional daily and intraday lognormal increments are exact. This is a deliberately defined discrete-time volatility regime process, not an Euler approximation to Heston.
3. The same switching volatility plus independent jumps: intensity 2/year; log jump -.15 with probability .7 and +.10 with probability .3. Independent Poisson counts allow multiple jumps per interval. The drift compensator is -lambda E[exp(J)-1], so S is a risk-neutral martingale. Jump sizes and volatility are bounded.

The simulation grid defaults to 2,016 increments/year, with nested hedge grids n=12,24,84,252,504,1008,2016. Quadratic variation is known from integrated variance plus the sum of squared individual log jumps, rather than estimated from the simulation grid. We never approximate jump quadratic variation by the square of a sum of jumps. Because volatility changes only on daily boundaries, prices at the relevant grids are sampled exactly in distribution. There is no Euler price time-step bias in these models. A finer internal grid is a reproducibility/sampling diagnostic, not a necessary correction for a hidden diffusion discretization error.

The jump model has additional variance by construction. Compare its convergence behavior with the switching model, but do not attribute all of its higher absolute RMSE to a failure of robustness.

## Keep the payoff fixed

Two separate targets are evaluated:

- Quadratic variation Q = integrated variance + sum of squared log jumps.
- Daily realized variance RV252 = sum of squared daily log returns, on a fixed 252-day schedule independent of hedge frequency.

Even a perfect continuous hedge of Q need not match RV252 pathwise. The output makes that distinction explicit. Returns are logarithmic; quantities are raw integrated variance, not percent-volatility points. At T=1, annualized variance is numerically the same. Multiplying variance units by 10,000 converts to squared percentage-return units. Do not interpret .001 as .1% volatility or as a dollar loss without specifying a contract notional.

## What the jump experiment does and does not show

With jumps, the continuous-price hedge has the exact limiting residual

    H_cont - Q = sum_j {2[exp(J_j)-1-J_j] - J_j^2}.

This is derived directly from the jump version of Itô's formula. Its expected value under our parameters is recorded in metadata.json. More frequent stock trading does not remove it. It is an expected failure of this simpler hedge under jumps, not a failure of Carr–Lee–Lorig's construction.

The `qv_oracle_jump_residual_removed` rows subtract the realized formula to isolate remaining trading error. They are an accounting diagnostic using realized jump information, not a self-financing or implementable hedge. The actual paper's jump hedge still needs to be implemented and validated.

## Costs

Evaluate c=0,1,5 basis points of stock notional on the same paths. At a rebalance charge c |theta_new-theta_old| S_now. Include the initial stock purchase and terminal liquidation. Use contemporaneous execution prices with no bid-ask feedback or slippage. The cost model changes the terminal residual but does not alter the simulated price process. These are stylized sensitivities, not calibrated estimates. Compare positive-cost strategies by a stated objective; an apparent best frequency on this small grid is not an investment recommendation or a universal optimum.

## What to do next

### This week: understand and reproduce

Work through the identity above on paper. Read the model assumptions and replication section of the jump paper, and list each instrument and independence assumption. Explain why the barrier paper allows volatility jumps while requiring continuous asset prices. Re-run the code, change the seed, and explain every output column. Plot zero-cost RMSE against n on log axes and check the approximate -1/2 slope in the continuous cases. Learn Itô's formula, quadratic variation, stochastic integration, self-financing portfolios, and Monte Carlo standard errors as needed.

### Next: reproduce one paper hedge

Choose a single tractable claim and one stated proposition from the Carr–Lee–Lorig jump paper; discuss that choice with Professor Lorig. A variance claim with two bounded log-jump sizes is a useful candidate because it matches this simulator's finite jump support. Write out the full strategy, including its European option/power claims, holdings, initial premium, and cash account. Implement it first under ideal instruments. Confirm convergence in a setting satisfying every theorem assumption. Do not call the current baseline that hedge.

### Then: isolate practical frictions

Add finite-strike option replication and separate strike-range truncation from strike-spacing error. Add execution delay and realistic option/stock costs one at a time. For a barrier extension, distinguish continuous barrier monitoring, discrete monitoring, and delayed hedge liquidation. Use a suitable crossing correction or event simulation where justified; a barrier detected only at grid points is not a continuously monitored barrier. Do not start by combining barriers, jumps, costs, and correlation.

### Before calling this a paper

Search related work on discrete variance hedging, jump corrections, discretely sampled variance swaps, transaction costs, and the precise paper strategy. Establish what is already known. Run multiple seeds and enough paths to quantify uncertainty, especially for tail losses. Use paired comparisons or paired bootstrap intervals when claiming one strategy improves on another. Include parameter sensitivity and an explanation/theoretical scaling result, not just plots. Define a narrow contribution with Lorig's feedback. Existing textbook convergence and jump residual formulas here are validation targets, not novel discoveries.

A good first deliverable is a two-page research note: question; one hedge formula and assumptions; experiment design; one convergence figure; one paragraph interpreting the failure modes; and the next question. Publication is a later decision. Keep a research log of changes, seeds, and negative results.

## Primary reading

- Carr, Lee, Lorig, *Robust Replication of Volatility and Hybrid Derivatives on Jump Diffusions*: https://arxiv.org/abs/2107.00554 . Section 2 states the independence, finite activity, boundedness, and instrument assumptions; Section 4 treats replication. Check the version you use and record it in your note.
- Carr, Lee, Lorig, *Robust replication of barrier-style claims on price and volatility*: https://arxiv.org/abs/1508.00632 . The asset price is continuous; volatility may jump.
- Carr and Lee, *Robust Replication of Volatility Derivatives*: https://math.uchicago.edu/~rl/rrvd.pdf . Background on robust volatility claims and the log-contract benchmark.
- Lin and Lorig, *On Carr and Lee's correlation immunization strategy*: https://arxiv.org/abs/1809.10256 . Relevant before any leverage/correlation stress experiment.

This is synthetic numerical research. No market data or novelty review has been completed, and the complete paper strategy is not yet reproduced.
