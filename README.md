# Discrete variance hedging

Monte Carlo experiments on the Carr–Lee–Lorig (CLL) variance hedge and a log-contract baseline. The experiments vary rebalancing frequency and transaction costs in synthetic markets with bounded switching volatility and two-point Poisson jumps.

In the fine-grid jump experiment, CLL had about 28% lower zero-fee RMSE than the log hedge at 16,128 intervals per year. At 252 intervals, CLL had higher RMSE. Turnover fees can reverse the fine-grid advantage. The [results](CLL_RESULTS.md) report the comparisons and the [methods](CLL_METHOD.md) describe funding and jump matching.

## Reproduce the experiments

Use Python 3.10 or later from the repository root.

```text
python -m pip install -r requirements.txt
python -m unittest test_cll test_reference test_risk_diagnostics
python run_cll.py --paths 20000
python validate_cll.py cll_results
```

To reproduce the additional seed and fine-grid runs:

```text
python run_cll.py --paths 20000 --seed 20261022 --out cll_second_seed
python run_cll.py --paths 10000 --steps 16128 --models switching_vol_jumps --frequencies 252 2016 8064 16128 --batch-size 100 --seed 20261024 --out cll_fine_results
python report_cll.py
```

Each result directory contains summary statistics, paired comparisons, representative trading ledgers and simulation settings. Match the recorded NumPy version and batch size for exact sample reproduction. The [original log-contract experiment](BASELINE_METHOD.md) uses a different funding convention for its raw jump-market residuals.

## Adaptive rebalancing and verification

- [Adaptive pilot](NEXT_STAGE_RESULTS.md) compares fixed schedules with thresholds on the marked value of position changes. Run `python run_adaptive.py` and `python validate_adaptive.py adaptive_pilot`.
- [Risk diagnostics](research/ROSENBAUM_TANKOV_MAPPING.md) derive a local tracking-error measure for diffusion, jumps and scheduled regime changes. Run `python risk_diagnostics.py`.
- [Verification note](review_packet/VERIFICATION_NOTE.md) documents the replication checks and the source sign convention. Run `python run_reference_audit.py` for pathwise comparisons with a separate cash ledger.

The CLL implementation covers the real-power construction in Section 4 of the [jump-diffusion paper](https://arxiv.org/abs/2107.00554) and its variance-payoff limit. It uses exact conditional prices for ideal European claims. Fees are proportional to gross traded value. Finite-strike replication, market impact, correlated volatility and barrier claims are outside the study.
