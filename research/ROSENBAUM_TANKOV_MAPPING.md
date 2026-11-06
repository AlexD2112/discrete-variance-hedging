# Rosenbaum–Tankov applicability and a CLL risk-signal specification

November 2, 2026. Scope: reading of the framework, assumptions, main error/cost results and barrier optimization; applicability analysis and local diagnostic calculations. This is not a verification of the article's technical proofs, an optimal-control solution or an external review.

## Finding

Rosenbaum and Tankov motivate comparing tracking error and trading costs under observation-driven execution. Their optimal barriers and convergence rates require assumptions that the simulated CLL hedge does not satisfy.

## Source and compact reading map

Rosenbaum and Tankov, *Asymptotically optimal discretization of hedging strategies with jumps*, Annals of Applied Probability 24(3), 1002–1048 (2014), DOI [10.1214/13-AAP940](https://doi.org/10.1214/13-AAP940). Read [arXiv v3 PDF](https://arxiv.org/pdf/1108.5940v3), dated April 28, 2014. The source uses X for scalar holdings and Y for one integrator. Conditions (HY), (HX), and Remark 3 concern absolutely continuous predictable asset variance and stable-like small jumps in holdings, with 1<alpha<2 and no Brownian holdings component. Equation (11) defines exit-interval execution. Theorems 1–2 supply error/cost asymptotics; Proposition 2 optimizes barrier shape, within the admissible class. Cost (10) sums powers of changes in units. It is not our multi-asset marked-value fee. These results are asymptotic, not finite-grid global optimality.

## Application to our project: reasons the main result does not directly apply

1. **The holdings themselves have Brownian variation.** Our stock holding is theta=b/s, so its dW coefficient between volatility boundaries is -theta*sigma, generally nonzero. Our power-claim holdings also vary with Brownian stock movements. The new diagnostic records a stock-holding coefficient about .19247 in one current-state example. The absence of a Brownian part in the paper's holdings process cannot be assumed here. Portfolio exposure cancellation does not make each holdings process Brownian-free.
2. **Asset jumps have finite activity.** Our stock has two finite-intensity jump types. Conditional jump tails do not have a nonzero stable-like small-jump limit. Selecting an alpha value by fitting two jump sizes would not repair this mismatch. Changing the simulator to meet a theorem purely to quote its optimality would change the original research question.
3. **We have a vector of traded prices.** The variance hedge uses stock, a log claim and four power claims. Their covariance and cross-hedging effects matter. Applying a scalar rule independently to each asset loses those cross terms. Defining a synthetic scalar instrument that internally rebalances the entire portfolio would hide the execution costs we want to study.
4. **Daily information jumps create predictable variance atoms.** Our volatility regime is revealed at fixed daily boundaries. European marks can jump then, even though stock remains continuous at that event. The resulting predictable covariance is not everywhere a time-density. This is an extra mismatch for any attempt to treat the whole price vector as satisfying a purely absolutely continuous bracket condition.
5. **Our costs are marked-value costs.** We charge c_a |Delta h_a| |Y_a| by instrument, plus entry and stock liquidation. A unit-change cost requires justification or a transformation before it can represent these fees. Our monitoring grid, immediate execution and finite horizon also need to be kept explicit.

These are applicability conclusions from our model equations. They are not a criticism of the paper or evidence that our research is novel. A broader literature can cover combinations of these features under other formulations.

## A correct starting identity for our own problem

Let Y be the vector of discounted traded prices, h* a valid exact continuous-time hedge and h_hat the actually held predictable positions. Before costs, with matching initial capital,

    e_T = integral_0^T (h_hat_t-h*_t)^T dY_t.

Under the pricing measure and the square-integrability conditions for the martingale integral,

    E[e_T^2] = E[integral_0^T ell_t^T d<Y>_t ell_t],
    ell_t = h_hat_t-h*_t.

This is the vector martingale-isometry identity, not a new theorem or an optimal trading policy. It uses the full predictable covariance measure, including atoms. It is a pre-cost identity. After costs, net error is e_T-C_T, and E[(e_T-C_T)^2] includes the cost/error cross term and C_T^2. Do not label the pre-cost identity a decomposition of net RMSE.

We can either minimize pre-cost MSE subject to an expected-fee budget, or state an explicit normalized risk-plus-cost objective. The pilot chose the latter. Real-world risk would require a stated probability measure and a treatment of any drift; pricing-measure calculations alone are not a calibrated physical-risk assessment.

## Derivation of the continuous-density components in our variance case

Instrument order is (s,L,P(p1),P(1-p1),P(p2),P(1-p2)). Denote the four exponents by a_j. Between daily regime changes, with current variance sigma^2, their Brownian gain vector is

    g = sigma * (s,1,a_1 P(a_1),a_2 P(a_2),a_3 P(a_3),a_4 P(a_4)).

At an asset log jump z_l the actual price-change vector is

    J_l = (s[exp(z_l)-1], z_l,
           P(a_1)[exp(a_1 z_l)-1], ..., P(a_4)[exp(a_4 z_l)-1]).

For the variance liability value V, the price-Brownian coefficient is zero and its jump at this event is z_l^2. Thus, for current held units h,

    R_diff = (h^T g)^2,
    R_jump = sum_l lambda_l (h^T J_l-z_l^2)^2,
    R_between = R_diff+R_jump.

Equivalently, when h* matches these exposures,

    C_between = g g^T + sum_l lambda_l J_l J_l^T,
    R_between = (h-h*)^T C_between (h-h*).

These are exact instantaneous conditional variance rates for our specified market away from scheduled information events. The compensators affect drift but not these variance rates. The matrix may be singular because instruments share risks; a risk signal need not invert it. No claim is made that a sample covariance diagonal or delta-only approximation is equivalent.

## Scheduled volatility-state changes must be handled separately

At a daily boundary d, condition on the previous state and stock/past QV immediately before the state revelation. Let pi_j be the conditional probabilities of new states. Calculate each possible post-revelation instrument-price vector Y_j and liability value V_j. Set

    Y_minus = sum_j pi_j Y_j,   V_minus = sum_j pi_j V_j,
    Delta Y_j = Y_j-Y_minus,   Delta V_j = V_j-V_minus.

The conditional error-variance atom for frozen pre-boundary units h is

    A_error = sum_j pi_j (h^T Delta Y_j-Delta V_j)^2.

The price covariance atom is sum_j pi_j Delta Y_j Delta Y_j^T. Using the correct pre-boundary CLL units gives zero error at this instantaneous event; stale holdings need not. This atom has units of payoff squared; it is not a per-year rate. Do not replace the scheduled state change by a constant Poisson hazard without declaring an approximation or changing the model.

The future stock at a boundary is not currently known. Forecasting a future boundary atom therefore requires conditioning over that future stock distribution, not inserting its simulated realized value into today's signal.

## Why this still does not give an optimal trigger

Immediately after resetting to a valid h*, R_between is zero. It does not follow that holding these positions for a finite interval has zero risk: future target holdings and instrument prices change, so mismatch reappears.

The risk of waiting H is a conditional expectation of accumulated future variance density and any boundary atoms. H times today's rate is only a frozen-state approximation. A full action comparison should consider the risk of waiting versus resetting, their execution costs and future decisions. Monitoring delay and finite-grid overshoot affect the comparison. Choosing a constant threshold on R, or comparing H R with gamma times costs, is a candidate heuristic until its forecast and objective are justified.

The next feasible step is a finite-horizon conditional risk forecast for this explicit market, followed by a myopic action rule labelled as such. Do not promise global optimality. Compare with the already implemented gross-exposure band, a diffusion-only signal and a jump-plus-volatility-information signal. Select parameters on training/validation samples and reserve fresh final-test seeds. Examine whether better risk forecasts improve performance enough to justify their extra model dependence.

## Other close work and novelty boundary

[Kennedy, Forsyth and Vetzal](https://cs.uwaterloo.ca/~paforsyt/transact.pdf) already combine jump-exposure penalties with trading costs while enforcing diffusion neutrality in a stock-and-options hedge. It is an important holdings-optimization benchmark; it should not be described as proving our execution-time policy optimal.

Potential contributions must go beyond calling the standard quadratic-form risk a new trigger. Candidates include a justified finite-horizon execution rule for the CLL representation, coordination across its instruments, or constrained low-turnover choice among redundant collars. Applicability gaps in one article do not prove that any such contribution is new. Search those narrower questions and obtain expert guidance.

## Artifacts and status

- `risk_diagnostics.py`: derived Brownian/jump density and scheduled-state covariance-atom diagnostics.
- `test_risk_diagnostics.py`: seven mathematical consistency checks, including fresh hedge cancellation, covariance-form equivalence and proper handling of the daily atom.
- `risk_review/diagnostics.json`: reproducible illustrative state, not a Monte Carlo performance result.
- No derived risk-trigger execution policy has been added to the adaptive pilot. Existing pilot results still concern gross exposure bands.
- No published theorem has been extended, and no human reviewer has yet confirmed this analysis.

Run `python test_risk_diagnostics.py` and `python risk_diagnostics.py` from the repository root.
