# Monitoring, execution and a possible research contribution

Prepared November 2, 2026. This is a proposed research direction, not a novelty or optimality claim. The existing reproduction remains in place. `review_packet/VERIFICATION_NOTE.md` is the first review deliverable.

## The user's idea

Observing current prices, recomputing desired holdings and actually submitting trades are different decisions. A strategy can inspect the market at many times but retain its positions while deviations remain tolerable. Trading less often saves transaction costs but increases exposure to movement while holdings are stale. There is no requirement that observation times equal execution times.

This idea is established in the literature under threshold rebalancing, hitting-time discretization and no-transaction bands. It is a sensible direction, but merely implementing a threshold is not a new discovery.

Primary reading:

- Whalley and Wilmott, [The best hedging strategy](https://users.ox.ac.uk/~ofrcinfo/file_links/mf_papers/1999mf17.pdf): explains no-transaction bands and the importance of defining the objective and cost structure. Under proportional costs, an optimal policy may trade to a band boundary rather than all the way to the frictionless target. Our pilot trades to the target and does not claim optimality.
- Fukasawa, [Efficient Discretization of Stochastic Integrals](https://arxiv.org/abs/1204.0637): cost-sensitive asymptotic discretization. Its conclusions do not automatically transfer to this multi-instrument jump setting.
- Cai, Fukasawa, Rosenbaum and Tankov, [Optimal discretization of hedging strategies with directional views](https://arxiv.org/abs/1407.4570): barrier-hitting schedules in a specified asymptotic/control setting. A threshold execution trigger is not the same thing as a barrier in the liability payoff.
- Carr, Lee and Lorig, [jump-diffusion replication](https://arxiv.org/html/2107.00554): predictable collar positions span finitely supported jumps; more collars than jump sizes are permitted when the spanning matrix has full row rank.
- Leung and Lorig, [Optimal Static Quadratic Hedging](https://arxiv.org/abs/1506.02074): relevant later when replacing ideal claims by a finite menu of European options.
- Lin and Lorig, [correlation immunization](https://arxiv.org/abs/1809.10256): an example of testing an assumption violation, rather than treating all robustness questions as equivalent.

This focused search does not rule out an existing paper addressing precisely the proposed CLL extension. Citation-chain searches and Lorig's guidance are still needed.

Follow-up: [Rosenbaum–Tankov applicability and derived CLL risk components](ROSENBAUM_TANKOV_MAPPING.md) checks the 2014 jump-discretization article in detail. Its main hypotheses do not hold in our current simulator. The note derives and tests local risk diagnostics for our market but does not supply an optimal execution rule. It also adds Kennedy–Forsyth–Vetzal as an important jump-risk/cost benchmark.

## Working question

Can observation-driven, portfolio-level rebalancing reduce the cost of a discretely implemented CLL variance hedge for a specified tracking-error tolerance, across independent volatility environments? How does this interact with the choice of jump-spanning European instruments?

Keep the payoff Q_T fixed. Use the same initial fair capital and the same market paths for competing policies. Clearly separate (a) implementation error within theorem assumptions, (b) assumption violations such as wrong jump support, and (c) finite-strike approximation error. Do not combine them initially.

## Three progressively stronger execution rules

### A. Fixed schedule: the benchmark

Observe on a common grid and rebalance on a deterministic subset. Benchmark several frequencies, selected on training data under the SAME economic objective as the adaptive rules. Do not compare an adaptive rule against an arbitrarily weak daily benchmark and call it superior.

### B. Exposure band: implemented as a pilot

Let h*(t) be prescribed CLL units, h_old the currently held units, and Y(t) current prices of stock, log claim and power claims. At observation t calculate

    gap(t) = sum_a |h*_a(t)-h_old,a| |Y_a(t)|.

Trade to h*(t) if gap(t)>epsilon; otherwise do nothing. The implemented epsilon is in current gross instrument-value units with normalized S0=1 and claim multiplier one. It must be rescaled if the claim notional changes. This is not a delta, risk estimate, probability, or percentage of the claim price. A second pilot applies separate stock/European-block thresholds, which can break useful cross-asset coordination.

The observation grid is discrete (2,016 intervals/year), so these are grid-detected thresholds, not exact continuous hitting times. The engine charges entry and stock liquidation, and permits unconstrained cash borrowing at zero rates. Monitoring is free in the pilot; actual data/computation costs can be added later as a separate explicit cost.

### C. Risk-aware trigger: a proposed extension

Gross notional distance ignores cancellation and the severity of different jump exposures. A more informative candidate uses ell=h_old-h*(t), a local covariance rate C(t) for instrument gains, and

    R(t)=ell^T C(t) ell.

For the stock diffusion plus finite jumps, contributions include a Brownian loading outer product and sum_l lambda_l DeltaY(z_l)DeltaY(z_l)^T. Volatility-information risk must also be included; our daily regime changes are scheduled, so treating them blindly as a constant continuous hazard is not exact.

An indicative rule would compare forecast holding risk over a chosen waiting horizon H with the estimated cost K(t) of rebalancing:

    H R(t) > gamma K(t).

Units of gamma and the forecast horizon must be specified. This is a candidate heuristic until derived or solved in a control problem; it is not an optimality formula from the cited continuous-model papers. Its extra model dependence should be tested, since a purported robust hedge can become fragile if its execution signal requires an inaccurate risk model.

## A potentially more distinctive contribution: redundant collars

Different full-rank collar choices can hedge the same ideal continuous-time payoff but have different finite-trading turnover and risk. CLL allows an m-column spanning matrix with m>=number of jump sizes. With two jump sizes and m>2, Dk=r can leave freedom in k.

At a current observation, consider choosing among valid representations to minimize immediate turnover, not just choosing a calendar frequency. With current stock s, old holdings theta and old power positions u_j,v_j, the variance hedge's proposed optimization is

    minimize over k:
      c_s |2+sum_j(1-2p_j)k_j-theta s|
      + sum_j [c_j+ |k_j-u_j P_t(p_j)|
               + c_j- |-k_j-v_j P_t(1-p_j)|]
    subject to D k=r.

For fixed current prices this has an absolute-value convex/linear-program form. Position bounds or regularization may be needed to prevent excessive gross exposure and sensitivity. This objective minimizes immediate ideal-instrument execution costs within the representation class; it does NOT establish globally minimum terminal hedging risk or cost. The initial instrument menu, pricing assumptions and bounds must be fixed in advance.

A stronger candidate paper could study the interaction of redundant collar choice, threshold execution and finite-trading error, possibly deriving a stability bound or a local risk/cost result. This is an inferred research opportunity, not a claim that no one has studied it.

## Publication-oriented experiment design

1. Complete external review of the derivation and source sign convention. Keep primary and reference-engine cross-checks.
2. Fix one primary payoff and predeclare comparison metrics. Make a training, validation and final-test separation before a larger study. The current pilot has training/holdout only and is exploratory.
3. Compare cost/error frontiers, not just trade counts. Report gross turnover, fees, net and pre-cost RMSE, lower-tail shortfall, conditioning and gross exposure. Average trades alone do not measure cost.
4. Compare periodic, common exposure-band and separately updated blocks. Add the risk-aware or optimized-collar rule only after understanding these controls.
5. Stress volatility persistence and additional bounded volatility specifications with consistent market marks. Hold out scenarios/parameters. Separate model misspecification experiments from changes inside the theorem's class.
6. Vary monitoring resolution independently of trading rules. Quantify execution delay and observation cost if claiming practical performance. Do not let a signal see the next price before its trade.
7. Study sensitivity to jump intensity/support, collar powers and cost coefficients. If tuning on simulation, evaluate selected rules on fresh paths; do not report the most favorable test case as the main result.
8. Derive an explanation of the outcome: which risk exposures trigger trades, which costs are avoided, and why separately updating stock/options can hurt. A theorem, error bound, or well-supported mechanism would strengthen a computational study.
9. Only then decide whether finite-strike vanilla execution belongs in the same paper. Exact ideal claims plus realistic-looking cost coefficients are not an actual executable option strategy.

## What would count as progress, versus a publication claim

Test whether a covariance-based rule improves the cost–error tradeoff against fixed schedules and exposure thresholds on held-out paths. Assess whether the advantage survives changes in monitoring resolution, volatility and jump parameters.

Possible working title: *Observation-driven execution of jump-robust variance hedges: trading schedules, collar choice and implementation costs*. This is a working scope, not a publication promise.
