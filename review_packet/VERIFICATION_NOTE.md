# CLL variance hedge: specification for independent review

Prepared November 2, 2026. Status: same-process mathematical derivation and implementation cross-check; external mathematical review is pending. No author-confirmed erratum or novel theorem is claimed.

## 1. Exact question and scope

We reproduce a real-power case of the CLL exponential construction and analytically differentiate it at the origin to hedge Q_T=[log(s)]_T, with normalized stock s_0=1, maturity T=1, zero interest and dividends. This note focuses on the variance limit. The separate exponential implementation is also cross-checked numerically.

Stock Brownian motion, volatility dynamics and asset-price jumps are independent. Volatility is bounded. The asset has compensated finite-activity jumps with two known log sizes z=(-.15,.10), at intensities (1.4,.6). European log/power claims are ideal tradable instruments with their true conditional market prices. Trading costs are zero for the continuous-time argument. Restrictions to finite monitoring/trading schedules are experimental frictions, not assumptions of the replication theorem.

Sources consulted: [CLL Section 4](https://arxiv.org/html/2107.00554), especially Theorem 4.4, Remark 4.5 and Corollary 4.8; [author-hosted published PDF](https://engineering.nyu.edu/sites/default/files/2021-07/mafi.12327.pdf). The HTML source has inconsistent-looking version labels (arXiv v1; internal August 24, 2026 date). Equation numbers here identify that rendering; freeze a specific PDF/version before manuscript submission.

## 2. Instruments and notation

Let x=log(s), q=[x], tau=T-t, and define

    P_t(p) = E_t[s_T^p],
    L_t = E_t[x_T],
    psi(p) = sum_l lambda_l [exp(p z_l)-1-p(exp(z_l)-1)],
    m_t = E_t[integral_t^T sigma_u^2 du],
    mu_J = sum_l lambda_l [z_l-(exp(z_l)-1)],
    j2 = sum_l lambda_l z_l^2.

The conditional claim values used in the simulator are

    L_t = x_t - m_t/2 + tau mu_J,
    V_t = q_t + m_t + tau j2,
    P_t(p) = exp(p x_t + tau psi(p)) E_t[exp(p(p-1)/2 * remaining IV)].

These prices use current information; the future simulated path never enters the conditional expectation. In the daily Markov volatility examples the conditional moments are calculated by summing over future state transitions.

## 3. Holdings

Choose collar powers p=(2,5), paired with 1-p=(-1,-4). Define

    D_lj = -exp(p_j z_l)+exp((1-p_j) z_l)+(2p_j-1)(exp(z_l)-1),
    r_l = 2[exp(z_l)-1-z_l]-z_l^2,
    D k = r,
    b = 2 + sum_j (1-2p_j) k_j.

Hold -2 units of L, b/s units of stock, and k_j/P_t(p_j), -k_j/P_t(1-p_j) units of each paired European instrument. Cash is whatever remains after purchasing those holdings out of current portfolio wealth. Initial wealth is V_0. At a prescribed rebalance, cash changes by minus the actual purchase cost, not by an externally supplied tracking-error correction.

For the selected support and powers, k is approximately (1.487884,-.0598802) and b approximately -1.92473. The spanning matrix is ill-conditioned (condition number about 9,080), so numerical tolerances and alternative powers must be examined. Negative k swaps the economic directions of that pair; the code's long/short leg labels describe algebraic orientation, not necessarily a positive position.

## 4. Verification outline, including risks away from stock jumps

**Asset-price jumps.** At a log jump z, with holdings frozen immediately beforehand, stock gains b(exp(z)-1), the base log position gains -2z, and pair j gains k_j[exp(p_j z)-exp((1-p_j)z)]. Their sum equals

    -2z + 2(exp(z)-1) - (Dk)(z) = z^2

at each support point. For z=-.15 this must be .0225; for z=.10 it must be .01. This matches the change of V at an independent stock jump. No realized jump amount is credited to cash by the simulator.

**Stock Brownian exposure.** The coefficient multiplying sigma dW in the gain process is

    -2 + b + sum_j k_j(2p_j-1) = 0.

The target V has no price-driving Brownian exposure when volatility is independent. Thus the construction cancels that exposure as well as matching stock jumps.

**Independent volatility information.** Complementary powers have the same coefficient p(p-1)/2 of future integrated variance. Their normalized European prices therefore have identical responses to volatility information, so their equal-value opposite positions cancel this exposure. The base position -2L carries the change in m_t that is present in V. For the daily regime market this can be checked by repricing all instruments at a state change while holding x and past q fixed; the frozen holdings generate exactly Delta V. This is now an explicit deterministic test.

**Drift and self-financing.** The target value and traded asset prices are pricing-measure martingales. Given integrability/localization, matching all their martingale risk components and the initial value identifies the continuous-time gains with V. A full mathematical reviewer should check this justification and the analytic differentiation of the exponential strategy; the numerical checks are not a substitute for that argument.

## 5. Source sign question

With the consulted rendering's F,G definitions, Theorem 4.4's balance (4.16) requires Gk=-VF for the exponential claim. The displayed positive matrix solve (4.28) appears inconsistent with that balance. Our signs are chosen by computing actual gains of held instruments at a jump. The variance limit has Dk=r above. Reversing that sign leaves a nonzero jump residual. Please check the definitions and the published PDF before interpreting this as a source error.

## 6. Discrete execution and a worked account

At node i first mark previously held assets at current prices. Let old holdings be h_old and cash B_old. Wealth before trading is B_old + sum h_old Y_i. A rebalance to h_new sets

    B_new = B_old - sum (h_new-h_old) Y_i - fees_i.

Hence wealth after trading equals wealth before trading minus fees_i. Holdings remain frozen until the next permitted trade. Claims settle automatically at T; only stock liquidation incurs a closing fee. The liability's theoretical value is a comparison series, never a wealth reset.

The new cross-engine example begins with V_0=.08047608908. Its zero-fee initial cash is about 1.92604238936 because it includes short stock and offsetting European legs. At the next daily observation its marked hedge wealth is .08019544862, while the target value is .08044632295. The difference is a genuine finite-trading tracking error; no money is added to eliminate it.

`verification/worked_ledger_zero_fees.csv` contains actual prices and units for each step. Instrument indices 0..5 are stock, log claim, powers 2,-1,5,-4. Multiplying each previous row's units by the next row's prices and adding previous cash reproduces the next wealth before trading. The one-basis-point ledger also debits every execution cost explicitly.

## 7. What has been checked

- Original 14 deterministic tests.
- Nine additional tests for the separate implementation, including a volatility-state repricing, no-future-information policy decisions, costs, and monitoring/trading separation.
- 144 pathwise primary-vs-reference comparisons: four markets, three collar bases, three frequencies, two claims and two cost settings, using 32 paths per case. Maximum observed terminal-wealth difference is about 2.11e-12.

Exact audit cases, numerical differences and source-file SHA256 hashes are in `verification/audit.json`. The reference ledger and primary engine use the same mathematical specification. External review is pending.

## 8. Questions for an external reviewer

1. Does the analytic variance limit give exactly these instrument holdings and initial cash, including the compensator terms?
2. Is the direct jump-matching sign consistent with the intended theorem and the published version?
3. Does the continuous-risk argument cover all volatility and jump risks under these assumptions, and what admissibility/integrability details need stating?
4. Is a study of collar-basis choice and observation-driven execution a useful extension, and which existing papers are closest?

No messages have been sent to a reviewer. External mathematical review remains pending.
