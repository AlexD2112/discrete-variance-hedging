# Carr–Lee–Lorig implementation

This module implements the real-power case of the exponential-claim strategy in Section 4 of Carr, Lee and Lorig, *Robust Replication of Volatility and Hybrid Derivatives on Jump Diffusions*, and an analytic variance-payoff limit of that strategy. It is separate from `simulate.py`.

Primary sources consulted on October 19, 2026:

- [arXiv HTML, Theorem 4.4, Remark 4.5, Proposition 4.7, Corollary 4.8](https://arxiv.org/html/2107.00554)
- [Author-hosted published PDF](https://engineering.nyu.edu/sites/default/files/2021-07/mafi.12327.pdf)

The HTML labels itself arXiv v1 but displays an August 24, 2026 internal version date. Equation references below refer to that HTML rendering. Save the exact source version when preparing a manuscript.

## Run

```powershell
python -m pip install -r requirements.txt
python test_cll.py
python run_cll.py --paths 20000
python validate_cll.py cll_results
```

`cll_hedge.py` contains the strategy and an exact conditional European-claim pricer for the chosen volatility process. `run_cll.py` produces common-path strategy comparisons, standard errors, cost sensitivities, and representative cash/position ledgers. `test_cll.py` tests mathematical identities independently of Monte Carlo convergence. `validate_cll.py` checks the generated output.

The strategy trades ideal European claims on normalized terminal stock price s_T=S_T/S0, alongside stock s_t and cash. It does not yet assemble these claims from finitely many vanilla calls and puts. Treating those claims as tradable is the ideal-instrument stage of the paper's construction.

## Definitions and jump balance

Write x=log(s), v=[x], and tau=1-t. For real p define the jump exponent

    psi(p) = sum_l lambda_l [exp(p z_l)-1-p(exp(z_l)-1)].

Let P_t(p)=E_t[s_T^p] be an observable European power-claim mark. The complementary power is 1-p. Define the deterministic spanning matrix

    D_lj = -exp(p_j z_l) + exp((1-p_j) z_l)
           + (2 p_j-1)(exp(z_l)-1).

Here the two log-jump sizes are -.15 and +.10; the two collar powers are 2 and 5, paired with -1 and -4 respectively. The resulting 2x2 D is invertible. Its condition number is about 9,080, so conditioning is explicitly recorded and the implementation rejects singular or excessively ill-conditioned systems. This is numerical sensitivity, not an estimate of economic risk. Different powers can lead to different finite-trading errors.

The equal-market-value collar at time t has holdings k_j/P_t(p_j) and -k_j/P_t(1-p_j). Its value is zero at entry. Its stock adjustment is k_j(1-2p_j)/s_t. Its gain at a jump of size z, with all positions frozen immediately before the jump, is -k_j D_lj. This direct calculation determines the sign of the solve.

### Sign convention and a source inconsistency

In the consulted HTML, the jump balance in (4.16) is Gamma + sum H(Omega_q-Omega_complement)=0. With the stated definitions of F and G, that gives **D k = -V F** for k=H RQ. However, (4.28) displays K=LH with K=VF, apparently omitting the minus sign. We follow the jump balance and verify it by computing the actual change in value of the held instruments at each allowed jump. A test verifies that using the opposite sign fails this check. This is an implementation convention derived from self-financing jump matching, not a claim of an author-confirmed erratum.

## Exponential payoff: a direct theorem case

Choose payoff exp(-a v_T), with a=.1. Restrict 0<a<1/8 so the selected power is real:

    p0 = 1/2 - sqrt(1/4-2a),
    p0(p0-1)/2 = -a.

This corresponds to omega=0 and eta=i a in the paper. Put

    chi(a) = sum_l lambda_l [exp(-a z_l^2)-1],
    A_t = exp(-p0 x_t-a v_t+tau[chi(a)-psi(p0)]),
    V_t = A_t P_t(p0),
    F_l(a) = exp(-a z_l^2)-exp(p0 z_l)+p0(exp(z_l)-1),
    k0 = solve(D, -F(a)),
    k_t = V_t k0.

At each rebalance hold:

- V_t/P_t(p0) units of the base power claim;
- k_tj/P_t(p_j) and -k_tj/P_t(1-p_j) units of each collar's two power claims;
- V_t[-p0 + sum_j k0_j(1-2p_j)]/s_t units of stock;
- cash equal to current wealth minus the market value of these positions.

The code evaluates V_t through the exact conditional distribution of future integrated variance; a deterministic test also checks the identity V_t=A_t P_t(p0). Holdings use only prices, current state and past realized quadratic variation. No simulated future volatility or future jump information is used.

## Variance payoff: analytic derivative, not numerical differencing

Since -d/da exp(-a v_T) at a=0 equals v_T, differentiate the exponential strategy analytically. At a=0, p0=0, p0'=2, and V=1. Define

    r_l = 2[exp(z_l)-1-z_l]-z_l^2,
    k = solve(D, r),
    b = 2 + sum_j k_j(1-2p_j).

The resulting strategy holds:

- -2 units of the log claim L_t=E_t[log(s_T)];
- k_j/P_t(p_j) and -k_j/P_t(1-p_j) units of each power pair;
- b/s_t stock units;
- the remaining wealth in cash.

For a jump z_l, the frozen portfolio gains

    -2 z_l + 2(exp(z_l)-1) - (D k)_l = z_l^2.

Thus the instruments generate the jump contribution of quadratic variation; the implementation never adds a realized jump correction to portfolio wealth. The analytic derivative is also checked against a small-a difference of independently calculated exponential collar weights.

## Exact conditional prices under independent daily regimes

Constant volatility is .20. Switching volatility has variance .01 or .07, equally likely initially, with .02 switching probability at each subsequent daily boundary. The current state is observable, and future states are unknown. Stock Brownian motion, regime transitions, and jumps are independent. These volatilities are bounded. Jumps have finite activity and two bounded sizes.

For c=p(p-1)/2, define M_t(c)=E_t[exp(c integral_t^T sigma^2 du)]. Then

    P_t(p) = exp(p x_t+tau psi(p)) M_t(c),
    L_t = x_t - .5 E_t[integral_t^T sigma^2 du] + tau mu_J,
    mu_J = sum_l lambda_l [z_l-exp(z_l)+1],
    variance value = v_t + E_t[remaining IV] + tau sum_l lambda_l z_l^2,
    exponential value = exp(-a v_t+tau chi(a)) M_t(-a).

The two-state pricing recursion sums over all future regime transitions exactly, including the fraction of the current day remaining. It does not use a nested Monte Carlo estimate or the realized future regime path. A final-day/two-day enumeration test checks that recursion.

The trading formulas are expressed in European market marks. The volatility model supplies a consistent synthetic market for those marks; it is not an additional volatility parameter fitted by the strategy. This experiment alone does not establish robustness to all possible independent volatility laws, and leverage/correlation violations are outside this implementation.

## Discrete execution and fair initial funding

Initial wealth equals the conditional fair price of the target payoff. At each trading interval accumulate stock and European-claim gains at positions frozen at the interval's left endpoint. Rebalancing changes the cash account; **it never resets wealth to the theoretical target value**. At maturity claims settle and stock is liquidated. Representative ledgers record positions, cash, wealth, target value and tracking error.

The variance comparator uses only the log claim, stock and cash, but starts with the **same fair variance capital**. This differs from the original baseline's raw payoff-replication residual. To relate them, let rho=sum lambda_l r_l. At T=1:

    fair-funded log-only residual = original raw baseline residual - rho.

Its limiting error is sum r(J)-rho, with mean zero but nonzero variance. Consequently, disappearance of mean bias is not evidence of successful replication. The appropriate comparison is paired squared error or RMSE, using identical funding and simulated paths.

The limiting fair-funded log-only RMSE is sqrt(sum lambda_l r_l^2). CLL's correctly spanned ideal continuous-trading hedge has no corresponding jump floor. At finite trading frequencies the extra instruments can still worsen error; the code makes no assertion that CLL must dominate on every grid.

## Trading costs and payoff sampling

Apply 0, 1 or 5 basis points to traded stock notional, including entry, rebalances and liquidation. A separate sensitivity applies 1 basis point to turnover in the ideal log/power instruments, including entry and rebalances but excluding automatic maturity settlement. This is a market-value cost proxy on the ideal instruments, not a calibrated cost of executing vanilla option strips. Gross turnover matters even when each collar has zero net value.

Fees are deducted from final wealth; at zero rates this is equivalent to debiting cash as incurred while borrowing/lending remains unrestricted and target holdings are unchanged. The representative ledgers are explicitly zero-fee ledgers.

The primary target stays fixed at quadratic variation as trading frequency changes. Fixed daily realized variance is separately reported. CLL's quadratic-variation strategy is not a pathwise replication of a discretely sampled variance contract.

## What is and is not established

Deterministic tests establish jump matching at both support points, the exponential/variance derivative relationship, power-pair price symmetry, conditional moment recursion, causality of marks, cash accounting, maturity values, and reduction to the original hedge when jump risk is absent. Monte Carlo output checks assess fair-price means and finite-frequency convergence. They do not prove a new theorem.

The current implementation does not cover arbitrary jump support, complex-power branches, barriers, finite-strike vanilla replication, empirically calibrated spreads, hidden volatility, correlation immunization, or trade impact. The next useful research extension is to keep this exact ideal-instrument benchmark and introduce one of those constraints separately.
