"""Build a modest pilot report and static cost/error SVG from saved outputs."""
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).parent


def main():
    folder=ROOT/'adaptive_pilot'
    with (folder/'summary.csv').open() as f:
        rows=list(csv.DictReader(f))
    with (folder/'paired_comparisons.csv').open() as f:
        pairs=list(csv.DictReader(f))
    selection=json.loads((folder/'locked_selection.json').read_text())['selection']
    checks=json.loads((folder/'validation.json').read_text())
    holdout=[r for r in rows if r['phase']=='holdout']
    chosen={family:next(r for r in holdout if r['name']==p['name']) for family,p in selection.items()}
    clock=chosen['periodic']; band=chosen['exposure_band']; separate=chosen['separate_band']
    improvement=100*(1-float(band['net_rmse'])/float(clock['net_rmse']))
    fee_saving=100*(1-float(band['mean_fees'])/float(clock['mean_fees']))
    pair=next(p for p in pairs if p['phase']=='holdout' and p['candidate']==band['name'] and p['metric']=='net_squared_error')
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="530" viewBox="0 0 920 530">',
        '<rect width="920" height="530" fill="white"/><g font-family="Arial" fill="#17202a">',
        '<text x="35" y="32" font-size="21">Adaptive CLL pilot: execution cost versus tracking error</text>',
        '<text x="35" y="55" font-size="12">4,000 held-out paths; 1 bp stock and ideal-claim fees; zero observation costs</text>']
    displayed=[r for r in holdout if r['policy']=='periodic']+[band,separate]
    xmin=0.; xmax=max(float(r['mean_fees']) for r in displayed)*1.18
    ymin=min(float(r['net_rmse']) for r in displayed)*.8; ymax=max(float(r['net_rmse']) for r in displayed)*1.3
    def xy(row):
        return 105+float(row['mean_fees'])/xmax*760,390-math.log(float(row['net_rmse'])/ymin)/math.log(ymax/ymin)*300
    for i in range(6):
        value=xmax*i/5; x=105+760*i/5
        lines.extend([f'<path d="M{x},90 V390" stroke="#e5e7eb"/>',
            f'<text x="{x}" y="414" font-size="11" text-anchor="middle">{value:.4f}</text>'])
        value=ymin*(ymax/ymin)**(i/5); y=390-i*60
        lines.extend([f'<path d="M105,{y} H865" stroke="#e5e7eb"/>',
            f'<text x="95" y="{y+4}" font-size="11" text-anchor="end">{value:.4f}</text>'])
    clocks=sorted([r for r in displayed if r['policy']=='periodic'],key=lambda r:int(r['frequency']))
    lines.append('<polyline points="'+' '.join(f'{xy(r)[0]},{xy(r)[1]}' for r in clocks)+'" fill="none" stroke="#94a3b8" stroke-width="2"/>')
    for row in displayed:
        x,y=xy(row); color='#94a3b8'
        if row['name']==clock['name']: color='#334155'
        elif row['policy']=='exposure_band': color='#0f766e'
        elif row['policy']=='separate_band': color='#b45309'
        lines.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
        if row['policy']=='periodic':
            lines.append(f'<text x="{x+8}" y="{y-8}" font-size="11">{row["frequency"]}/year</text>')
        else:
            label='Portfolio band' if row['policy']=='exposure_band' else 'Separate blocks'
            lines.append(f'<text x="{x-10}" y="{y+18}" text-anchor="end" font-size="12" fill="{color}">{label}</text>')
    lines.extend(['<text x="485" y="446" font-size="13" text-anchor="middle">Mean execution fees (normalized claim payoff units)</text>',
        '<text transform="translate(25,240) rotate(-90)" font-size="13" text-anchor="middle">Net terminal RMSE (variance units; log scale)</text>',
        '<text x="35" y="480" font-size="12">Dark/colored points were selected using training data; pale points show the fixed-schedule comparison grid.</text>',
        '<text x="35" y="502" font-size="11">Source: adaptive_pilot/summary.csv; one-year synthetic market; thresholds detected on a 2,016-interval grid.</text>',
        '<text x="35" y="519" font-size="11">Exploratory study: no optimality claim, no vanilla-strip execution, no market calibration.</text></g></svg>'])
    (ROOT/'adaptive_pilot.svg').write_text('\n'.join(lines),encoding='utf-8')
    audit=json.loads((ROOT/'verification/audit.json').read_text())
    note=f'''# Verification and adaptive-execution pilot

November 2, 2026. Research preparation and exploratory experiments; no external peer review or publication claim.

## Components prepared

- `review_packet/VERIFICATION_NOTE.md`: equations, source mapping, sign question, continuous-risk argument, worked account and focused human-review questions.
- `reference_ledger.py`: separately expressed market marks, holdings and explicit purchase/sale cash debits; no primary-engine strategy/pricer imports.
- `verification/audit.json`: {len(audit['checks'])} cross-engine cases, maximum pathwise wealth difference {audit['max_wealth_difference']:.3g}, with file hashes.
- `test_reference.py`: nine additional tests including causality of execution decisions, actual cost debits and a volatility-state repricing.
- `research/ADAPTIVE_RESEARCH_PLAN.md`: literature, candidate threshold formulas, redundant-collar cost minimization, experiment design and limits on novelty claims.

Both implementations use the same specification. External review of the jump-sign convention and the variance-limit derivation is pending.

## Pilot design

Observe on a grid with 2,016 intervals per year. Compare six deterministic trading frequencies and two types of exposure bands. An exposure band calculates the gross current-value difference between held and prescribed positions, and trades only when that difference exceeds a threshold. Portfolio-level bands update all hedge legs together. Separate-block bands allow stock and European positions to trigger independently.

Use 1,000 training paths to choose one policy in each family, followed by 4,000 fresh holdout paths. Selection minimizes normalized pre-cost squared tracking error plus 0.1 times normalized mean fees. Because this criterion uses pre-cost error, it does not count fees twice. Net RMSE and actual fees are reported separately. The risk/cost weight is an economic choice, not an estimated universal constant.

All policies face the same market paths and initial capital. Fees are one basis point of stock and ideal European/log instrument turnover. Entry costs and stock liquidation are charged; European claims settle without a closing fee. Monitoring is free, execution is immediate at the current synthetic market mark, and borrowing is unconstrained at zero interest. These assumptions limit practical conclusions.

## Held-out result

The training-selected fixed schedule was **{clock['frequency']} intervals/year**, with **{float(clock['mean_rebalances']):.0f}** intermediate rebalances, net RMSE **{float(clock['net_rmse']):.6f}**, and mean fees **{float(clock['mean_fees']):.6f}**.

The selected portfolio-band threshold was **{band['band']}** normalized gross-value units. It made about **{float(band['mean_rebalances']):.1f}** intermediate rebalances while checking **{float(band['mean_monitoring_checks']):.0f}** times, with net RMSE **{float(band['net_rmse']):.6f}** and mean fees **{float(band['mean_fees']):.6f}**: **{improvement:.1f}% lower RMSE and {fee_saving:.1f}% lower mean fees** than the training-selected clock rule.

The paired difference in net squared error (band minus clock) was **{float(pair['paired_mean_difference']):.3g}**, with approximate pointwise 95% interval **[{float(pair['ci_low']):.3g}, {float(pair['ci_high']):.3g}]**. This is conditional on the selected pilot design; it is not an estimate of all selection/model uncertainty.

The separate-block band had net RMSE **{float(separate['net_rmse']):.6f}** and fees **{float(separate['mean_fees']):.6f}**. It saved execution costs but worsened tracking relative to the selected fixed schedule. This supports investigating coordination between hedge legs; it does not yet establish the mechanism.

![Pilot cost/error comparison](adaptive_pilot.svg)

## Robustness checks and limitations

The same selected policies, without retuning, were tested on three additional 1,000-path scenarios: slower volatility switching, faster switching, and a three-state bounded volatility process. The portfolio band's paired score and net-MSE differences favored it in these pilot scenarios; the separate-block rule performed worse on those measures. These finite examples do not prove model-free robustness. European prices remain consistent with the true scenario law, so this is within-class transfer, not deliberate price-model misspecification.

The selected thresholds are at the smallest edge of the searched threshold grid (.05). A later training/validation stage should expand that range. Do not search the existing holdout for a new preferred threshold and keep treating it as a final test. A larger preregistered study needs fresh final-test seeds, cost-weight sensitivity, finer observation grids, tail-loss uncertainty and comparison at matched cost/error budgets. Savings relative to one selected grid point are not global optimality.

The pilot passed **{sum(c['passed'] for c in checks)}/{len(checks)}** saved-output checks. Original tests and nine reference tests also pass. Scripts and pathwise selected-policy outputs make the reported comparisons reproducible.

## Why this is not yet novel/publication-ready

Threshold rebalancing and no-transaction bands are established. A stronger candidate contribution would be a jump-exposure-aware execution rule or cost-aware selection among redundant jump-spanning collars, accompanied by a derivation/bound and held-out evidence for the CLL construction. The focused literature search cannot certify novelty. Review the plan's primary sources and ask a knowledgeable reviewer about overlap before making that claim.

## Reproduce

```powershell
python test_cll.py
python test_reference.py
python run_reference_audit.py
python run_adaptive.py
python validate_adaptive.py adaptive_pilot
python report_adaptive.py
```

Source and audit details: `review_packet/VERIFICATION_NOTE.md`. Research question and publication criteria: `research/ADAPTIVE_RESEARCH_PLAN.md`. No reviewer has been contacted automatically.
'''
    (ROOT/'NEXT_STAGE_RESULTS.md').write_text(note,encoding='utf-8')
    print('Wrote NEXT_STAGE_RESULTS.md and adaptive_pilot.svg')


if __name__=='__main__':
    main()
