"""Regenerate the research note and standalone SVG plots from saved CLL runs."""
import csv
import html
import json
import math
from pathlib import Path
import numpy as np
from cll_hedge import Market

ROOT=Path(__file__).parent


def load(folder,name='summary.csv'):
    with (ROOT/folder/name).open() as f:
        return list(csv.DictReader(f))


def select(rows,n,strategy='cll',stock=0.,power=0.):
    return next(r for r in rows if r['model']=='switching_vol_jumps' and int(r['frequency'])==n
                and r['strategy']==strategy and r['target']=='quadratic_variation'
                and float(r['stock_cost_bps'])==stock and float(r['power_cost_bps'])==power)


def chart(rows,filename,title,caption,fees=(0.,0.),floor=None):
    series=[]
    for strategy,color,label in [('log_only','#475569','Fair-funded log hedge'),('cll','#0f766e','CLL variance hedge')]:
        data=sorted([r for r in rows if r['model']=='switching_vol_jumps' and r['strategy']==strategy
                     and r['target']=='quadratic_variation' and float(r['stock_cost_bps'])==fees[0]
                     and float(r['power_cost_bps'])==fees[1]],key=lambda r:int(r['frequency']))
        series.append((data,color,label))
    frequencies=sorted({int(r['frequency']) for data,_,_ in series for r in data})
    yvals=[float(r['rmse']) for data,_,_ in series for r in data]
    if floor:
        yvals.append(floor)
    ymin,ymax=min(yvals)*.7,max(yvals)*1.3
    left,right,top,bottom=100,830,80,365
    def xy(n,y):
        return (left+math.log(n/frequencies[0])/math.log(frequencies[-1]/frequencies[0])*(right-left),
                bottom-math.log(y/ymin)/math.log(ymax/ymin)*(bottom-top))
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="490" viewBox="0 0 900 490">',
           '<rect width="900" height="490" fill="white"/>','<g font-family="Arial" fill="#17202a">',
           f'<text x="32" y="30" font-size="20">{html.escape(title)}</text>',
           f'<text x="32" y="53" font-size="12">Switching volatility + two bounded jump sizes; stock fees {fees[0]:g} bp; power fees {fees[1]:g} bp</text>']
    for n in frequencies:
        x,_=xy(n,ymin)
        parts.extend([f'<path d="M{x},{top} V{bottom}" stroke="#e5e7eb"/>',
                      f'<text x="{x}" y="388" font-size="11" text-anchor="middle">{n:,}</text>'])
    for y in np.geomspace(ymin,ymax,6):
        _,py=xy(frequencies[0],float(y))
        parts.extend([f'<path d="M{left},{py} H{right}" stroke="#e5e7eb"/>',
                      f'<text x="90" y="{py+4}" font-size="11" text-anchor="end">{y:.4f}</text>'])
    if floor:
        _,y=xy(frequencies[0],floor)
        parts.extend([f'<path d="M{left},{y} H{right}" stroke="#94a3b8" stroke-dasharray="5 4"/>',
                      f'<text x="{left+10}" y="{y-6}" font-size="11">Log hedge continuous-trading RMSE limit</text>'])
    for data,color,label in series:
        points=' '.join(f'{xy(int(r["frequency"]),float(r["rmse"]))[0]},{xy(int(r["frequency"]),float(r["rmse"]))[1]}' for r in data)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for r in data:
            n=int(r['frequency']); rmse=float(r['rmse']); se=float(r['rmse_se'])
            x,y=xy(n,rmse)
            _,ylo=xy(n,max(rmse-1.96*se,ymin)); _,yhi=xy(n,rmse+1.96*se)
            parts.extend([f'<path d="M{x},{ylo} V{yhi}" stroke="{color}"/>',
                          f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>'])
        index=0 if label.startswith('Fair') else 1
        parts.append(f'<text x="{120+index*320}" y="439" font-size="13" fill="{color}">{label}</text>')
    parts.extend(['<text x="460" y="412" font-size="13" text-anchor="middle">Scheduled rebalances per year (log scale)</text>',
                  '<text transform="translate(24,235) rotate(-90)" font-size="13" text-anchor="middle">Terminal RMSE (integrated variance; log scale)</text>',
                  f'<text x="32" y="465" font-size="11">{html.escape(caption)}</text>',
                  '<text x="32" y="482" font-size="11">Error bars: pointwise approximate 95% Monte Carlo intervals for RMSE (delta method).</text>',
                  '</g></svg>'])
    (ROOT/filename).write_text('\n'.join(parts),encoding='utf-8')


def main():
    main_rows=load('cll_results'); fine=load('cll_fine_results'); second=load('cll_second_seed')
    m=Market(); z=np.asarray(m.jump_sizes); r=2*(np.expm1(z)-z)-z*z
    floor=float(np.sqrt(m.rates@(r*r)))
    chart(main_rows,'cll_comparison.svg','Discrete variance hedging: initial comparison',
          'Source: cll_results/summary.csv; 20,000 paths; jump-model seed 20261023; T = 1 year.',floor=floor)
    chart(fine,'cll_fine_comparison.svg','Finer trading separates the two variance hedges',
          'Source: cll_fine_results/summary.csv; 10,000 paths; seed 20261024; T = 1 year.',floor=floor)
    chart(fine,'cll_cost_comparison.svg','Trading costs can outweigh the jump-hedging gain',
          'Source: cll_fine_results/summary.csv; 10,000 paths; seed 20261024; T = 1 year.',fees=(1.,1.))
    def rm(rows,n,strategy='cll',stock=0.,power=0.):
        return float(select(rows,n,strategy,stock,power)['rmse'])
    comparisons=load('cll_results','paired_comparisons.csv')
    pair=next(r for r in comparisons if r['model']=='switching_vol_jumps' and r['frequency']=='2016'
              and r['stock_cost_bps']=='0.0' and r['metric']=='squared_error_difference')
    validates=[]
    for folder in ('cll_results','cll_second_seed','cll_fine_results'):
        checks=json.loads((ROOT/folder/'validation.json').read_text())
        validates.append(f"- {folder}: {sum(c['passed'] for c in checks)}/{len(checks)} output checks")
    note=f'''# CLL hedge: implementation and first paired results

October 19, 2026 · synthetic numerical experiments · not a novelty claim

## What is now implemented

The separate `cll_hedge.py` implements the real-power exponential claim from the paper's Section 4 and its analytically differentiated variance hedge. It trades stock, cash, an ideal log/base power claim, and two complementary European power pairs. Holdings are fixed between trading dates. Initial capital is the fair claim price, and the cash account finances every rebalance without resetting wealth to the theoretical price.

The original `simulate.py` is preserved. The new comparison uses identical fair initial capital for the CLL and log-only strategies. Its baseline numbers consequently differ from the original raw payoff residual by a deterministic jump-premium adjustment. See `CLL_METHOD.md` for the derivation, source references, sign discrepancy, and pricing assumptions.

## What the runs show

In the 20,000-path switching-volatility/jump sample, at 252 trades/year, zero-fee RMSE is **{rm(main_rows,252):.6f} for CLL**, compared with **{rm(main_rows,252,'log_only'):.6f} for the log hedge**. The extra hedge is worse on this trading schedule.

At 2,016 trades/year, zero-fee RMSE is **{rm(main_rows,2016):.6f} for CLL** and **{rm(main_rows,2016,'log_only'):.6f} for the log hedge**. The paired mean squared-error difference (CLL minus baseline) is **{float(pair['mean']):.3g}**, with a pointwise approximate 95% interval **[{float(pair['mean_ci_low']):.3g}, {float(pair['mean_ci_high']):.3g}]**. Negative favors CLL.

In the separate 10,000-path finer-grid run, at 16,128 trades/year, RMSE is **{rm(fine,16128):.6f} for CLL** and **{rm(fine,16128,'log_only'):.6f} for the baseline**, a **{100*(1-rm(fine,16128)/rm(fine,16128,'log_only')):.1f}%** reduction. The baseline's analytically known continuous-trading RMSE limit is **{floor:.6f}**. We do not claim the observed CLL curve has already reached its limiting regime or prove a convergence rate from these few grids.

The independent 20,000-path repeat gives CLL/baseline RMSEs of **{rm(second,252):.6f}/{rm(second,252,'log_only'):.6f}** at 252 trades/year and **{rm(second,2016):.6f}/{rm(second,2016,'log_only'):.6f}** at 2,016 trades/year. Each within-run comparison is paired. The fine-grid run reuses the repeat's jump-market seed with a different grid and batch size, so it is a grid sensitivity run rather than a third independent seed replication.

![Initial paired comparison](cll_comparison.svg)

![Finer-grid comparison](cll_fine_comparison.svg)

## Costs change the conclusion

In the finer-grid sample at 16,128 trades/year, applying a stylized one basis point to both stock and ideal power/log claim turnover raises CLL RMSE to **{rm(fine,16128,stock=1,power=1):.6f}**, versus **{rm(fine,16128,'log_only',stock=1,power=1):.6f}** for the simpler hedge. These ideal-claim fees are a sensitivity calculation, not the cost of an executable vanilla option strip. A zero-value collar can require substantial gross trading.

![Cost sensitivity](cll_cost_comparison.svg)

## Validation

Fourteen deterministic tests cover jump matching at both allowed sizes, the analytic derivative, power-price symmetry, the paper's pricing representation, conditional moment recursion, frozen-position self-financing, terminal values, causality, and the no-jump limit. Generated output validation:

{chr(10).join(validates)}

Normal/delta-method Monte Carlo intervals are approximate and pointwise, with no adjustment for testing multiple grids. Tail quantiles need larger samples and separate uncertainty estimates. The spanning matrix condition number is about 9,080 and is recorded in metadata; numerical identity tests pass despite that sensitivity.

## Next research work

1. Reproduce the derivation in `CLL_METHOD.md` independently and review the documented source sign convention with Professor Lorig before citing these results as a paper reproduction.
2. Replace the ideal European power/log instruments by explicitly funded finite-strike vanilla portfolios. Separate strike truncation, strike spacing, and transaction costs. Keep the ideal version as the control.
3. Test collar-power choices and parameter sensitivity on fresh samples. The finite-frequency crossover is specific to the selected support, powers, volatility process and fees; it is not a universal threshold.
4. Review related literature before claiming novelty, then write a short note around one clearly defined friction and an explanation of its error scaling.

This implements a specific CLL exponential strategy and variance limit under independent volatility and two-point jumps. It does not implement the barrier paper, arbitrary jump distributions, or a market-ready hedge.
'''
    (ROOT/'CLL_RESULTS.md').write_text(note,encoding='utf-8')
    print('Wrote CLL_RESULTS.md and three SVG figures')


if __name__=='__main__':
    main()
