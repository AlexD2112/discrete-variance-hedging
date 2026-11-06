"""Check Monte Carlo outputs; deterministic identities live in test_cll.py."""
import csv
import json
from pathlib import Path
import sys

folder=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'cll_results'
with (folder/'summary.csv').open() as f:
    rows=list(csv.DictReader(f))
with (folder/'diagnostics.csv').open() as f:
    diagnostics=list(csv.DictReader(f))
checks=[]


def check(label,passed,**details):
    checks.append(dict(check=label,passed=bool(passed),**details))


for row in rows:
    if float(row['stock_cost_bps'])==0 and float(row['power_cost_bps'])==0:
        mean,se=float(row['mean']),float(row['mean_se'])
        # Daily RV has a different fair price; it is not an unbiased QV hedge target.
        if row['target']!='daily_realized_variance':
            check('zero-cost martingale residual '+row['model']+' '+row['claim']+' '+row['strategy']+' n='+row['frequency'],
                  abs(mean)<max(5*se,1e-11),mean=mean,se=se)

for row in diagnostics:
    label=row['quantity']; mean,se=float(row['mean']),float(row['mean_se'])
    if label=='terminal_stock':
        check(row['model']+' stock martingale',abs(mean-1)<max(5*se,1e-11))
    if label.endswith('_minus_initial_price'):
        check(row['model']+' fair-price check '+label,abs(mean)<max(5*se,1e-11),mean=mean,se=se)

groups={}
for row in rows:
    key=(row['model'],row['frequency'],row['claim'],row['strategy'],row['target'])
    groups.setdefault(key,[]).append(row)
for key,group in groups.items():
    by_cost={(float(r['stock_cost_bps']),float(r['power_cost_bps'])):float(r['mean']) for r in group}
    check('fee monotonicity '+' '.join(key),by_cost[(0.,0.)]>by_cost[(1.,0.)]>by_cost[(5.,0.)]
          and by_cost[(1.,0.)]>=by_cost[(1.,1.)])

for model in {r['model'] for r in rows}:
    if model.endswith('_jumps'):
        continue
    for n in {r['frequency'] for r in rows if r['model']==model}:
        subset=[r for r in rows if r['model']==model and r['frequency']==n and r['claim']=='variance'
                and r['target']=='quadratic_variation' and float(r['stock_cost_bps'])==0]
        check(model+' CLL reduces to log hedge without jumps n='+n,len(subset)==2
              and abs(float(subset[0]['rmse'])-float(subset[1]['rmse']))<1e-12)

# Finer-grid convergence is a diagnostic, not an assumed assertion that CLL
# must beat the baseline at every finite trading frequency.
for model in {r['model'] for r in rows}:
    for claim in ('variance','exponential'):
        subset=sorted([r for r in rows if r['model']==model and r['claim']==claim and r['strategy']=='cll'
                       and r['target']!='daily_realized_variance' and float(r['stock_cost_bps'])==0],key=lambda r:int(r['frequency']))
        if len(subset)>1:
            check(model+' '+claim+' finer trading improves endpoint RMSE',float(subset[-1]['rmse'])<float(subset[0]['rmse']))

(folder/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
failed=[c for c in checks if not c['passed']]
print(f'{len(checks)-len(failed)}/{len(checks)} CLL output checks passed')
if failed:
    print(json.dumps(failed,indent=2))
    raise SystemExit(1)
