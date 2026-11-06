"""Independent economic and statistical checks of generated CSV output."""
import csv
import json
import math
from pathlib import Path
import sys

folder=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'results'
with (folder/'summary.csv').open() as f:
    rows=list(csv.DictReader(f))
with (folder/'diagnostics.csv').open() as f:
    diagnostics=list(csv.DictReader(f))
meta=json.loads((folder/'metadata.json').read_text())
checks=[]


def close(row, expected, label):
    error=abs(float(row['mean'])-expected)
    tolerance=max(5*float(row['mean_se']),1e-12)
    checks.append(dict(check=label,passed=error<tolerance,observed=float(row['mean']),expected=expected,tolerance=tolerance))


for row in diagnostics:
    if row['quantity']=='terminal_ratio':
        close(row,1.,row['model']+' martingale mean')
    if row['quantity']=='qv':
        expected=.04+(1.4*.15**2+.6*.10**2 if row['model']=='bounded_jumps' else 0.)
        close(row,expected,row['model']+' expected quadratic variation')
    if row['quantity']=='jump_correction':
        expected=meta['expected_jump_residual'] if row['model']=='bounded_jumps' else 0.
        close(row,expected,row['model']+' expected jump residual')

for row in rows:
    if row['target']=='qv_oracle_jump_residual_removed':
        close(row,0.,row['model']+' zero mean after oracle jump subtraction, n='+row['trades_per_year'])
    if row['model']=='constant_vol' and row['target']=='quadratic_variation' and float(row['cost_bps'])==0:
        n=int(row['trades_per_year']); v=.04/n
        expected=math.sqrt(4*n*(math.expm1(v)-v))
        checks.append(dict(check='GBM RMSE n='+str(n),passed=abs(float(row['rmse'])-expected)<5*float(row['rmse_se'])))

# Costs cannot improve the residual on any path: their sample means must decrease.
for model in {r['model'] for r in rows}:
    for n in meta['frequency_grid']:
        subset=sorted([r for r in rows if r['model']==model and int(r['trades_per_year'])==n and r['target']=='quadratic_variation'],key=lambda r:float(r['cost_bps']))
        means=[float(r['mean']) for r in subset]
        checks.append(dict(check=model+' fee monotonicity n='+str(n),passed=all(a>b for a,b in zip(means,means[1:]))))

(folder/'economic_checks.json').write_text(json.dumps(checks,indent=2))
failed=[c for c in checks if not c['passed']]
print(f'{len(checks)-len(failed)}/{len(checks)} economic/statistical checks passed')
if failed:
    print(json.dumps(failed,indent=2))
    raise SystemExit(1)
