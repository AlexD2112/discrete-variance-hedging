"""Check locked selection, paired comparisons, pricing means and cash-cost identities."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

folder=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'adaptive_pilot'
with (folder/'summary.csv').open() as f:
    rows=list(csv.DictReader(f))
with (folder/'paired_comparisons.csv').open() as f:
    pairs=list(csv.DictReader(f))
selection=json.loads((folder/'locked_selection.json').read_text())['selection']
checks=[]


def check(name,passed,**details):
    checks.append(dict(check=name,passed=bool(passed),**details))


for family,config in selection.items():
    candidates=[r for r in rows if r['phase']=='training' and r['policy']==family]
    winner=min(candidates,key=lambda r:float(r['mean_score']))
    check('training-only selection '+family,winner['name']==config['name'])

for phase in ('holdout','slow_volatility','fast_volatility','three_state_volatility'):
    with np.load(folder/(phase+'_selected_paths.npz')) as data:
        baseline=data['periodic']
        for family,config in selection.items():
            samples=data[family]
            row=next(r for r in rows if r['phase']==phase and r['name']==config['name'])
            check(phase+' '+family+' RMSE matches pathwise data',np.isclose(float(row['net_rmse']),np.sqrt(np.mean(samples[:,0]**2))))
            check(phase+' '+family+' fees match turnover',np.allclose(samples[:,1],1e-4*(samples[:,5]+samples[:,6])))
            # Current pilot costs are 1bp each. Prevent silently validating other settings.
            pre=samples[:,0]+samples[:,1]
            se=np.std(pre,ddof=1)/np.sqrt(len(pre))
            check(phase+' '+family+' zero-fee martingale mean',abs(pre.mean())<max(5*se,1e-10),mean=float(pre.mean()),se=float(se))
            check(phase+' '+family+' causal execution counts',np.all(samples[:,2]<=samples[:,8]))
            if family=='periodic':
                continue
            for metric,diff in [('score',samples[:,7]-baseline[:,7]),
                                ('net_squared_error',samples[:,0]**2-baseline[:,0]**2),
                                ('fees',samples[:,1]-baseline[:,1])]:
                pair=next(p for p in pairs if p['phase']==phase and p['candidate']==config['name'] and p['metric']==metric)
                check(phase+' '+family+' paired '+metric,np.isclose(float(pair['paired_mean_difference']),diff.mean(),rtol=1e-10,atol=1e-14))

meta=json.loads((folder/'metadata.json').read_text())
if meta['stock_bps']!=1 or meta['claim_bps']!=1:
    raise ValueError('this validator currently covers the documented 1bp/1bp pilot only')
(folder/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
failures=[c for c in checks if not c['passed']]
print(f'{len(checks)-len(failures)}/{len(checks)} adaptive pilot checks passed')
if failures:
    print(json.dumps(failures,indent=2))
    raise SystemExit(1)
