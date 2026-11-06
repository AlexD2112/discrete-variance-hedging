"""Reproducible primary-vs-reference audit, including pathwise fees and prices."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import cll_hedge as primary
from reference_ledger import market_tape,execute
from test_reference import specification

ROOT=Path(__file__).parent


def main():
    folder=ROOT/'verification'; folder.mkdir(exist_ok=True)
    checks=[]
    for model_index,name in enumerate(('constant_vol','constant_vol_jumps','switching_vol','switching_vol_jumps')):
        for powers in ((2.,3.),(2.,5.),(3.,6.)):
            m=primary.Market(model=name,collar_powers=powers)
            spec=specification(m)
            paths=primary.simulate_paths(np.random.default_rng(77100+model_index),32,2016,m)
            for n in (12,252,2016):
                grid=paths.on_grid(n)
                for claim in ('variance','exponential'):
                    tape=market_tape(spec,grid.times,grid.log_s,grid.qv,grid.state,claim)
                    expected=primary.hedge(grid,m,claim)
                    for stock_bps,claim_bps in ((0.,0.),(1.,1.)):
                        actual=execute(tape,frequency=n,stock_bps=stock_bps,claim_bps=claim_bps,
                                       audit_path=name=='switching_vol_jumps' and powers==(2.,5.) and n==252 and claim=='variance')
                        wealth_expected=expected['wealth']-1e-4*(stock_bps*expected['stock_turnover']+claim_bps*expected['power_turnover'])
                        error=float(np.max(np.abs(actual['wealth']-wealth_expected)))
                        checks.append(dict(model=name,powers=powers,frequency=n,claim=claim,stock_bps=stock_bps,
                            claim_bps=claim_bps,max_pathwise_wealth_difference=error,
                            max_cash_accounting_residual=actual['max_accounting_error'],passed=error<1e-8 and actual['max_accounting_error']<1e-10))
                        if actual['ledger']:
                            target=folder/('worked_ledger_zero_fees.csv' if stock_bps==0 else 'worked_ledger_one_bp.csv')
                            with target.open('w',newline='') as f:
                                writer=csv.DictWriter(f,fieldnames=list(actual['ledger'][0])); writer.writeheader(); writer.writerows(actual['ledger'])
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('cll_hedge.py','reference_ledger.py','test_cll.py','test_reference.py','run_reference_audit.py')}
    result=dict(date='2026-11-02',checks=checks,source_sha256=hashes,
                caveat='Both implementations use the same specification.',
                max_wealth_difference=max(c['max_pathwise_wealth_difference'] for c in checks))
    (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not all(c['passed'] for c in checks):
        raise RuntimeError('cross-engine audit failed; inspect verification/audit.json')
    print(f'{len(checks)}/{len(checks)} pathwise cross-engine cases passed; maximum wealth difference {result["max_wealth_difference"]:.3g}')


if __name__=='__main__':
    main()
