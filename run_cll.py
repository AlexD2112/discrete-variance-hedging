"""Run paired CLL-versus-log-hedge experiments. See CLL_METHOD.md."""
import argparse
import csv
import json
from pathlib import Path
import platform
import numpy as np
from cll_hedge import Market, EuropeanMarket, simulate_paths, hedge
from simulate import summary


def save_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths',type=int,default=20000)
    parser.add_argument('--steps',type=int,default=2016)
    parser.add_argument('--seed',type=int,default=20261021)
    parser.add_argument('--batch-size',type=int,default=250)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'cll_results')
    parser.add_argument('--models',nargs='+',default=['constant_vol','switching_vol','switching_vol_jumps'])
    parser.add_argument('--frequencies',nargs='+',type=int,default=[12,24,84,252,504,1008,2016])
    parser.add_argument('--collar-powers',nargs=2,type=float,default=[2.,5.])
    args=parser.parse_args()
    if args.paths<2 or args.batch_size<1 or args.steps<=0 or args.steps%252 or any(n<=0 or args.steps%n for n in args.frequencies):
        parser.error('paths>=2, batch-size>=1, steps a positive multiple of 252, frequencies positive divisors of steps')
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]; pairs=[]; diagnostics=[]; model_metadata=[]
    for model_index,model in enumerate(args.models):
        m=Market(model=model,collar_powers=tuple(args.collar_powers))
        rng=np.random.default_rng(args.seed+model_index)
        samples={}; comparisons={}; diag={}
        for start in range(0,args.paths,args.batch_size):
            paths=simulate_paths(rng,min(args.batch_size,args.paths-start),args.steps,m)
            for key,data in [('terminal_stock',np.exp(paths.log_s[:,-1])),('qv',paths.qv[:,-1])]:
                diag.setdefault(key,[]).append(data)
            pricer=EuropeanMarket(m,paths.times[[0,-1]])
            endpoints=paths.on_grid(1)
            diag.setdefault('variance_payoff_minus_initial_price',[]).append(
                paths.qv[:,-1]-pricer.variance_value(endpoints)[:,0])
            diag.setdefault('exponential_payoff_minus_initial_price',[]).append(
                np.exp(-.1*paths.qv[:,-1])-pricer.exponential_value(endpoints,.1)[:,0])
            for power in (2.,5.):
                marks=np.exp(pricer.power_log_price(endpoints,power))
                diag.setdefault('power_'+str(power)+'_terminal_minus_initial_price',[]).append(marks[:,1]-marks[:,0])
            daily=paths.on_grid(252)
            rv=np.square(np.diff(daily.log_s,axis=1)).sum(axis=1)
            for n in args.frequencies:
                grid=paths.on_grid(n)
                outcomes={}
                for claim,strategy in [('variance','log_only'),('variance','cll'),('exponential','cll')]:
                    result=hedge(grid,m,claim,strategy,ledger=start==0 and n==252)
                    if 'ledger' in result:
                        save_csv(args.out/f'ledger_{model}_{claim}_{strategy}.csv',result['ledger'])
                    targets=[('quadratic_variation',result['payoff']),('daily_realized_variance',rv)] if claim=='variance' else [('exp_minus_0.1_qv',result['payoff'])]
                    for stock_bps,power_bps in ((0.,0.),(1.,0.),(5.,0.),(1.,1.)):
                        fee=stock_bps*1e-4*result['stock_turnover']+power_bps*1e-4*result['power_turnover']
                        for target,payoff in targets:
                            error=result['wealth']-fee-payoff
                            key=(n,claim,strategy,target,stock_bps,power_bps)
                            samples.setdefault(key,[]).append(error)
                            outcomes[(claim,strategy,target,stock_bps,power_bps)]=error
                    if n==252:
                        for key,data in [('initial_'+claim+'_'+strategy,result['initial_price']),
                                         ('stock_turnover_'+claim+'_'+strategy,result['stock_turnover']),
                                         ('power_turnover_'+claim+'_'+strategy,result['power_turnover'])]:
                            diag.setdefault(key,[]).append(data)
                for stock_bps,power_bps in ((0.,0.),(1.,0.),(5.,0.),(1.,1.)):
                    base=outcomes[('variance','log_only','quadratic_variation',stock_bps,power_bps)]
                    cll=outcomes[('variance','cll','quadratic_variation',stock_bps,power_bps)]
                    # Paired MSE difference is the comparison estimand. Negative favors CLL.
                    for metric,data in [('squared_error_difference',cll*cll-base*base),
                                        ('error_difference',cll-base)]:
                        comparisons.setdefault((n,stock_bps,power_bps,metric),[]).append(data)
            if start==0 or (start+args.batch_size)%5000==0:
                print(model,min(start+args.batch_size,args.paths),'/',args.paths,flush=True)
        for key,chunks in samples.items():
            n,claim,strategy,target,stock_bps,power_bps=key
            rows.append(dict(model=model,frequency=n,claim=claim,strategy=strategy,target=target,
                             stock_cost_bps=stock_bps,power_cost_bps=power_bps,paths=args.paths,
                             **summary(np.concatenate(chunks))))
        for key,chunks in comparisons.items():
            n,stock_bps,power_bps,metric=key
            pairs.append(dict(model=model,frequency=n,stock_cost_bps=stock_bps,power_cost_bps=power_bps,
                              metric=metric,**summary(np.concatenate(chunks))))
        for key,chunks in diag.items():
            diagnostics.append(dict(model=model,quantity=key,**summary(np.concatenate(chunks))))
        model_metadata.append(dict(model=model,seed=args.seed+model_index,
            variance_collars=m.variance_collars().tolist(),exponential_collars=m.exponential_collars(.1).tolist(),
            jump_matrix=m.jump_matrix().tolist(),matrix_condition_number=float(np.linalg.cond(m.jump_matrix())),
            jump_rates=m.rates.tolist(),jump_sizes=m.jump_sizes,
            variance_states=m.variance_states,collar_powers=m.collar_powers))
        save_csv(args.out/'summary.csv',rows)
        save_csv(args.out/'paired_comparisons.csv',pairs)
        save_csv(args.out/'diagnostics.csv',diagnostics)
    metadata={**vars(args),'out':args.out.name,'python':platform.python_version(),'numpy':np.__version__,
        'models_detail':model_metadata,'claim_a':.1,'normalized_S0':1,'maturity':1,
        'source':'Carr-Lee-Lorig, Theorem 4.4 and Corollary 4.8; variance from analytic derivative',
        'source_url':'https://arxiv.org/html/2107.00554','source_access_date':'2026-10-19',
        'sign_convention':'Theorem 4.4 jump balance: G k = -F; see CLL_METHOD.md',
        'fee_convention':'stock: entry/rebalance/liquidation; power/log: entry/rebalance, no settlement fee',
        'funding':'same conditional fair initial variance capital for log_only and cll; no wealth reset',
        'power_fees':'ideal power/log claim market-value turnover proxy, not finite vanilla strip costs'}
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Saved',str(args.out),flush=True)


if __name__=='__main__':
    main()
