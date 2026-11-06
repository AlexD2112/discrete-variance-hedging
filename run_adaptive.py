"""Exploratory monitoring-versus-trading pilot with training/holdout separation.

Uses reference_ledger, not the primary gains engine. Fees are ideal-instrument
proxies. This is not an optimal-control solution or a publication-ready study.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import numpy as np
from reference_ledger import Specification,simulate_market,market_tape,execute

ROOT=Path(__file__).parent


def save(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def policy_grid():
    configs=[]
    for frequency in (12,84,252,504,1008,2016):
        configs.append(dict(name=f'periodic_{frequency}',policy='periodic',frequency=frequency,band=0.))
    for policy in ('exposure_band','separate_band'):
        for band in (.05,.1,.2,.4,.8):
            configs.append(dict(name=f'{policy}_{band:g}',policy=policy,frequency=0,band=band))
    return configs


def experiment(spec,count,seed,policies,batch_size,args,phase,monitor_stride=1):
    rng=np.random.default_rng(seed)
    collected={p['name']:[] for p in policies}
    for start in range(0,count,batch_size):
        sample=simulate_market(rng,min(batch_size,count-start),args.steps,spec)
        tape=market_tape(spec,**sample)
        for config in policies:
            result=execute(tape,policy=config['policy'],frequency=config['frequency'],band=config['band'],
                           monitor_stride=monitor_stride,stock_bps=args.stock_bps,claim_bps=args.claim_bps)
            if result['max_accounting_error']>1e-10:
                raise RuntimeError('cash ledger did not balance')
            pre_cost=result['error']+result['fees']
            # A fixed economic scale, ex ante for this scenario. Uniform initial regimes.
            scale=(spec.constant_variance if spec.constant_variance is not None else float(np.mean(spec.variances)))
            scale+=sum(rate*z*z for rate,z in zip(spec.jump_rates,spec.jump_sizes))
            score=pre_cost**2/scale**2+args.cost_weight*result['fees']/scale
            collected[config['name']].append(np.column_stack([result['error'],result['fees'],
                result['rebalance_events'],result['stock_events'],result['claim_events'],
                result['stock_turnover'],result['claim_turnover'],score,result['monitoring_events']]))
        if start==0 or (start+batch_size)%1000==0:
            print(phase,min(start+batch_size,count),'/',count,flush=True)
    return {key:np.concatenate(parts) for key,parts in collected.items()}


def summarize(values,phase,config):
    error=values[:,0]; scores=values[:,7]; n=len(values)
    return dict(phase=phase,name=config['name'],policy=config['policy'],frequency=config['frequency'],band=config['band'],paths=n,
        error_mean=float(error.mean()),error_mean_se=float(error.std(ddof=1)/np.sqrt(n)),
        net_rmse=float(np.sqrt(np.mean(error**2))),frictionless_rmse=float(np.sqrt(np.mean((error+values[:,1])**2))),
        mean_fees=float(values[:,1].mean()),mean_rebalances=float(values[:,2].mean()),
        mean_stock_rebalances=float(values[:,3].mean()),mean_claim_rebalances=float(values[:,4].mean()),
        mean_stock_turnover=float(values[:,5].mean()),mean_claim_turnover=float(values[:,6].mean()),
        mean_score=float(scores.mean()),score_se=float(scores.std(ddof=1)/np.sqrt(n)),
        mean_monitoring_checks=float(values[:,8].mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-paths',type=int,default=1000)
    parser.add_argument('--test-paths',type=int,default=4000)
    parser.add_argument('--stress-paths',type=int,default=1000)
    parser.add_argument('--steps',type=int,default=2016)
    parser.add_argument('--batch-size',type=int,default=250)
    parser.add_argument('--stock-bps',type=float,default=1.)
    parser.add_argument('--claim-bps',type=float,default=1.)
    parser.add_argument('--cost-weight',type=float,default=.1)
    parser.add_argument('--out',type=Path,default=ROOT/'adaptive_pilot')
    args=parser.parse_args()
    if min(args.train_paths,args.test_paths,args.stress_paths)<2 or args.steps!=2016 or args.batch_size<1:
        parser.error('paths>=2, positive batch size; pilot grid currently fixed at 2016')
    started=time.time(); args.out.mkdir(exist_ok=True,parents=True)
    base=Specification(); configs=policy_grid(); summaries=[]; comparisons=[]
    train=experiment(base,args.train_paths,20261102,configs,args.batch_size,args,'training')
    training_rows=[summarize(train[c['name']],'training',c) for c in configs]
    summaries.extend(training_rows)
    selected={}
    for family in ('periodic','exposure_band','separate_band'):
        winner=min((r for r in training_rows if r['policy']==family),key=lambda r:r['mean_score'])
        selected[family]=next(c for c in configs if c['name']==winner['name'])
    (args.out/'locked_selection.json').write_text(json.dumps(dict(selection=selected,criterion=
        'mean(pre-cost error^2)/ex-ante mean QV^2 + cost_weight*mean(fees)/ex-ante mean QV',
        cost_weight=args.cost_weight,training_seed=20261102,holdout_seed=20261103),indent=2))
    # Policies locked before any holdout samples are generated. Grid rows on
    # holdout are descriptive; do not choose a different winner using them.
    scenarios=[('holdout',base,args.test_paths,20261103,configs,1),
        ('slow_volatility',Specification(transition=((.998,.002),(.002,.998))),args.stress_paths,20261104,list(selected.values()),1),
        ('fast_volatility',Specification(transition=((.90,.10),(.10,.90))),args.stress_paths,20261105,list(selected.values()),1),
        ('three_state_volatility',Specification(variances=(.005,.035,.08),
             transition=((.98,.01,.01),(.01,.98,.01),(.01,.01,.98))),args.stress_paths,20261106,list(selected.values()),1)]
    for phase,spec,count,seed,policies,stride in scenarios:
        results=experiment(spec,count,seed,policies,args.batch_size,args,phase,stride)
        np.savez_compressed(args.out/(phase+'_selected_paths.npz'),
                            **{family:results[config['name']] for family,config in selected.items()})
        summaries.extend(summarize(results[c['name']],phase,c) for c in policies)
        baseline=results[selected['periodic']['name']]
        for family in ('exposure_band','separate_band'):
            candidate=results[selected[family]['name']]
            for metric,diff in [('score',candidate[:,7]-baseline[:,7]),
                                ('net_squared_error',candidate[:,0]**2-baseline[:,0]**2),
                                ('fees',candidate[:,1]-baseline[:,1])]:
                mean=float(diff.mean()); se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
                comparisons.append(dict(phase=phase,candidate=selected[family]['name'],baseline=selected['periodic']['name'],
                    metric=metric,paired_mean_difference=mean,mean_se=se,ci_low=mean-1.96*se,ci_high=mean+1.96*se))
        save(args.out/'summary.csv',summaries); save(args.out/'paired_comparisons.csv',comparisons)
    metadata={**vars(args),'out':str(args.out),'date':'2026-11-02','base_market':asdict(base),
              'scenario_markets':{phase:asdict(spec) for phase,spec,*_ in scenarios},
              'seeds':dict(training=20261102,**{phase:seed for phase,_,_,seed,_,_ in scenarios}),
              'seconds':time.time()-started,'numpy':np.__version__,'python':platform.python_version(),
              'pathwise_columns':['net_error','fees','rebalance_events','stock_events','claim_events',
                                  'stock_turnover','claim_turnover','risk_cost_score','monitoring_checks'],
              'limits':['same development process, no human review','ideal European instruments','zero monitoring cost',
                        'discrete observation grid, no exact hitting-time claim','heuristic bands, not optimal control',
                        'training-based selection; holdout grid rows exploratory; stress results not model-free proof']}
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Saved adaptive pilot to',args.out,flush=True)


if __name__=='__main__':
    main()
