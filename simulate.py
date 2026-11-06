"""Reproducible variance-hedge baseline; requires Python 3.10+ and numpy.
Run: python simulate.py --paths 20000 --fine-steps 2016 --out results
This implements the continuous-price log-contract hedge, NOT the Carr-Lee-Lorig jump hedge.
"""
import argparse
import csv
import json
from pathlib import Path
import platform
import numpy as np

FREQUENCIES = (12, 24, 84, 252, 504, 1008, 2016)


def summary(x):
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    rmse = float(np.sqrt(np.mean(x*x)))
    rmse_se = float(np.std(x*x, ddof=1)/(2*rmse*np.sqrt(len(x)))) if rmse else 0.
    return dict(mean=mean, mean_se=se, mean_ci_low=mean-1.96*se,
                mean_ci_high=mean+1.96*se, rmse=rmse, rmse_se=rmse_se,
                q01=float(np.quantile(x, .01)), q05=float(np.quantile(x, .05)),
                median=float(np.median(x)), q95=float(np.quantile(x, .95)),
                q99=float(np.quantile(x, .99)))


def paths_batch(rng, count, steps, model):
    dt = 1 / steps
    if model == 'constant_vol':
        sigma = np.full((count, steps), .2)
    else:
        # Independent, bounded volatility, fixed daily switching dates.
        # Initial state is stationary; daily transition probability is .02.
        switches = rng.random((count, 252)) < .02
        switches[:, 0] = rng.random(count) < .5
        state = np.logical_xor.accumulate(switches, axis=1)
        sigma = np.repeat(np.where(state, np.sqrt(.07), .10), steps//252, axis=1)
    integrated_variance = np.sum(sigma*sigma, axis=1)*dt
    dx = -.5*sigma*sigma*dt + sigma*np.sqrt(dt)*rng.standard_normal((count, steps))
    jump_qv = np.zeros(count)
    jump_correction = np.zeros(count)
    if model == 'bounded_jumps':
        # Two independent Poisson counts = intensity 2, J=-.15 (70%) or +.10 (30%).
        down = rng.poisson(1.4*dt, (count, steps))
        up = rng.poisson(.6*dt, (count, steps))
        compensation = 1.4*np.expm1(-.15) + .6*np.expm1(.10)
        dx += -.15*down + .10*up - compensation*dt
        nd, nu = down.sum(axis=1), up.sum(axis=1)
        jump_qv = .15**2*nd + .10**2*nu
        jump_correction = (2*(np.expm1(-.15)+.15)-.15**2)*nd + (2*(np.expm1(.10)-.10)-.10**2)*nu
    log_s = np.column_stack((np.zeros(count), np.cumsum(dx, axis=1)))
    return log_s, integrated_variance+jump_qv, jump_correction


def hedge(log_s, n, cost_bps):
    grid = log_s[:, ::((log_s.shape[1]-1)//n)]
    returns = np.diff(grid, axis=1)
    # Holding 2/S at each trading time gives gains 2*(S_next/S_now - 1).
    terminal_value = -2*log_s[:, -1] + 2*np.expm1(returns).sum(axis=1)
    # Proportional cost on underlying notional: initial trade + rebalances + liquidation.
    rebalance_notional = 2*np.abs(1-np.exp(returns[:, :-1])).sum(axis=1)
    liquidation_notional = 2*np.exp(returns[:, -1])
    fees = cost_bps*1e-4*(2+rebalance_notional+liquidation_notional)
    return terminal_value, fees


def checks():
    grid = np.array([[0., .1, -.05]])
    h, fees = hedge(grid, 2, 1.)
    assert np.allclose(h, .1+2*(np.expm1(.1)+np.expm1(-.15)))
    assert np.allclose(fees, 1e-4*(2+2*abs(1-np.exp(.1))+2*np.exp(-.15)))
    flat = np.zeros((3, 25))
    assert np.allclose(hedge(flat, 12, 0)[0], 0)
    assert np.allclose(hedge(flat, 12, 1)[1], .0004)
    # One jump, zero diffusion: exact pathwise jump residual.
    j = -.15
    jump = np.array([[0., j]])
    assert np.allclose(hedge(jump, 1, 0)[0]-j*j, 2*(np.expm1(j)-j)-j*j)


def write_svg(rows, out):
    models = ('constant_vol', 'switching_vol', 'bounded_jumps')
    colors = ('#2563eb', '#0f766e', '#b45309')
    selected = [r for r in rows if r['target']=='quadratic_variation' and r['cost_bps']==0]
    width, height = 850, 470
    left, right, top, bottom = 100, 810, 65, 365
    values = [r['rmse'] for r in selected]
    ymin, ymax = min(values)*.8, max(values)*1.2
    def xy(n, y):
        return (left+(np.log(n)-np.log(12))/(np.log(2016)-np.log(12))*(right-left),
                bottom-(np.log(y)-np.log(ymin))/(np.log(ymax)-np.log(ymin))*(bottom-top))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial" fill="#17202a">',
             '<text x="32" y="28" font-size="20">Variance hedge: terminal replication error</text>',
             '<text x="32" y="49" font-size="12">Ideal log contract; zero trading costs; one-year maturity; both axes logarithmic</text>']
    for n in FREQUENCIES:
        x,_=xy(n,ymin)
        parts += [f'<path d="M{x},{top} V{bottom}" stroke="#e5e7eb"/>', f'<text x="{x}" y="387" text-anchor="middle" font-size="11">{n}</text>']
    for y in np.geomspace(ymin, ymax, 5):
        _,py=xy(12,y)
        parts += [f'<path d="M{left},{py} H{right}" stroke="#e5e7eb"/>',f'<text x="90" y="{py+4}" text-anchor="end" font-size="11">{y:.4f}</text>']
    for model,color in zip(models,colors):
        series = [r for r in selected if r['model']==model]
        points=' '.join(f'{xy(r["trades_per_year"], r["rmse"])[0]},{xy(r["trades_per_year"], r["rmse"])[1]}' for r in series)
        parts.append(f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="2"/>')
        for r in series:
            x,y=xy(r['trades_per_year'],r['rmse'])
            parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        idx=models.index(model)
        parts.append(f'<text x="{110+idx*245}" y="437" fill="{color}" font-size="13">{model.replace("_", " ")}</text>')
    parts += ['<text x="450" y="412" text-anchor="middle" font-size="13">Underlying rebalances per year</text>',
              '<text transform="translate(22,235) rotate(-90)" text-anchor="middle" font-size="13">RMSE (integrated log-variance units)</text>',
              '<text x="32" y="460" font-size="11">Source: simulate.py, seeded synthetic Monte Carlo; jump curve uses uncorrected continuous-price hedge.</text>', '</g></svg>']
    (out/'convergence.svg').write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=int, default=20000)
    parser.add_argument('--fine-steps', type=int, default=2016)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results')
    args=parser.parse_args()
    if args.paths<2 or args.batch_size<1 or args.fine_steps%2016:
        parser.error('paths >=2, batch-size >=1, and fine-steps a positive multiple of 2016 required')
    if args.fine_steps<=0:
        parser.error('fine-steps must be positive')
    checks()
    args.out.mkdir(parents=True, exist_ok=True)
    rows=[]
    diagnostics=[]
    for model_index,model in enumerate(('constant_vol','switching_vol','bounded_jumps')):
        rng=np.random.default_rng(args.seed+model_index)
        samples={}
        diagnostic={key:[] for key in ('qv','terminal_ratio','jump_correction','daily_sampling_error')}
        for start in range(0,args.paths,args.batch_size):
            log_s,qv,correction=paths_batch(rng,min(args.batch_size,args.paths-start),args.fine_steps,model)
            daily_rv=np.square(np.diff(log_s[:,::(args.fine_steps//252)],axis=1)).sum(axis=1)
            for key,x in [('qv',qv),('terminal_ratio',np.exp(log_s[:,-1])),('jump_correction',correction),('daily_sampling_error',daily_rv-qv)]:
                diagnostic[key].append(x)
            for n in FREQUENCIES:
                for bps in (0.,1.,5.):
                    value,fees=hedge(log_s,n,bps)
                    for target,payoff in [('quadratic_variation',qv),('daily_realized_variance',daily_rv)]:
                        samples.setdefault((n,bps,target),[]).append(value-fees-payoff)
                # Oracle diagnostic only: removes observed jump residual, not a tradable jump hedge.
                value,_=hedge(log_s,n,0)
                samples.setdefault((n,0.,'qv_oracle_jump_residual_removed'),[]).append(value-qv-correction)
        for (n,bps,target),chunks in samples.items():
            rows.append(dict(model=model,trades_per_year=n,cost_bps=bps,target=target,paths=args.paths,**summary(np.concatenate(chunks))))
        for key,chunks in diagnostic.items():
            x=np.concatenate(chunks)
            diagnostics.append(dict(model=model,quantity=key,**summary(x)))
            print(model,key,json.dumps(summary(x)))
        print(model,'complete',flush=True)
    with (args.out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.out/'diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(diagnostics[0])); writer.writeheader(); writer.writerows(diagnostics)
    metadata={**vars(args),'out':args.out.name,'python':platform.python_version(),'numpy':np.__version__,
              'frequency_grid':FREQUENCIES,'horizon_years':1,'S0':100,
              'error_definition':'ideal log payoff + discrete stock gains - underlying fees - target payoff',
              'jump_intensity':2,'jump_sizes':[-.15,.10],'jump_probabilities':[.7,.3],
              'volatility_states':[.10,float(np.sqrt(.07))],'daily_volatility_switch_probability':.02,
              'expected_jump_residual':1.4*(2*(np.expm1(-.15)+.15)-.15**2)+.6*(2*(np.expm1(.10)-.10)-.10**2),
              'checks':'algebraic hedge, turnover, flat path, single-jump identity passed'}
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    write_svg(rows,args.out)
    validation=[]
    for row in rows:
        if row['model']=='constant_vol' and row['cost_bps']==0 and row['target']=='quadratic_variation':
            n=row['trades_per_year']; v=.04/n
            theoretical=float(np.sqrt(4*n*(np.expm1(v)-v)))
            passed=abs(row['rmse']-theoretical)<5*row['rmse_se'] and abs(row['mean'])<5*row['mean_se']
            validation.append(dict(check='GBM exact mean and RMSE',n=n,theoretical_rmse=theoretical,passed=bool(passed)))
    (args.out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    if not all(item['passed'] for item in validation):
        raise RuntimeError('Monte Carlo validation outside five standard errors; inspect validation.json')


if __name__=='__main__':
    main()
