"""Variance-rate diagnostics for our ideal CLL variance hedge.

These identities are NOT a Rosenbaum-Tankov optimal stopping/execution rule.
Between scheduled regime changes, Brownian and compensated finite Poisson
exposures give an instantaneous conditional error-variance rate. Regime changes
at fixed daily times have separate covariance atoms, not a per-year hazard.
"""
import json
from pathlib import Path
import numpy as np
from reference_ledger import Specification,market_tape


def target_units(spec,prices):
    y=np.asarray(prices,float)
    k,_=spec.coefficients('variance')
    powers=np.asarray(spec.powers)
    units=np.empty_like(y)
    units[...,0]=(2+np.dot(1-2*powers,k))/y[...,0]
    units[...,1]=-2
    units[...,2:]=np.array([k[0],-k[0],k[1],-k[1]])/y[...,2:]
    return units


def local_risk(spec,prices,held,variance):
    """Per-year conditional variance rate away from daily volatility boundaries.

    prices/held have last dimension [stock, log, p1,1-p1,p2,1-p2].
    Target QV-value Brownian exposure is zero; its jump is z^2.
    """
    y=np.asarray(prices,float); h=np.asarray(held,float)
    if y.shape!=h.shape or y.shape[-1]!=6 or variance<0:
        raise ValueError('matching six-asset vectors and nonnegative variance required')
    powers=np.array([spec.powers[0],1-spec.powers[0],spec.powers[1],1-spec.powers[1]])
    loading=np.concatenate([y[...,0,None],np.ones_like(y[...,0,None]),y[...,2:]*powers],axis=-1)*np.sqrt(variance)
    brownian=np.sum(h*loading,axis=-1)
    covariance=loading[..., :,None]*loading[...,None,:]
    jump_terms=[]; jump_vectors=[]
    for z,rate in zip(spec.jump_sizes,spec.jump_rates):
        jump=np.concatenate([y[...,0,None]*np.expm1(z),np.full_like(y[...,0,None],z),
                             y[...,2:]*np.expm1(powers*z)],axis=-1)
        residual=np.sum(h*jump,axis=-1)-z*z
        jump_terms.append(rate*residual*residual)
        jump_vectors.append(jump)
        covariance+=rate*jump[..., :,None]*jump[...,None,:]
    return dict(brownian_loading=brownian,brownian_variance_rate=brownian*brownian,
        jump_variance_rates=np.stack(jump_terms,axis=-1),
        total_variance_rate=brownian*brownian+sum(jump_terms),
        covariance_rate=covariance,jump_vectors=jump_vectors)


def boundary_risk(spec,time,log_stock,past_qv,previous_state,held=None):
    """Conditional variance AT a scheduled daily regime revelation.

    Keep stock/past QV fixed while possible new volatility states are repriced.
    The pre-boundary mark is the transition-probability-weighted post mark.
    The returned variance is an atom (payoff^2), not a variance rate.
    """
    if not 0<time<1 or abs(time*252-round(time*252))>1e-8:
        raise ValueError('use a daily volatility boundary strictly before maturity')
    probabilities=np.asarray(spec.transition)[previous_state]
    count=len(spec.variances)
    tape=market_tape(spec,np.array([time]),np.full((count,1),log_stock),
                     np.full((count,1),past_qv),np.arange(count)[:,None])
    prices=tape['prices'][:,0]; values=tape['target_value'][:,0]
    before=probabilities@prices; target_before=float(probabilities@values)
    price_jumps=prices-before; value_jumps=values-target_before
    if held is None:
        held=target_units(spec,before)
    errors=price_jumps@np.asarray(held)-value_jumps
    atom=np.einsum('i,ij,ik->jk',probabilities,price_jumps,price_jumps)
    return dict(prices_before=before,target_before=target_before,
                error_jumps=errors,variance_atom=float(probabilities@(errors*errors)),
                covariance_atom=atom,price_jumps=price_jumps,probabilities=probabilities,
                target_value_jumps=value_jumps)


def main():
    spec=Specification()
    initial=market_tape(spec,np.array([0.]),np.array([[0.]]),np.array([[0.]]),np.array([[0]]))
    moved=market_tape(spec,np.array([.001]),np.array([[np.log(1.05)]]),np.array([[.00001]]),np.array([[0]]))
    old=initial['target_holdings'][0,0]; y=moved['prices'][0,0]
    stale=local_risk(spec,y,old,.01); fresh=local_risk(spec,y,target_units(spec,y),.01)
    before_atom=boundary_risk(spec,1/252,0.,.01/252,0)
    # Deliberately stale holdings taken before a different stock-price snapshot.
    stale_atom=boundary_risk(spec,1/252,np.log(1.05),.01/252,0,old)
    initial_prices=initial['prices'][0,0]
    sigma=.1
    coefficient=np.array([-old[0]*sigma,0.,-2*old[2]*sigma,old[3]*sigma,
                           -5*old[4]*sigma,4*old[5]*sigma])
    report=dict(date='2026-11-02',stock_holding_brownian_coefficient=float(coefficient[0]),
        all_holdings_brownian_coefficients=coefficient.tolist(),
        finite_asset_jump_rate_per_year=float(sum(spec.jump_rates)),
        fresh_local_variance_rate=float(fresh['total_variance_rate']),
        illustrative_stale_local_variance_rate=float(stale['total_variance_rate']),
        daily_price_covariance_atom_trace=float(np.trace(before_atom['covariance_atom'])),
        fresh_daily_error_variance_atom=before_atom['variance_atom'],
        illustrative_stale_daily_error_variance_atom=stale_atom['variance_atom'],
        limits=['local identities, not an optimal trigger','example snapshot, not a simulated performance comparison',
                'rates exclude scheduled volatility atoms, which are reported separately',
                'risk measured under pricing-model probabilities, not calibrated physical probabilities'])
    folder=Path(__file__).parent/'risk_review'; folder.mkdir(exist_ok=True)
    (folder/'diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
