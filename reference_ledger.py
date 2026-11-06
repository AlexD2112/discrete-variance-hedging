"""Separately expressed CLL holdings, conditional marks, and literal cash ledger.

No imports from cll_hedge.py or its pricing/gain helpers. Same author/process:
this is an implementation cross-check, not external independent peer review.
Stock is normalized to one; European claims settle automatically without fees.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Specification:
    jump_sizes: tuple = (-.15,.10)
    jump_rates: tuple = (1.4,.6)
    powers: tuple = (2.,5.)
    variances: tuple = (.01,.07)
    transition: tuple = ((.98,.02),(.02,.98))
    constant_variance: object = None

    def __post_init__(self):
        p=np.asarray(self.transition,float)
        if p.shape!=(len(self.variances),len(self.variances)) or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
            raise ValueError('invalid transition matrix')
        if len(self.powers)!=2 or len(self.jump_sizes)!=2 or len(self.jump_rates)!=2:
            raise ValueError('two jump supports and two collar powers required')

    def coefficients(self, claim='variance', a=.1):
        if not 0<a<.125:
            raise ValueError('real exponential branch requires 0<a<1/8')
        z=np.asarray(self.jump_sizes); p=np.asarray(self.powers)
        # Construct each row explicitly rather than calling the primary matrix helper.
        matrix=np.array([[1-np.exp(q*j)+q*np.expm1(j)
                         -(1-np.exp((1-q)*j)+(1-q)*np.expm1(j)) for q in p] for j in z])
        base_power=.5-np.sqrt(.25-2*a)
        if claim=='variance':
            rhs=np.array([2*(np.exp(j)-1-j)-j*j for j in z])
        elif claim=='exponential':
            rhs=np.array([np.exp(base_power*j)-np.exp(-a*j*j)-base_power*np.expm1(j) for j in z])
        else:
            raise ValueError('unknown claim')
        if not any(self.jump_rates):
            k=np.zeros(2)
        else:
            if np.linalg.cond(matrix)>1e10:
                raise ValueError('ill-conditioned collar system')
            k=np.linalg.solve(matrix,rhs)
        return k,base_power

    def jump_moment(self,p):
        return sum(rate*(np.exp(p*z)-1-p*(np.exp(z)-1)) for rate,z in zip(self.jump_rates,self.jump_sizes))


def conditional_variance_functions(spec,times,c=None):
    """Expected future IV or its exponential moment, for every current state.

    Recursion is indexed by number of FULL future days, followed by a partial
    current-day factor. Independent of any realized future state sequence.
    """
    times=np.asarray(times)
    if spec.constant_variance is not None:
        iv=spec.constant_variance*(1-times)
        out=iv if c is None else np.exp(c*iv)
        return np.repeat(out[:,None],len(spec.variances),axis=1)
    v=np.asarray(spec.variances); transition=np.asarray(spec.transition)
    days=np.zeros((253,len(v))) if c is None else np.ones((253,len(v)))
    for n in range(1,253):
        if c is None:
            days[n]=v/252+transition@days[n-1]
        else:
            days[n]=np.exp(c*v/252)*(transition@days[n-1])
    out=[]
    for t in times:
        if t>=1:
            out.append(np.zeros(len(v)) if c is None else np.ones(len(v)))
            continue
        current_day=min(int(np.floor(252*t+1e-10)),251)
        fraction=(current_day+1)/252-t
        after=transition@days[251-current_day]
        out.append(fraction*v+after if c is None else np.exp(c*fraction*v)*after)
    return np.array(out)


def market_tape(spec,times,log_stock,observed_qv,states,claim='variance',a=.1):
    """Return explicit prices, target holdings/value, and payoff on recorded paths."""
    times=np.asarray(times); x=np.asarray(log_stock); q=np.asarray(observed_qv); states=np.asarray(states)
    if x.shape!=q.shape or x.shape!=states.shape or x.shape[1]!=len(times):
        raise ValueError('inconsistent path arrays')
    def select(table):
        return table[np.arange(len(times))[None,:],states]
    future=select(conditional_variance_functions(spec,times))
    tau=1-times
    drift=sum(rate*(z-np.expm1(z)) for rate,z in zip(spec.jump_rates,spec.jump_sizes))
    jump_variance=sum(rate*z*z for rate,z in zip(spec.jump_rates,spec.jump_sizes))
    k,p0=spec.coefficients(claim,a)
    powers=np.array([spec.powers[0],1-spec.powers[0],spec.powers[1],1-spec.powers[1]])
    def european_power(p):
        moment=select(conditional_variance_functions(spec,times,p*(p-1)/2))
        return np.exp(p*x+tau*spec.jump_moment(p))*moment
    stock=np.exp(x)
    if claim=='variance':
        base=x-future/2+tau*drift
        target=q+future+tau*jump_variance
        base_units=np.full_like(x,-2.)
        scale=np.ones_like(x)
        stock_dollars=2+sum(k[j]*(1-2*spec.powers[j]) for j in range(2))
        payoff=q[:,-1]
    else:
        base=european_power(p0)
        chi=sum(rate*(np.exp(-a*z*z)-1) for rate,z in zip(spec.jump_rates,spec.jump_sizes))
        target=np.exp(-a*q+tau*chi)*select(conditional_variance_functions(spec,times,-a))
        base_units=target/base
        scale=target
        stock_dollars=-p0+sum(k[j]*(1-2*spec.powers[j]) for j in range(2))
        payoff=np.exp(-a*q[:,-1])
    prices=np.stack([stock,base]+[european_power(p) for p in powers],axis=2)
    holdings=np.empty_like(prices)
    holdings[:,:,0]=stock_dollars*scale/stock
    holdings[:,:,1]=base_units
    for index,weight in enumerate((k[0],-k[0],k[1],-k[1]),start=2):
        holdings[:,:,index]=weight*scale/prices[:,:,index]
    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(holdings)):
        raise FloatingPointError('nonfinite instrument prices/holdings')
    return dict(times=times,prices=prices,target_holdings=holdings,target_value=target,payoff=payoff)


def simulate_market(rng,count,steps,spec):
    """Independent path generator; daily Markov volatility and Poisson counts."""
    if steps<=0 or steps%252:
        raise ValueError('steps must be a positive multiple of 252')
    state=np.empty((count,252),dtype=int)
    state[:,0]=rng.integers(0,len(spec.variances),count)
    transition=np.asarray(spec.transition)
    for day in range(1,252):
        cumulative=np.cumsum(transition[state[:,day-1]],axis=1)
        state[:,day]=(rng.random(count)[:,None]>cumulative).sum(axis=1)
    grid_states=np.repeat(state,steps//252,axis=1)
    variance=(np.full((count,steps),spec.constant_variance) if spec.constant_variance is not None
              else np.asarray(spec.variances)[grid_states])
    dx=np.sqrt(variance/steps)*rng.standard_normal((count,steps))-.5*variance/steps
    dq=variance/steps
    for z,rate in zip(spec.jump_sizes,spec.jump_rates):
        numbers=rng.poisson(rate/steps,(count,steps))
        dx+=z*numbers-rate*np.expm1(z)/steps
        dq+=z*z*numbers
    zeros=np.zeros((count,1))
    return dict(times=np.linspace(0,1,steps+1),log_stock=np.concatenate([zeros,np.cumsum(dx,axis=1)],axis=1),
                observed_qv=np.concatenate([zeros,np.cumsum(dq,axis=1)],axis=1),
                states=np.column_stack([grid_states,grid_states[:,-1]]))


def execute(tape,policy='periodic',frequency=252,band=.1,monitor_stride=1,
            stock_bps=0.,claim_bps=0.,audit_path=False):
    """Literal marked holdings + cash account, with costs debited at execution.

    Policies: periodic, exposure_band (all assets), separate_band (stock and
    European block separately). Signals use current marks, never future prices.
    Band = gross current-dollar distance to target holdings, normalized S0=1.
    """
    prices=tape['prices']; desired=tape['target_holdings']; count,nodes,assets=prices.shape
    steps=nodes-1
    if monitor_stride<1 or steps%monitor_stride or band<0:
        raise ValueError('invalid monitoring stride/band')
    if policy=='periodic' and (frequency<=0 or steps%frequency or (steps//frequency)%monitor_stride):
        raise ValueError('periodic frequency must divide monitoring grid')
    if policy not in ('periodic','exposure_band','separate_band'):
        raise ValueError('unknown policy')
    held=np.zeros((count,assets)); cash=tape['target_value'][:,0].copy()
    rate=np.array([stock_bps]+[claim_bps]*(assets-1))*1e-4
    fees=np.zeros(count); stock_turnover=np.zeros(count); claim_turnover=np.zeros(count)
    decisions=np.zeros(count); stock_events=np.zeros(count); claim_events=np.zeros(count)
    monitoring_events=np.zeros(count); ledger=[]
    max_accounting_error=0.
    trade_mask=np.zeros((count,nodes),dtype=bool) if audit_path else None
    for i in range(steps):
        mark=prices[:,i,:]
        before=cash+np.sum(held*mark,axis=1)
        delta=desired[:,i,:]-held
        monitor=i%monitor_stride==0
        update=np.zeros_like(held,dtype=bool)
        if i==0:
            update[:]=True
        elif monitor:
            monitoring_events+=1
            if policy=='periodic':
                update[:]=i%(steps//frequency)==0
            elif policy=='exposure_band':
                update[:]=(np.sum(np.abs(delta*mark),axis=1)>band)[:,None]
            else:
                update[:,0]=np.abs(delta[:,0]*mark[:,0])>band
                update[:,1:]=(np.sum(np.abs(delta[:,1:]*mark[:,1:]),axis=1)>band)[:,None]
        if i>0:
            actual=np.abs(np.where(update,delta,0))>1e-13
            decisions+=np.any(actual,axis=1)
            stock_events+=actual[:,0]
            claim_events+=np.any(actual[:,1:],axis=1)
        change=np.where(update,delta,0)
        notional=np.abs(change*mark)
        charge=np.sum(notional*rate,axis=1)
        cash-=np.sum(change*mark,axis=1)+charge
        held+=change
        fees+=charge; stock_turnover+=notional[:,0]; claim_turnover+=notional[:,1:].sum(axis=1)
        after=cash+np.sum(held*mark,axis=1)
        max_accounting_error=max(max_accounting_error,float(np.max(np.abs(after-before+charge))))
        if audit_path:
            trade_mask[:,i]=np.any(np.abs(change)>1e-13,axis=1)
            ledger.append(dict(time=float(tape['times'][i]),wealth_before=float(before[0]),wealth_after=float(after[0]),
                cash=float(cash[0]),fee=float(charge[0]),monitor=bool(monitor),
                traded=bool(trade_mask[0,i]),target_value=float(tape['target_value'][0,i]),
                **{f'price_{j}':float(mark[0,j]) for j in range(assets)},
                **{f'units_{j}':float(held[0,j]) for j in range(assets)}))
    mark=prices[:,-1,:]
    liquidation=np.abs(held[:,0]*mark[:,0])
    charge=liquidation*rate[0]
    terminal=cash+np.sum(held*mark,axis=1)-charge
    fees+=charge; stock_turnover+=liquidation
    if audit_path:
        ledger.append(dict(time=1.,wealth_before=float(terminal[0]+charge[0]),wealth_after=float(terminal[0]),
            cash=float(terminal[0]),fee=float(charge[0]),monitor=False,traded=False,
            target_value=float(tape['target_value'][0,-1]),**{f'price_{j}':float(mark[0,j]) for j in range(assets)},
            **{f'units_{j}':0. for j in range(assets)}))
    return dict(wealth=terminal,payoff=tape['payoff'],error=terminal-tape['payoff'],fees=fees,
                stock_turnover=stock_turnover,claim_turnover=claim_turnover,
                rebalance_events=decisions,stock_events=stock_events,claim_events=claim_events,
                monitoring_events=monitoring_events,max_accounting_error=max_accounting_error,
                ledger=ledger,trade_mask=trade_mask)
