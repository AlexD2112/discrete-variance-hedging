"""Carr-Lee-Lorig Section 4, real-power implementation with ideal European claims.

Stock is normalized to S/S0, so price starts at 1. Claims pay S_T**p.
Variance is the analytic -d/da limit of the exp(-a [log S]_T) strategy.
See CLL_METHOD.md for equations, source/sign convention, and instrument limits.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Market:
    model: str = 'switching_vol_jumps'
    jump_sizes: tuple = (-.15, .10)
    jump_rates: tuple = (1.4, .6)
    variance_states: tuple = (.01, .07)
    switch_probability: float = .02
    collar_powers: tuple = (2., 5.)

    def __post_init__(self):
        if self.model not in ('constant_vol', 'constant_vol_jumps', 'switching_vol', 'switching_vol_jumps'):
            raise ValueError('unknown market model')
        if len(self.jump_sizes)!=2 or len(self.jump_rates)!=2 or len(self.collar_powers)!=2:
            raise ValueError('this implementation requires two jump sizes and two collars')
        if any(x<0 for x in self.jump_rates) or not 0<=self.switch_probability<=1:
            raise ValueError('invalid intensity or transition probability')
        if len(self.variance_states)!=2 or any(v<0 for v in self.variance_states):
            raise ValueError('two nonnegative variance states required')

    @property
    def rates(self):
        return np.asarray(self.jump_rates) if self.model.endswith('_jumps') else np.zeros(2)

    def psi(self, power):
        p=np.asarray(power)[..., None]
        z=np.asarray(self.jump_sizes)
        return np.sum(self.rates*(np.expm1(p*z)-p*np.expm1(z)), axis=-1)

    @property
    def jump_log_drift(self):
        z=np.asarray(self.jump_sizes)
        return float(np.sum(self.rates*(z-np.expm1(z))))

    @property
    def jump_qv_rate(self):
        return float(np.sum(self.rates*np.square(self.jump_sizes)))

    def target_jump_exponent(self, a):
        return float(np.sum(self.rates*np.expm1(-a*np.square(self.jump_sizes))))

    def jump_matrix(self):
        z=np.asarray(self.jump_sizes)[:,None]
        p=np.asarray(self.collar_powers)[None,:]
        return -np.expm1(p*z)+np.expm1((1-p)*z)+(2*p-1)*np.expm1(z)

    def solve_collars(self, rhs):
        if not np.any(self.rates):
            return np.zeros(2)
        matrix=self.jump_matrix()
        if np.linalg.matrix_rank(matrix)<2 or np.linalg.cond(matrix)>1e10:
            raise ValueError('singular or excessively ill-conditioned jump-spanning matrix')
        return np.linalg.solve(matrix, rhs)

    def variance_collars(self):
        z=np.asarray(self.jump_sizes)
        return self.solve_collars(2*(np.expm1(z)-z)-z*z)

    def exponential_collars(self, a):
        p=laplace_power(a)
        z=np.asarray(self.jump_sizes)
        f=np.expm1(-a*z*z)-np.expm1(p*z)+p*np.expm1(z)
        # Theorem 4.4's jump balance: F + G*k = 0.
        return self.solve_collars(-f)


def laplace_power(a):
    if not 0<a<.125:
        raise ValueError('real-power branch requires 0 < a < 1/8')
    # Stable evaluation of .5-sqrt(.25-2a).
    return 2*a/(.5+np.sqrt(.25-2*a))


@dataclass
class Paths:
    times: np.ndarray
    log_s: np.ndarray
    qv: np.ndarray
    state: np.ndarray

    def on_grid(self, frequency):
        steps=len(self.times)-1
        if frequency<=0 or steps%frequency:
            raise ValueError('hedge frequency must divide simulation steps')
        stride=steps//frequency
        return Paths(self.times[::stride],self.log_s[:,::stride],self.qv[:,::stride],self.state[:,::stride])


def simulate_paths(rng, count, steps, market):
    if steps<=0 or steps%252:
        raise ValueError('simulation steps must be a positive multiple of 252')
    dt=1/steps
    if market.model.startswith('constant_vol'):
        state=np.zeros((count,steps),dtype=np.int8)
        variance=np.full((count,steps),.04)
    else:
        flips=rng.random((count,252))<market.switch_probability
        flips[:,0]=rng.random(count)<.5
        daily=np.logical_xor.accumulate(flips,axis=1)
        state=np.repeat(daily.astype(np.int8),steps//252,axis=1)
        variance=np.asarray(market.variance_states)[state]
    dx=-.5*variance*dt+np.sqrt(variance*dt)*rng.standard_normal((count,steps))
    dq=variance*dt
    z=np.asarray(market.jump_sizes)
    for size,rate in zip(z,market.rates):
        if rate:
            counts=rng.poisson(rate*dt,(count,steps))
            dx += size*counts-rate*np.expm1(size)*dt
            dq += size*size*counts
    return Paths(np.linspace(0,1,steps+1),
                 np.column_stack((np.zeros(count),np.cumsum(dx,axis=1))),
                 np.column_stack((np.zeros(count),np.cumsum(dq,axis=1))),
                 np.column_stack((state,state[:,-1])))


class EuropeanMarket:
    """Exact conditional marks under the simulator's independent daily regimes.

    This supplies market prices to the robust strategy. Only conditional current
    information is used; future simulated states/returns never enter marks.
    """
    def __init__(self, market, times):
        self.market=market
        self.times=np.asarray(times)
        self.tau=1-self.times
        self.cache={}

    def remaining_iv(self, c=None):
        """E[remaining IV] if c=None, else E[exp(c remaining IV)], by state."""
        key='mean' if c is None else float(c)
        if key in self.cache:
            return self.cache[key]
        if self.market.model.startswith('constant_vol'):
            v=.04*self.tau
            result=np.repeat((v if c is None else np.exp(c*v))[:,None],2,axis=1)
        else:
            variances=np.asarray(self.market.variance_states)
            p=self.market.switch_probability
            transition=np.array([[1-p,p],[p,1-p]])
            daily=np.zeros((253,2)) if c is None else np.ones((253,2))
            for d in range(251,-1,-1):
                future=daily[d+1] if d==251 else transition@daily[d+1]
                daily[d]=variances/252+future if c is None else np.exp(c*variances/252)*future
            days=np.minimum(np.floor(self.times*252+1e-10).astype(int),251)
            remaining=np.maximum((days+1)/252-self.times,0)
            future=daily[days+1]@transition.T
            future[days==251]=daily[252]
            result=(remaining[:,None]*variances+future if c is None
                    else np.exp(c*remaining[:,None]*variances)*future)
            result[self.times>=1]=0 if c is None else 1
        self.cache[key]=result
        return result

    @staticmethod
    def select(by_state, state):
        return by_state[np.arange(state.shape[1])[None,:],state]

    def mean_log_price(self, paths):
        return (paths.log_s-.5*self.select(self.remaining_iv(),paths.state)
                +self.tau*self.market.jump_log_drift)

    def variance_value(self, paths):
        return (paths.qv+self.select(self.remaining_iv(),paths.state)
                +self.tau*self.market.jump_qv_rate)

    def exponential_value(self, paths, a):
        return (np.exp(-a*paths.qv+self.tau*self.market.target_jump_exponent(a))
                *self.select(self.remaining_iv(-a),paths.state))

    def power_log_price(self, paths, power):
        mgf=self.select(self.remaining_iv(.5*power*(power-1)),paths.state)
        return power*paths.log_s+self.tau*self.market.psi(power)+np.log(mgf)

    def power_ratio(self, paths, power):
        return np.exp(np.diff(self.power_log_price(paths,power),axis=1))


def hedge(paths, market, claim='variance', strategy='cll', a=.1, ledger=False):
    """Return terminal wealth, stock/power turnover, payoff, and optional cash ledger.

    Wealth accumulates realized instrument gains from left-end holdings. It is
    never reset to the theoretical target value at rebalances. Both strategies
    start with the same conditional fair claim price. Fees are applied by runner.
    """
    if claim not in ('variance','exponential') or strategy not in ('cll','log_only'):
        raise ValueError('unsupported claim or strategy')
    if claim=='exponential' and strategy!='cll':
        raise ValueError('log-only comparison is defined only for variance')
    pricer=EuropeanMarket(market,paths.times)
    stock=np.exp(paths.log_s)
    stock_return=np.expm1(np.diff(paths.log_s,axis=1))
    collars=(market.variance_collars() if claim=='variance' else market.exponential_collars(a))
    if strategy=='log_only':
        collars=np.zeros(2)
    powers=np.asarray(market.collar_powers)
    if claim=='variance':
        value=pricer.variance_value(paths)
        scale=np.ones_like(value)
        log_mark=pricer.mean_log_price(paths)
        base_gains=-2*np.diff(log_mark,axis=1)
        base_values=-2*log_mark
        base_turnover=np.abs(base_values[:,0])
        stock_coefficient=2+float(collars@(1-2*powers))
        payoff=paths.qv[:,-1]
    else:
        value=pricer.exponential_value(paths,a)
        scale=value
        p=laplace_power(a)
        base_ratio=pricer.power_ratio(paths,p)
        base_gains=value[:,:-1]*(base_ratio-1)
        base_values=value
        base_turnover=(np.abs(value[:,0])+
                       np.abs(value[:,1:-1]-value[:,:-2]*base_ratio[:,:-1]).sum(axis=1))
        stock_coefficient=-p+float(collars@(1-2*powers))
        payoff=np.exp(-a*paths.qv[:,-1])
    dollar_stock=stock_coefficient*scale
    gains=base_gains+dollar_stock[:,:-1]*stock_return
    power_turnover=base_turnover.copy()
    for power,k in zip(powers,collars):
        if k==0:
            continue
        for leg_power,sign in ((power,1),(1-power,-1)):
            ratio=pricer.power_ratio(paths,leg_power)
            gains+=sign*k*scale[:,:-1]*(ratio-1)
            power_turnover+=abs(k)*(np.abs(scale[:,0])+
                np.abs(scale[:,1:-1]-scale[:,:-2]*ratio[:,:-1]).sum(axis=1))
    # Initial purchase, changes at t1...t_(n-1), terminal stock liquidation.
    stock_turnover=(np.abs(dollar_stock[:,0])+
        np.abs(dollar_stock[:,1:-1]-dollar_stock[:,:-2]*(1+stock_return[:,:-1])).sum(axis=1)+
        np.abs(dollar_stock[:,-2]*(1+stock_return[:,-1])))
    wealth_path=value[:,0,None]+np.column_stack((np.zeros(len(stock)),np.cumsum(gains,axis=1)))
    result=dict(wealth=wealth_path[:,-1],payoff=payoff,initial_price=value[:,0],
                stock_turnover=stock_turnover,power_turnover=power_turnover)
    if ledger:
        # Representative path, immediately after each scheduled rebalance.
        # At T the stock is liquidated and all claims settle into cash.
        s=stock[0]; mark=base_values[0]+dollar_stock[0]
        cash=wealth_path[0]-mark
        records=[]
        if claim=='variance':
            base_marks=log_mark[0]
            base_units=np.full(len(paths.times),-2.)
        else:
            base_marks=np.exp(pricer.power_log_price(paths,p)[0])
            base_units=value[0]/base_marks
        for i,t in enumerate(paths.times):
            terminal=i==len(paths.times)-1
            record=dict(time=float(t),stock=float(s[i]),observed_qv=float(paths.qv[0,i]),
                target_value=float(value[0,i]),wealth=float(wealth_path[0,i]),
                stock_units=0. if terminal else float(dollar_stock[0,i]/s[i]),
                base_claim_value=0. if terminal else float(base_values[0,i]),
                base_claim_units=0. if terminal else float(base_units[i]),
                base_claim_mark=float(base_marks[i]),
                cash=float(wealth_path[0,i] if terminal else cash[i]),
                collar_net_value=0.,external_cash_flow=0.,
                tracking_error=float(wealth_path[0,i]-value[0,i]))
            # Equal-valued opposite collar legs cancel at every rebalance.
            for j,(power,k) in enumerate(zip(powers,collars),start=1):
                for leg_power,sign,name in ((power,1,'long'),(1-power,-1,'short')):
                    log_price=pricer.power_log_price(paths,leg_power)[0,i]
                    record[f'collar_{j}_{name}_units']=(0. if terminal else float(sign*k*scale[0,i]*np.exp(-log_price)))
                    record[f'collar_{j}_{name}_mark']=float(np.exp(log_price))
            records.append(record)
        result['ledger']=records
    return result
