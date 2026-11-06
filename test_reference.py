"""Cross-engine and adaptive execution tests. Reference engine shares no helpers."""
import unittest
import numpy as np
import cll_hedge as primary
from reference_ledger import Specification,market_tape,simulate_market,execute


def specification(m):
    p=m.switch_probability
    return Specification(jump_sizes=m.jump_sizes,jump_rates=tuple(m.rates),powers=m.collar_powers,
        variances=m.variance_states,transition=((1-p,p),(p,1-p)),
        constant_variance=.04 if m.model.startswith('constant_vol') else None)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.spec=Specification()
        self.paths=simulate_market(np.random.default_rng(101),8,2016,self.spec)

    def test_reference_does_not_import_primary_engine(self):
        from pathlib import Path
        import ast
        tree=ast.parse(Path(__file__).with_name('reference_ledger.py').read_text())
        imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        self.assertNotIn('cll_hedge',imports)

    def test_primary_reference_prices_and_wealth(self):
        for name in ('constant_vol','switching_vol','constant_vol_jumps','switching_vol_jumps'):
            m=primary.Market(model=name); spec=specification(m)
            paths=primary.simulate_paths(np.random.default_rng(12),8,2016,m)
            for n in (12,252,2016):
                grid=paths.on_grid(n)
                for claim in ('variance','exponential'):
                    tape=market_tape(spec,grid.times,grid.log_s,grid.qv,grid.state,claim)
                    result=execute(tape,frequency=n,stock_bps=1,claim_bps=1)
                    expected=primary.hedge(grid,m,claim)
                    np.testing.assert_allclose(result['wealth'],expected['wealth']-
                        1e-4*(expected['stock_turnover']+expected['power_turnover']),atol=3e-10,rtol=1e-10)
                    np.testing.assert_allclose(result['stock_turnover'],expected['stock_turnover'],atol=1e-9)
                    np.testing.assert_allclose(result['claim_turnover'],expected['power_turnover'],atol=1e-9)

    def test_volatility_state_jump_matching(self):
        p={key:array[:,:1] if key!='times' else array[:1] for key,array in self.paths.items()}
        # Test a repricing at t=0 without any stock move, past-QV change, or trade.
        for claim in ('variance','exponential'):
            old=market_tape(self.spec,**p,claim=claim)
            changed={**p,'states':1-p['states']}
            new=market_tape(self.spec,**changed,claim=claim)
            gain=np.sum(old['target_holdings']*(new['prices']-old['prices']),axis=2)
            np.testing.assert_allclose(gain,new['target_value']-old['target_value'],atol=1e-12)

    def test_zero_band_equals_rebalance_every_observation(self):
        tape=market_tape(self.spec,**self.paths)
        regular=execute(tape,frequency=2016,stock_bps=1,claim_bps=1)
        adaptive=execute(tape,policy='exposure_band',band=0,stock_bps=1,claim_bps=1)
        np.testing.assert_allclose(regular['wealth'],adaptive['wealth'],atol=1e-12)
        np.testing.assert_allclose(regular['fees'],adaptive['fees'],atol=1e-12)

    def test_infinite_band_is_buy_and_hold(self):
        tape=market_tape(self.spec,**self.paths)
        result=execute(tape,policy='exposure_band',band=np.inf)
        held=tape['target_holdings'][:,0]
        cash=tape['target_value'][:,0]-np.sum(held*tape['prices'][:,0],axis=1)
        expected=cash+np.sum(held*tape['prices'][:,-1],axis=1)
        np.testing.assert_allclose(result['wealth'],expected,atol=1e-12)
        np.testing.assert_array_equal(result['rebalance_events'],0)

    def test_cost_debits_exact_and_decisions_unchanged(self):
        tape=market_tape(self.spec,**self.paths)
        zero=execute(tape,policy='separate_band',band=.2,audit_path=True)
        paid=execute(tape,policy='separate_band',band=.2,stock_bps=1,claim_bps=2,audit_path=True)
        np.testing.assert_allclose(paid['wealth'],zero['wealth']-paid['fees'],atol=1e-12)
        np.testing.assert_array_equal(zero['trade_mask'],paid['trade_mask'])
        self.assertLess(paid['max_accounting_error'],1e-12)

    def test_no_future_information_in_trade_times(self):
        tape=market_tape(self.spec,**self.paths)
        altered={key:array.copy() for key,array in self.paths.items()}
        altered['log_stock'][:,1009:]+=.1
        changed=market_tape(self.spec,**altered)
        for policy in ('exposure_band','separate_band'):
            left=execute(tape,policy=policy,band=.2,audit_path=True)
            right=execute(changed,policy=policy,band=.2,audit_path=True)
            np.testing.assert_array_equal(left['trade_mask'][:,:1009],right['trade_mask'][:,:1009])

    def test_monitoring_and_trading_are_separate(self):
        tape=market_tape(self.spec,**self.paths)
        result=execute(tape,policy='exposure_band',band=.3,audit_path=True)
        self.assertTrue(np.all(result['rebalance_events']<result['monitoring_events']))
        coarse=execute(tape,policy='exposure_band',band=.3,monitor_stride=8,audit_path=True)
        self.assertFalse(np.any(coarse['trade_mask'][:,1::8]))

    def test_three_state_support_and_terminal_values(self):
        spec=Specification(variances=(.005,.035,.08),transition=((.98,.01,.01),(.01,.98,.01),(.01,.01,.98)))
        paths=simulate_market(np.random.default_rng(17),4,252,spec)
        for claim in ('variance','exponential'):
            tape=market_tape(spec,**paths,claim=claim)
            np.testing.assert_allclose(tape['target_value'][:,-1],tape['payoff'],atol=1e-13)
            for index,p in enumerate((2.,-1.,5.,-4.),start=2):
                np.testing.assert_allclose(tape['prices'][:,-1,index],np.exp(p*paths['log_stock'][:,-1]))


if __name__=='__main__':
    unittest.main(verbosity=2)
