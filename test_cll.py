"""Deterministic mathematical tests for the CLL implementation (no pytest needed)."""
import unittest
import numpy as np
from cll_hedge import Market, Paths, EuropeanMarket, hedge, laplace_power, simulate_paths


class CLLTests(unittest.TestCase):
    def setUp(self):
        self.market=Market()

    def test_jump_matching_variance(self):
        m=self.market; powers=np.asarray(m.collar_powers); k=m.variance_collars()
        for z in m.jump_sizes:
            gain=-2*z+(2+k@(1-2*powers))*np.expm1(z)
            gain+=k@(np.exp(powers*z)-np.exp((1-powers)*z))
            self.assertAlmostEqual(gain,z*z,places=12)

    def test_jump_matching_exponential(self):
        m=self.market; powers=np.asarray(m.collar_powers)
        for a in (.001,.05,.1):
            p=laplace_power(a); k=m.exponential_collars(a)
            for z in m.jump_sizes:
                gain=np.expm1(p*z)+(-p+k@(1-2*powers))*np.expm1(z)
                gain+=k@(np.exp(powers*z)-np.exp((1-powers)*z))
                self.assertAlmostEqual(gain,np.expm1(-a*z*z),places=12)

    def test_variance_is_analytic_derivative(self):
        a=1e-6
        np.testing.assert_allclose(-self.market.exponential_collars(a)/a,
                                   self.market.variance_collars(),rtol=1e-5)

    def test_wrong_sign_fails_jump_match(self):
        m=self.market; z=np.asarray(m.jump_sizes)
        residual=2*(np.expm1(z)-z)-z*z
        np.testing.assert_allclose(m.jump_matrix()@(-m.variance_collars()),-residual,atol=1e-13)
        self.assertGreater(np.linalg.norm(2*residual),.001)

    def test_power_symmetry(self):
        m=self.market
        paths=simulate_paths(np.random.default_rng(1),4,504,m)
        pricer=EuropeanMarket(m,paths.times)
        for p in m.collar_powers:
            left=pricer.power_log_price(paths,p)-p*paths.log_s+(1-paths.times)*m.psi(1-p)
            right=pricer.power_log_price(paths,1-p)-(1-p)*paths.log_s+(1-paths.times)*m.psi(p)
            np.testing.assert_allclose(left,right,atol=1e-13)

    def test_target_price_equals_paper_A_times_power_price(self):
        m=self.market; a=.1; p=laplace_power(a)
        paths=simulate_paths(np.random.default_rng(10),5,504,m)
        pricer=EuropeanMarket(m,paths.times)
        A=np.exp(-p*paths.log_s-a*paths.qv+(1-paths.times)*(m.target_jump_exponent(a)-m.psi(p)))
        np.testing.assert_allclose(A*np.exp(pricer.power_log_price(paths,p)),pricer.exponential_value(paths,a),atol=1e-13)

    def test_mgf_by_exhaustive_two_day_enumeration(self):
        m=self.market; p=m.switch_probability; v=np.array(m.variance_states)
        t=np.array([250/252,250.5/252,251/252,1.])
        pricer=EuropeanMarket(m,t)
        for c in (-.1,1.,10.):
            marks=pricer.remaining_iv(c)
            for state in (0,1):
                expected=np.exp(c*v[state]/252)*((1-p)*np.exp(c*v[state]/252)+p*np.exp(c*v[1-state]/252))
                self.assertAlmostEqual(marks[0,state],expected,places=13)
                expected=np.exp(c*v[state]/504)*((1-p)*np.exp(c*v[state]/252)+p*np.exp(c*v[1-state]/252))
                self.assertAlmostEqual(marks[1,state],expected,places=13)
        np.testing.assert_allclose(pricer.remaining_iv()[-1],0)

    def test_no_jump_collars_vanish(self):
        m=Market(model='constant_vol')
        np.testing.assert_array_equal(m.variance_collars(),0)
        np.testing.assert_array_equal(m.exponential_collars(.1),0)

    def test_variance_matches_old_baseline_without_jumps(self):
        from simulate import hedge as old_hedge
        m=Market(model='constant_vol')
        paths=simulate_paths(np.random.default_rng(2),5,2016,m)
        for n in (12,252,2016):
            result=hedge(paths.on_grid(n),m)
            expected,fees=old_hedge(paths.log_s,n,1.)
            np.testing.assert_allclose(result['wealth'],expected,atol=1e-13)
            np.testing.assert_allclose(result['stock_turnover']*1e-4,fees,atol=1e-13)

    def test_cash_ledger_and_no_target_reset(self):
        m=self.market; paths=simulate_paths(np.random.default_rng(3),2,2016,m).on_grid(12)
        for claim in ('variance','exponential'):
            result=hedge(paths,m,claim=claim,ledger=True)
            errors=[]
            for row in result['ledger']:
                assets=row['cash']+row['stock_units']*row['stock']+row['base_claim_value']+row['collar_net_value']
                self.assertAlmostEqual(assets,row['wealth'],places=12)
                self.assertEqual(row['external_cash_flow'],0)
                errors.append(row['tracking_error'])
            self.assertGreater(max(abs(x) for x in errors),1e-8)

    def test_prefix_causality(self):
        m=self.market; paths=simulate_paths(np.random.default_rng(4),3,504,m)
        edited=Paths(paths.times,paths.log_s.copy(),paths.qv.copy(),paths.state.copy())
        edited.log_s[:,253:]+=1
        edited.qv[:,253:]+=.1
        edited.state[:,253:]=1-edited.state[:,253:]
        for data in (paths,edited):
            pricer=EuropeanMarket(m,data.times)
            marks=pricer.power_log_price(data,2.)[:,:253]
            if data is paths:
                expected=marks.copy()
            else:
                np.testing.assert_array_equal(marks,expected)

    def test_self_financing_from_frozen_positions(self):
        m=self.market; paths=simulate_paths(np.random.default_rng(11),2,2016,m).on_grid(252)
        for claim in ('variance','exponential'):
            records=hedge(paths,m,claim=claim,ledger=True)['ledger']
            for previous,current in zip(records,records[1:]):
                value=previous['cash']+previous['stock_units']*current['stock']
                value+=previous['base_claim_units']*current['base_claim_mark']
                for j in (1,2):
                    for leg in ('long','short'):
                        value+=previous[f'collar_{j}_{leg}_units']*current[f'collar_{j}_{leg}_mark']
                self.assertAlmostEqual(value,current['wealth'],places=11)

    def test_reject_singular_jump_support(self):
        m=Market(jump_sizes=(-.15,-.15))
        with self.assertRaises(ValueError):
            m.variance_collars()

    def test_terminal_values(self):
        m=self.market; paths=simulate_paths(np.random.default_rng(5),3,252,m)
        pricer=EuropeanMarket(m,paths.times)
        np.testing.assert_allclose(pricer.variance_value(paths)[:,-1],paths.qv[:,-1])
        np.testing.assert_allclose(pricer.exponential_value(paths,.1)[:,-1],np.exp(-.1*paths.qv[:,-1]))
        for p in (2.,5.):
            np.testing.assert_allclose(pricer.power_log_price(paths,p)[:,-1],p*paths.log_s[:,-1])


if __name__=='__main__':
    unittest.main(verbosity=2)
