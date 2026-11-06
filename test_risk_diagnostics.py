import unittest
import numpy as np
from reference_ledger import Specification,market_tape
from risk_diagnostics import target_units,local_risk,boundary_risk


class RiskDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.spec=Specification()
        self.tape=market_tape(self.spec,np.array([.2]),np.array([[np.log(1.1)]]),np.array([[.012]]),np.array([[1]]))
        self.y=self.tape['prices'][0,0]

    def test_prescribed_hedge_cancels_local_price_risks(self):
        h=target_units(self.spec,self.y)
        result=local_risk(self.spec,self.y,h,.07)
        self.assertLess(float(result['total_variance_rate']),1e-20)
        np.testing.assert_allclose(h,self.tape['target_holdings'][0,0],atol=1e-13)

    def test_covariance_quadratic_form_equals_exposure_calculation(self):
        h0=target_units(self.spec,self.y)
        error=np.array([.03,.02,-.01,.04,.05,-.02])
        result=local_risk(self.spec,self.y,h0+error,.07)
        expected=error@result['covariance_rate']@error
        self.assertAlmostEqual(float(result['total_variance_rate']),float(expected),places=12)

    def test_covariance_is_symmetric_positive_semidefinite(self):
        result=local_risk(self.spec,self.y,np.zeros(6),.07)
        matrix=result['covariance_rate']
        np.testing.assert_allclose(matrix,matrix.T,atol=1e-13)
        self.assertGreater(np.min(np.linalg.eigvalsh(matrix)),-1e-12)

    def test_unhedged_claim_has_poisson_variance_rate(self):
        result=local_risk(self.spec,self.y,np.zeros(6),.07)
        expected=sum(rate*z**4 for rate,z in zip(self.spec.jump_rates,self.spec.jump_sizes))
        self.assertAlmostEqual(float(result['total_variance_rate']),expected,places=13)

    def test_fresh_hedge_matches_daily_volatility_information(self):
        result=boundary_risk(self.spec,1/252,0.,.01/252,0)
        self.assertLess(result['variance_atom'],1e-20)
        self.assertGreater(np.trace(result['covariance_atom']),0)
        np.testing.assert_allclose(result['probabilities']@result['price_jumps'],0,atol=1e-13)

    def test_daily_atom_mismatch_is_its_covariance_quadratic_form(self):
        before=boundary_risk(self.spec,1/252,0.,.01/252,0)
        fresh=target_units(self.spec,before['prices_before'])
        gap=np.array([0.,.2,.01,.03,-.02,.04])
        result=boundary_risk(self.spec,1/252,0.,.01/252,0,fresh+gap)
        expected=gap@before['covariance_atom']@gap
        self.assertAlmostEqual(result['variance_atom'],expected,places=12)

    def test_daily_atom_not_defined_as_a_continuous_hazard(self):
        with self.assertRaises(ValueError):
            boundary_risk(self.spec,.001,0.,0.,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
