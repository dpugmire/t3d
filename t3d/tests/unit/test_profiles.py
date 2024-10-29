import unittest
import numpy as np
from t3d.Profiles import GridProfile, FluxProfile
from t3d.Grid import Grid


class TestGridProfile(unittest.TestCase):

    def test(self):
        grid = Grid({})
        rho_prof = GridProfile(grid.rho, grid)
        self.assertEqual(rho_prof.length, grid.N_radial)

        mid_prof = rho_prof.toFluxProfile()
        assert isinstance(mid_prof, FluxProfile)

        # test multiplication by scalar
        test = 2*rho_prof
        assert isinstance(test, GridProfile)

        test = rho_prof*2
        assert isinstance(test, GridProfile)

        # test multiplication of two GridProfiles
        test2 = test * rho_prof
        assert isinstance(test2, GridProfile)

        # test multiplication of a GridProfile and an array (of same length)
        test2 = test * grid.rho
        assert isinstance(test2, GridProfile)

        test2 = grid.rho * test
        assert isinstance(test2, GridProfile)

        # test that multiplication of a GridProfile and a different-sized array
        # throws an error
        with self.assertRaises(Exception):
            test2 = test * grid.rho[:-1]

        # test that a GridProfile * FluxProfile throws an error
        with self.assertRaises(Exception):
            test = mid_prof * rho_prof

        # test addition of two GridProfiles
        test2 = test + rho_prof
        assert isinstance(test2, GridProfile)

        # test subtraction of GridProfile and an array (of same length)
        test = rho_prof.profile - 2*rho_prof
        assert isinstance(test, GridProfile)
        assert (test.profile == -1*rho_prof.profile).all()

        test = rho_prof - 2*rho_prof.profile
        assert isinstance(test, GridProfile)
        assert (test.profile == -1*rho_prof.profile).all()

        # test division of two GridProfiles
        test = (2*rho_prof) / (5*rho_prof)
        assert isinstance(test, GridProfile)
        np.testing.assert_allclose(test.profile, 2/5*np.ones(grid.N_radial))

        test = (2*rho_prof.profile) / (5*rho_prof)
        assert isinstance(test, GridProfile)
        np.testing.assert_allclose(test.profile, 2/5*np.ones(grid.N_radial))

        test = (2*rho_prof) / (5*rho_prof.profile)
        assert isinstance(test, GridProfile)
        np.testing.assert_allclose(test.profile, 2/5*np.ones(grid.N_radial))

        test = rho_prof / 5
        assert isinstance(test, GridProfile)
        np.testing.assert_allclose(test.profile, 0.2*rho_prof.profile)

        # test gradient_4pt method
        grad = rho_prof.gradient_4pt()
        assert isinstance(grad, GridProfile)

        # test gradient_5pt method
        grad = rho_prof.gradient_5pt()
        assert isinstance(grad, GridProfile)

        # test log_gradient method
        grad_log = rho_prof.log_gradient()
        assert isinstance(grad_log, GridProfile)

        # test gradient method
        grad = rho_prof.gradient_as_FluxProfile()
        assert isinstance(grad, FluxProfile)

        # test log_gradient method
        grad_log = rho_prof.log_gradient_as_FluxProfile()
        assert isinstance(grad_log, FluxProfile)

    def test_plus_minus(self):
        grid = Grid({})
        n_core = 4
        n_edge = 0.5
        n_prof = GridProfile((n_core - n_edge)*(1 - (grid.rho/grid.rho_edge)**2) + n_edge, grid)
        n_mid = FluxProfile((n_core - n_edge)*(1 - (grid.midpoints/grid.rho_edge)**2) + n_edge, grid)
        np.testing.assert_allclose(n_mid, n_prof.plus())
        np.testing.assert_allclose(n_mid[:-1], n_prof.minus()[1:])


class TestFluxProfile(unittest.TestCase):

    def test(self):
        grid = Grid({})
        test = FluxProfile(0, grid)
        np.testing.assert_allclose(test.profile, np.zeros(grid.N_radial-1))


if __name__ == '__main__':
    unittest.main()
