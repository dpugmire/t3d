import unittest
import numpy as np
from t3d.TrinityIO import Input
from t3d.Species import SpeciesDict, Species
from t3d.Geometry import Geometry
from t3d.trinity_lib import TrinNormalizations
from t3d.Grid import Grid


class TestSpecies(unittest.TestCase):

    def test_1(self):
        inputs = Input('tests/unit/test-species.in').input_dict
        grid = Grid(inputs)

        species = SpeciesDict(inputs, grid)
        self.assertEqual(species.N_species, 4)

        self.assertEqual(species.ref_species.type, "deuterium")
        self.assertEqual(species.n_evolve_keys, ['deuterium', 'electron'])
        self.assertEqual(species.qneut_species.type, "x")
        self.assertEqual(species.T_evolve_keys, ['deuterium', 'tritium'])
        self.assertEqual(species.species_dict['electron'].temperature_equal_to, 'deuterium')
        self.assertEqual(species.species_dict['tritium'].temperature_equal_to, None)

    def test_2(self):
        inputs = Input('tests/unit/test-species-2.in').input_dict
        grid = Grid(inputs)

        species = SpeciesDict(inputs, grid)
        self.assertEqual(species.N_species, 3)

        self.assertEqual(species.ref_species.type, "tritium")

    def test_3(self):
        inputs = Input('tests/unit/test-species.in').input_dict
        grid = Grid(inputs)
        geo = Geometry(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor)

        species = SpeciesDict(inputs, grid)
        species2 = SpeciesDict(inputs, grid)

        species.init_profiles_and_sources(geo, norms)
        species2.init_profiles_and_sources(geo, norms)

        nt_vec = species.get_vec_from_profs()

        species.get_profs_from_vec(nt_vec)

        nt_vec = species.get_vec_from_profs()
        nt_vec2 = species2.get_vec_from_profs()
        np.testing.assert_allclose(nt_vec, nt_vec2)

    def test_4(self):
        inputs = Input('tests/unit/test-species.in').input_dict
        grid = Grid(inputs)
        geo = Geometry(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor)
        species = SpeciesDict(inputs, grid)

        species.init_profiles_and_sources(geo, norms)

        n_D = species['deuterium'].n()
        self.assertAlmostEqual(n_D[0], species['deuterium'].n_coefs[0], places=3)

        T_D = species['deuterium'].T()
        self.assertAlmostEqual(T_D[0], species['deuterium'].T_core, places=3)
        self.assertAlmostEqual(T_D[-1], species['deuterium'].T_edge)

        n_e = species['electron'].n()
        self.assertAlmostEqual(np.average(n_e.profile), species['electron'].n_lineavg)
        self.assertAlmostEqual(n_e[-1], species['electron'].n_edge)

        n_T = species['tritium'].n()
        self.assertAlmostEqual(geo.volume_average(n_T), species['tritium'].n_volavg, places=5)
        self.assertAlmostEqual(n_T[-1], species['tritium'].n_edge)

        np.testing.assert_allclose(species['tritium'].T().profile, 1-grid.rho**2)

        Sn_aux_e = species['electron'].Sn_labeled['aux']*norms.Sn_ref_SI20
        self.assertAlmostEqual(geo.volume_integrate(Sn_aux_e, interpolate=False), 1, places=5)

        # Sp_aux_e should be zero because T_e is not self-consistently evolving (equal_to = "deuterium")
        Sp_aux_e = species['electron'].Sp_labeled['aux']*norms.P_ref_MWm3
        self.assertAlmostEqual(geo.volume_integrate(Sp_aux_e, interpolate=False), 0)

        np.testing.assert_allclose(species['x'].T().profile, 0.1*species['electron'].T().profile)

    def test_5(self):
        inputs = Input('tests/unit/test-species.in').input_dict
        grid = Grid(inputs)
        geo = Geometry(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor)
        species = SpeciesDict(inputs, grid)

        species.init_profiles_and_sources(geo, norms)
        alpha = Species({'mass': 4, 'Z': 1, 'type':"alpha"}, grid)
        alpha.init_profiles(grid, None, norms)
        self.assertEqual(alpha.mass, 4)
        alpha.nu_equil(species['electron'])

    def test_6(self):
        inputs = Input('tests/unit/test-species-with-impurity.in').input_dict
        grid = Grid(inputs)
        species = SpeciesDict(inputs, grid)

        geo = Geometry(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor)
        species.init_profiles_and_sources(geo, norms)
        species.print_info()

        np.testing.assert_allclose(species['deuterium'].T().profile, 0.9*species['electron'].T().profile)

        np.testing.assert_allclose(species['tritium'].T().profile, 0.9*species['electron'].T().profile)

        np.testing.assert_allclose(species['deuterium'].n().profile, species['tritium'].n().profile)

        self.assertEqual(species['deuterium'].evolve_temperature, True)
        self.assertEqual(species['tritium'].evolve_temperature, False)
        self.assertEqual(species['tungsten'].evolve_temperature, True)

        # check impurity species type == tungsten
        self.assertEqual(species.impurity.type, 'tungsten')

        # check Zeff = 1.2
        Zeff = species.Zeff().profile
        np.testing.assert_allclose(Zeff, np.ones(len(Zeff))*1.2)


if __name__ == '__main__':
    unittest.main()
