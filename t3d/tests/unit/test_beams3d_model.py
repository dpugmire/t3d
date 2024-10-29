import unittest
from t3d.TrinityIO import Input
from t3d.Grid import Grid
from t3d.Time import Time
from t3d.Geometry import Geometry
from t3d.Species import SpeciesDict
from t3d.Import import Import
from t3d.SourceModels import SourceModelDict
from t3d.trinity_lib import TrinNormalizations


class TestBEAMS3D_FluxModel(unittest.TestCase):

    def test_1(self):
        inputs = Input('tests/unit/test-beams3d-model.in').input_dict
        grid = Grid(inputs)
        time = Time(inputs)
        imported = Import(inputs, grid)
        try:
            source_models = SourceModelDict(inputs, grid, time)
        except RuntimeError:
            raise unittest.SkipTest("BEAMS3D not found. Skipping test.")
        geo = Geometry(inputs, grid, None, imported)
        species = SpeciesDict(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor, vt_sqrt_2=False, m_ref=species.ref_species.mass)
        species.init_all_profiles(geo, norms, imported)
        species.init_all_sources(geo, None, norms, imported)

        species.clear_sources()
        for model in source_models.get_sources_list():
            model.compute_sources(species, geo, norms)


if __name__ == '__main__':
    unittest.main()
