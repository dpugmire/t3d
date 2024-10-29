import unittest
from t3d.Import import TranspReader


class TestTranspReader(unittest.TestCase):

    def test_1(self):
        transp = TranspReader('tests/data/pr08_jet_42982_2d.dat', time=15.2)
        print(transp.data.keys())
        print(transp.data['NE'])

        print(transp.find_ion_species_index(species_mass=3.0))
        print(transp.get_density(species_mass=3.0))


if __name__ == '__main__':
    unittest.main()
