import unittest
from t3d.TrinityIO import Input
from t3d.Geometry import Geometry
from t3d.Grid import Grid
from t3d.Import import Import


class TestGeometry(unittest.TestCase):

    def test_geometry_basic(self):
        inputs = Input('tests/unit/test-geometry-basic.in').input_dict
        grid = Grid(inputs)
        _ = Geometry(inputs, grid)

    def test_geometry_miller(self):
        inputs = Input('tests/unit/test-geometry-miller.in').input_dict
        grid = Grid(inputs)
        imported = Import(inputs, grid)
        _ = Geometry(inputs, grid, imported=imported)

    def test_geometry_vmec(self):
        inputs = Input('tests/unit/test-geometry-vmec.in').input_dict
        grid = Grid(inputs)
        _ = Geometry(inputs, grid)


if __name__ == '__main__':
    unittest.main()
