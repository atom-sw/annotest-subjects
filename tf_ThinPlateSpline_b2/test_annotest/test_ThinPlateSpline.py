import unittest
import hypothesis as hy
import hypothesis.strategies as st
from ThinPlateSpline import ThinPlateSpline, generator_ThinPlateSpline_U, generator_ThinPlateSpline_coord, generator_ThinPlateSpline_out_size, generator_ThinPlateSpline_vector


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(st_for_data=st.data())
    @hy.settings(deadline=None)
    def test_ThinPlateSpline(self, st_for_data):
        U = generator_ThinPlateSpline_U()
        coord = generator_ThinPlateSpline_coord()
        vector = generator_ThinPlateSpline_vector()
        out_size = generator_ThinPlateSpline_out_size()
        ThinPlateSpline(U, coord, vector, out_size)
