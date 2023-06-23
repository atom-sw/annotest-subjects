import unittest
import hypothesis as hy
import hypothesis.strategies as st
from ThinPlateSpline2 import ThinPlateSpline2, generator_ThinPlateSpline2_U, generator_ThinPlateSpline2_source, generator_ThinPlateSpline2_target


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(out_size=st.sampled_from([[256, 263]]), st_for_data=st.data())
    @hy.settings(deadline=None)
    def test_ThinPlateSpline2(self, out_size, st_for_data):
        U = generator_ThinPlateSpline2_U()
        source = generator_ThinPlateSpline2_source()
        target = generator_ThinPlateSpline2_target()
        ThinPlateSpline2(U, source, target, out_size)
