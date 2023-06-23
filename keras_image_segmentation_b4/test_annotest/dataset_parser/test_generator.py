import unittest
import hypothesis as hy
import hypothesis.strategies as st
import hypothesis.extra.numpy as hynp
import numpy as np
import numpy as np
from dataset_parser.generator import get_result_map


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(y_img=hynp.arrays(dtype=np.dtype('float32'), shape=st.
        sampled_from([(4, 256, 512, 1)])))
    @hy.settings(deadline=None)
    def test_get_result_map(self, y_img):
        get_result_map(y_img)
