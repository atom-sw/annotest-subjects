import unittest
import hypothesis as hy
import hypothesis.strategies as st
from keras_adversarial.legacy import BatchNormalization, l1l2


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(l1=st.just(0), l2=st.just(0))
    @hy.settings(deadline=None)
    def test_l1l2(self, l1, l2):
        l1l2(l1, l2)

    @hy.given(mode=st.just(0))
    @hy.settings(deadline=None)
    def test_BatchNormalization(self, mode):
        BatchNormalization(mode)
