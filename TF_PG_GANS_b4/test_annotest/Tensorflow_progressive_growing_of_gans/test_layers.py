import unittest
import hypothesis as hy
import hypothesis.strategies as st
from Tensorflow_progressive_growing_of_gans.layers import GDropLayer, MinibatchStatConcatLayer, PixelNormLayer


class Test_PixelNormLayer(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_init(self, noArgCall):
        PixelNormLayer()


class Test_MinibatchStatConcatLayer(unittest.TestCase):

    @hy.given(averaging=st.just('all'))
    @hy.settings(deadline=None)
    def test_init(self, averaging):
        MinibatchStatConcatLayer(averaging)


class Test_GDropLayer(unittest.TestCase):

    @hy.given(mode=st.just('mul'), strength=st.just(0.4), axes=st.just((0, 
        3)), normalize=st.just(False))
    @hy.settings(deadline=None)
    def test_init(self, mode, strength, axes, normalize):
        GDropLayer(mode, strength, axes, normalize)
