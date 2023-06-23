import unittest
import hypothesis as hy
import hypothesis.strategies as st
from keras_frcnn.config import Config


class Test_Config(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_init(self, noArgCall):
        Config()
