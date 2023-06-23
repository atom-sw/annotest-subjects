import unittest
import hypothesis as hy
import hypothesis.strategies as st
from examples.mnist_utils import mnist_data


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_mnist_data(self, noArgCall):
        mnist_data()
