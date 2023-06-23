import unittest
import hypothesis as hy
import hypothesis.strategies as st
from examples.cifar10_utils import cifar10_data


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_cifar10_data(self, noArgCall):
        cifar10_data()
