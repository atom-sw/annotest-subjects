import unittest
import hypothesis as hy
import hypothesis.strategies as st
from __future__ import print_function
from dataset_parser.make_h5 import make_h5py


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_make_h5py(self, noArgCall):
        make_h5py()
