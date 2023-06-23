import unittest
import hypothesis as hy
import hypothesis.strategies as st
from __future__ import print_function
from dataset_parser.h5_test import h5py_test, make_h5py, read_h5py_example


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_make_h5py(self, noArgCall):
        make_h5py()

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_read_h5py_example(self, noArgCall):
        read_h5py_example()

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_h5py_test(self, noArgCall):
        h5py_test()
