import unittest
import hypothesis as hy
import hypothesis.strategies as st
from data_loader import _get_file_path, load_class_names, load_test_data, load_training_data


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(filename=st.just(''))
    @hy.settings(deadline=None)
    def test__get_file_path(self, filename):
        _get_file_path(filename)

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_load_class_names(self, noArgCall):
        load_class_names()

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_load_training_data(self, noArgCall):
        load_training_data()

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_load_test_data(self, noArgCall):
        load_test_data()
