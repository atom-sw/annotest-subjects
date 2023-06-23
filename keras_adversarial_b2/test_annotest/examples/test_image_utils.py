import unittest
import hypothesis as hy
import hypothesis.strategies as st
from examples.image_utils import channel_axis


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_channel_axis(self, noArgCall):
        channel_axis()
