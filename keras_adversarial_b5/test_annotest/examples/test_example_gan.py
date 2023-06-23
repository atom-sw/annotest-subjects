import unittest
import hypothesis as hy
import hypothesis.strategies as st
from examples.example_gan import main


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_main(self, noArgCall):
        main()
