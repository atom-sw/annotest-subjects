import unittest
import hypothesis as hy
import hypothesis.strategies as st
from squad.prepro import get_args, main


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_main(self, noArgCall):
        main()

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_get_args(self, noArgCall):
        get_args()
