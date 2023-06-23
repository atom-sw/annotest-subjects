import unittest
import hypothesis as hy
import hypothesis.strategies as st
from src.main import predict, train


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_train(self, noArgCall):
        train()

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_predict(self, noArgCall):
        predict()
