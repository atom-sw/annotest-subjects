import unittest
import hypothesis as hy
import hypothesis.strategies as st
from keras_adversarial.adversarial_optimizers import AdversarialOptimizerAlternating


class Test_AdversarialOptimizerAlternating(unittest.TestCase):

    @hy.given(reverse=st.just(False))
    @hy.settings(deadline=None)
    def test_init(self, reverse):
        AdversarialOptimizerAlternating(reverse)
