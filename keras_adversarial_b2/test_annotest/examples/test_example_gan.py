from keras_adversarial.legacy import l1l2
import unittest
import hypothesis as hy
import hypothesis.strategies as st
import hypothesis.extra.numpy as hynp
from examples.example_gan import main, model_discriminator


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(input_shape=hynp.array_shapes(min_dims=2, max_dims=2,
        min_side=1, max_side=28), hidden_dim=st.one_of(st.integers(
        min_value=1024, max_value=2048), st.just(1024)), reg=st.just(lambda :
        l1l2(1e-05, 1e-05)), output_activation=st.just('sigmoid'))
    @hy.settings(deadline=None)
    def test_model_discriminator(self, input_shape, hidden_dim, reg,
        output_activation):
        model_discriminator(input_shape, hidden_dim, reg, output_activation)

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_main(self, noArgCall):
        main()
