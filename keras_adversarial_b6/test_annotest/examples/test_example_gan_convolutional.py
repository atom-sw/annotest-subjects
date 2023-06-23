import unittest
import hypothesis as hy
import hypothesis.strategies as st
import hypothesis.extra.numpy as hynp
from examples.example_gan_convolutional import mnist_data, model_discriminator, model_generator


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_model_generator(self, noArgCall):
        model_generator()

    @hy.given(input_shape=st.one_of(hynp.array_shapes(min_dims=3, max_dims=
        3, min_side=1, max_side=60), st.just((1, 28, 28))), dropout_rate=st
        .one_of(st.floats(min_value=0, max_value=1, allow_nan=None,
        allow_infinity=None, width=64, exclude_min=False, exclude_max=True),
        st.just(0.5)))
    @hy.settings(deadline=None)
    def test_model_discriminator(self, input_shape, dropout_rate):
        model_discriminator(input_shape, dropout_rate)

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_mnist_data(self, noArgCall):
        mnist_data()
