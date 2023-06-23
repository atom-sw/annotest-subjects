import unittest
import hypothesis as hy
import hypothesis.strategies as st
from examples.example_gan_cifar10 import main, model_discriminator, model_generator


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_model_generator(self, noArgCall):
        model_generator()

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_model_discriminator(self, noArgCall):
        model_discriminator()

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_main(self, noArgCall):
        main()
