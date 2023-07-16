import unittest
import hypothesis as hy
import hypothesis.strategies as st
from src.unet import UNet


class Test_UNet(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_init(self, noArgCall):
        UNet()

    @hy.given(st_for_data=st.data())
    @hy.settings(deadline=None)
    def test_model(self, st_for_data):
        obj = UNet()
        obj.model()
