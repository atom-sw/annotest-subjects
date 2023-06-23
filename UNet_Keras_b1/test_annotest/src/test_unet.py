import unittest
import hypothesis as hy
import hypothesis.strategies as st
from src.unet import UNet, generator_UNet__add_encode_layers_input_layer


class Test_UNet(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_init(self, noArgCall):
        UNet()

    @hy.given(filters=st.sampled_from([32, 64, 128, 256, 512]), st_for_data
        =st.data())
    @hy.settings(deadline=None)
    def test___add_Encode_layers(self, filters, st_for_data):
        obj = UNet()
        inputLayer = generator_UNet__add_encode_layers_input_layer()
        obj.__add_Encode_layers(filters, inputLayer)

    @hy.given(st_for_data=st.data())
    @hy.settings(deadline=None)
    def test_model(self, st_for_data):
        obj = UNet()
        obj.model()
