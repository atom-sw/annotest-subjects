import unittest
import hypothesis as hy
import hypothesis.strategies as st
from src.unet import UNet, generator_UNet__add_decode_layers_input_layer, generator_UNet__add_decode_layers_concat_layer


class Test_UNet(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_init(self, noArgCall):
        UNet()

    @hy.given(filters=st.sampled_from([512]), st_for_data=st.data())
    @hy.settings(deadline=None)
    def test___add_Decode_layers(self, filters, st_for_data):
        obj = UNet()
        inputLayer = generator_UNet__add_decode_layers_input_layer()
        concatLayer = generator_UNet__add_decode_layers_concat_layer()
        obj.__add_Decode_layers(filters, inputLayer, concatLayer)

    @hy.given(st_for_data=st.data())
    @hy.settings(deadline=None)
    def test_model(self, st_for_data):
        obj = UNet()
        obj.model()
