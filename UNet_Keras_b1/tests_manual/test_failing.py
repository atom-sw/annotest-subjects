from keras import Input

from src.unet import UNet


def test_failing():
    filters = 32
    obj = UNet()
    inputLayer = Input((258, 258, 3))
    obj.__add_Encode_layers(filters, inputLayer)
