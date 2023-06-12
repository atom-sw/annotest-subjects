from keras import Input

from src.unet import UNet


def generator_UNet__add_decode_layers_input_layer():
    inputs = Input((572, 572, 1))
    unetObject = UNet()

    encodeLayer1 = unetObject.__add_Encode_layers(64, inputs, is_first=True)
    encodeLayer2 = unetObject.__add_Encode_layers(128, encodeLayer1)
    encodeLayer3 = unetObject.__add_Encode_layers(256, encodeLayer2)
    encodeLayer4 = unetObject.__add_Encode_layers(512, encodeLayer3)
    encodeLayer5 = unetObject.__add_Encode_layers(1024, encodeLayer4)

    return encodeLayer5


def generator_UNet__add_decode_layers_concat_layer():
    inputs = Input((572, 572, 1))
    unetObject = UNet()

    encodeLayer1 = unetObject.__add_Encode_layers(64, inputs, is_first=True)
    encodeLayer2 = unetObject.__add_Encode_layers(128, encodeLayer1)
    encodeLayer3 = unetObject.__add_Encode_layers(256, encodeLayer2)
    encodeLayer4 = unetObject.__add_Encode_layers(512, encodeLayer3)

    return encodeLayer4


def test_failing():
    filters = 512
    obj = UNet()
    inputLayer = generator_UNet__add_decode_layers_input_layer()
    concatLayer = generator_UNet__add_decode_layers_concat_layer()
    obj.__add_Decode_layers(filters, inputLayer, concatLayer)
