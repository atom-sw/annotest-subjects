from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate

from annotest import an_language as an


@an.generator()
@an.exclude()
def generator_UNet__add_decode_layers_input_layer():
    inputs = Input((572, 572, 1))
    unetObject = UNet()

    encodeLayer1 = unetObject._add_Encode_layers(64, inputs, is_first=True)
    encodeLayer2 = unetObject._add_Encode_layers(128, encodeLayer1)
    encodeLayer3 = unetObject._add_Encode_layers(256, encodeLayer2)
    encodeLayer4 = unetObject._add_Encode_layers(512, encodeLayer3)
    encodeLayer5 = unetObject._add_Encode_layers(1024, encodeLayer4)

    return encodeLayer5


@an.generator()
@an.exclude()
def generator_UNet__add_decode_layers_concat_layer():
    inputs = Input((572, 572, 1))
    unetObject = UNet()

    encodeLayer1 = unetObject._add_Encode_layers(64, inputs, is_first=True)
    encodeLayer2 = unetObject._add_Encode_layers(128, encodeLayer1)
    encodeLayer3 = unetObject._add_Encode_layers(256, encodeLayer2)
    encodeLayer4 = unetObject._add_Encode_layers(512, encodeLayer3)

    return encodeLayer4


class UNet(object):
    def __init__(self):

        inputs = Input((572, 572, 1))

        encodeLayer1 = self.__add_Encode_layers(64, inputs)
        encodeLayer2 = self.__add_Encode_layers(128, encodeLayer1)
        encodeLayer3 = self.__add_Encode_layers(256, encodeLayer2)
        encodeLayer4 = self.__add_Encode_layers(512, encodeLayer3)

        conv1 = Conv2D(1024, (3, 3), strides=(3, 3),
                       activation='relu')(encodeLayer4)
        conv2 = Conv2D(1024, (3, 3), strides=(3, 3),
                       activation='relu')(conv1)

        decodeLayer1 = self.__add_Decode_layers(512, conv2, encodeLayer4)
        decodeLayer2 = self.__add_Decode_layers(
            256, decodeLayer1, encodeLayer3)
        decodeLayer3 = self.__add_Decode_layers(
            128, decodeLayer2, encodeLayer2)
        decodeLayer4 = self.__add_Decode_layers(64, decodeLayer3, encodeLayer1)

        outputs = Conv2D(1, (3, 3), strides=(
            3, 3), activation='relu')(decodeLayer4)

        self.MODEL = Model(inputs=inputs, outputs=outputs)

    # def __add_Encode_layers(self, filters, inputLayer):  # repo_change
    #     layer = Conv2D(filters, (3, 3), strides=(  # repo_change
    #         3, 3), activation='relu')(inputLayer)  # repo_change
    #     layer = Conv2D(filters, (3, 3), strides=(  # repo_change
    #         3, 3), activation='relu')(layer)  # repo_change
    #     layer = MaxPooling2D((2, 2))(layer)  # repo_change
    def __add_Encode_layers(self, filters, inputLayer, is_first=False):  # repo_change
        layer = inputLayer  # repo_change
        if is_first:  # repo_change
            layer = Conv2D(filters, 3, activation='relu',  # repo_change
                           input_shape=(572, 572, 1))(layer)  # repo_change
        else:  # repo_change
            layer = MaxPooling2D(2)(layer)  # repo_change
            layer = Conv2D(filters, 3, activation='relu')(layer)  # repo_change
        layer = Conv2D(filters, 3, activation='relu')(layer)  # repo_change
        return layer

    @an.arg("filters", an.sampled([512]))
    @an.arg("inputLayer", an.obj(generator_UNet__add_decode_layers_input_layer))
    @an.arg("concatLayer", an.obj(generator_UNet__add_decode_layers_concat_layer))
    def __add_Decode_layers(self, filters, inputLayer, concatLayer):
        layer = UpSampling2D((2, 2))(inputLayer)  # repo_bug (not clear)
        layer = Concatenate()([layer, concatLayer])  # repo_bug (not clear)
        layer = Conv2D(filters, (3, 3), strides=(  # repo_bug (not clear)
            3, 3), activation='relu')(layer)  # repo_bug (not clear)
        layer = Conv2D(filters, (3, 3), strides=(  # repo_bug (not clear)
            3, 3), activation='relu')(layer)  # repo_bug (not clear)
        layer = MaxPooling2D((2, 2))(layer)  # repo_bug (not clear)
        return layer

    def model(self):
        return self.MODEL
