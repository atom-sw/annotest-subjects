from keras import Input
from keras.layers import Concatenate, Reshape
import keras.backend as K

from Tensorflow_progressive_growing_of_gans.layers import PixelNormLayer
from Tensorflow_progressive_growing_of_gans.model import G_convblock, lrelu, lrelu_init


def generator_G_convblock_net(label_size, latent_size, normalize_latents):
    inputs = [Input(shape=[latent_size], name='Glatents')]
    net = inputs[-1]
    if normalize_latents:
        net = PixelNormLayer(name='Gnorm')(net)
    if label_size:
        inputs += [Input(shape=[label_size], name='Glabels')]
        net = Concatenate(name='G1na')([net, inputs[-1]])
    net = Reshape((1, 1, K.int_shape(net)[1]), name='G1nb')(net)
    return net


def test_failing():
    co_net_label_size = 0
    co_net_latent_size = 1
    co_net_normalize_latents = True
    net = generator_G_convblock_net(co_net_label_size,
                                    co_net_latent_size,
                                    co_net_normalize_latents)
    num_filter = 296
    filter_size = 1
    actv = lrelu
    init = lrelu_init
    pad = 'full'
    use_wscale = True
    use_pixelnorm = True
    use_batchnorm = False
    name = 'SomeName'

    G_convblock(net, num_filter, filter_size, actv, init, pad,
                use_wscale, use_pixelnorm, use_batchnorm, name)
