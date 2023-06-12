from densenet import DenseNet


def test_failing():
    input_shape = (20, 20, 1)
    dense_blocks = 2
    dense_layers = -1
    growth_rate = 1
    nb_classes = 2
    dropout_rate = None
    bottleneck = True
    compression = 5e-324
    weight_decay = 1e-4
    depth = 10
    DenseNet(input_shape, dense_blocks, dense_layers,
             growth_rate, nb_classes, dropout_rate,
             bottleneck, compression, weight_decay,
             depth)

