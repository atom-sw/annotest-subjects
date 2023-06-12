from densenet import DenseNet


def test_failing():
    input_shape = (30, 30, 1)
    dense_blocks = 3
    dense_layers = -1
    growth_rate = 12
    nb_classes = 2
    dropout_rate = 5e-324
    bottleneck = False
    compression = 1
    weight_decay = 1e-4
    depth = 40
    DenseNet(input_shape, dense_blocks, dense_layers,
             growth_rate, nb_classes, dropout_rate,
             bottleneck, compression, weight_decay,
             depth)

