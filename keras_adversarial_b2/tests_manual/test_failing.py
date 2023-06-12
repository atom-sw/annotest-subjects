from examples.example_gan import model_discriminator
from keras_adversarial.legacy import l1l2


def test_failing():
    input_shape = (1, 1)
    hidden_dim = 1024
    reg = lambda: l1l2(1e-5, 1e-5)
    output_activation = 'sigmoid'

    model_discriminator(input_shape, hidden_dim, reg, output_activation)
