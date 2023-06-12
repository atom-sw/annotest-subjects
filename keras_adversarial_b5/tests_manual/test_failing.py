from examples.example_gan_convolutional import model_discriminator


def test_failing():
    input_shape = (1, 28, 28)
    dropout_rate = 0.5

    model_discriminator(input_shape, dropout_rate)
