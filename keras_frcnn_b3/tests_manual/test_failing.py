from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization


def test_failing():
    epsilon = 0.001
    axis = -1
    weights = None
    beta_init = 'zero'
    gamma_init = 'one'
    gamma_regularizer = None
    beta_regularizer = None

    FixedBatchNormalization(epsilon, axis, weights,
                            beta_init, gamma_init,
                            gamma_regularizer, beta_regularizer)
