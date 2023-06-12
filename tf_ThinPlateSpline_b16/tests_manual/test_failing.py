import os

import numpy as np
import tensorflow as tf

from ThinPlateSpline2 import ThinPlateSpline2

Path_to_file = (os.path.expanduser('~') + "/annotest_subjects_data/tf_ThinPlateSpline_b16/original.png")


def generator_ThinPlateSpline2_U():
    from PIL import Image
    img = np.array(Image.open(Path_to_file))
    out_size = list(img.shape)
    shape = [1] + out_size + [1]

    t_img = tf.constant(img.reshape(shape), dtype=tf.float32)

    return t_img


def generator_ThinPlateSpline2_source():
    s_ = np.array([  # source position
        [-0.5, -0.5],
        [0.5, -0.5],
        [-0.5, 0.5],
        [0.5, 0.5]])

    s = tf.constant(s_.reshape([1, 4, 2]), dtype=tf.float32)

    return s


def generator_ThinPlateSpline2_target():
    t_ = np.array([  # target position
        [-0.3, -0.3],
        [0.3, -0.3],
        [-0.3, 0.3],
        [0.3, 0.3]])

    t = tf.constant(t_.reshape([1, 4, 2]), dtype=tf.float32)

    return t


def test_failing():
    U = generator_ThinPlateSpline2_U()
    source = generator_ThinPlateSpline2_source()
    target = generator_ThinPlateSpline2_target()
    out_size = [256, 263]
    ThinPlateSpline2(U, source, target, out_size)
