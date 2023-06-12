import os

import tensorflow as tf
import numpy as np
from PIL import Image
from ThinPlateSpline import ThinPlateSpline

Path_to_file = (os.path.expanduser('~') + "/annotest_subjects_data/tf_ThinPlateSpline_b2/original.png")


def generator_ThinPlateSpline_U():
    img = np.array(Image.open(Path_to_file))
    out_size = list(img.shape)
    shape = [1] + out_size + [1]

    t_img = tf.constant(img.reshape(shape), dtype=tf.float32)

    return t_img


def generator_ThinPlateSpline_coord():
    p = np.array([
        [-0.5, -0.5],
        [0.5, -0.5],
        [-0.5, 0.5],
        [0.5, 0.5]])

    p = tf.constant(p.reshape([1, 4, 2]), dtype=tf.float32)

    return p


def generator_ThinPlateSpline_vector():
    v = np.array([
        [0.2, 0.2],
        [0.4, 0.4],
        [0.6, 0.6],
        [0.8, 0.8]])

    v = tf.constant(v.reshape([1, 4, 2]), dtype=tf.float32)

    return v


def generator_ThinPlateSpline_out_size():
    img = np.array(Image.open(Path_to_file))
    out_size = list(img.shape)

    return out_size


def test_failing():
    U = generator_ThinPlateSpline_U()
    coord = generator_ThinPlateSpline_coord()
    vector = generator_ThinPlateSpline_vector()
    out_size = generator_ThinPlateSpline_out_size()

    ThinPlateSpline(U, coord, vector, out_size)
