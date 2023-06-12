import numpy as np

from dataset_parser.generator import get_result_map


def test_failing():
    input_shape = (4, 256, 512, 1)
    d_type = np.dtype('float32')
    y_img = np.zeros(shape=input_shape, dtype=d_type)
    get_result_map(y_img)
