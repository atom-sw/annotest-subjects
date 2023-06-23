import unittest
import hypothesis as hy
import hypothesis.strategies as st
from __future__ import print_function
from __future__ import absolute_import
from keras_frcnn.resnet import nn_base


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(input_tensor=st.just(None), trainable=st.just(False))
    @hy.settings(deadline=None)
    def test_nn_base(self, input_tensor, trainable):
        nn_base(input_tensor, trainable)
