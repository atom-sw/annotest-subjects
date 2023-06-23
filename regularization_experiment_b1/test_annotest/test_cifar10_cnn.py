from __future__ import print_function
import unittest
import hypothesis as hy
import hypothesis.strategies as st
from cifar10_cnn import main, parse_arg


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_parse_arg(self, noArgCall):
        parse_arg()

    @hy.given(nb_epoch=st.just(1), data_augmentation=st.just(True), noise=
        st.just(True), maxout=st.just(True), dropout=st.just(True), l1_reg=
        st.just(False), l2_reg=st.just(True), max_pooling=st.just(True),
        deep=st.just(False), noise_sigma=st.just(0.01))
    @hy.settings(deadline=None)
    def test_main(self, nb_epoch, data_augmentation, noise, maxout, dropout,
        l1_reg, l2_reg, max_pooling, deep, noise_sigma):
        main(nb_epoch, data_augmentation, noise, maxout, dropout, l1_reg,
            l2_reg, max_pooling, deep, noise_sigma)
