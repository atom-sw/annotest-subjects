from __future__ import print_function
import unittest
import hypothesis as hy
import hypothesis.strategies as st
from go import main, parse_arg


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(noArgCall=st.just(None))
    @hy.settings(deadline=None)
    def test_parse_arg(self, noArgCall):
        parse_arg()

    @hy.given(nb_epoch=st.just(1), data_augmentation=st.one_of(st.
        sampled_from([True, False]), st.just(False)), noise=st.one_of(st.
        sampled_from([True, False]), st.just(False)), maxout=st.one_of(st.
        sampled_from([True, False]), st.just(False)), dropout=st.one_of(st.
        sampled_from([True, False]), st.just(True)), l1_reg=st.one_of(st.
        sampled_from([True, False]), st.just(False)), l2_reg=st.one_of(st.
        sampled_from([True, False]), st.just(True)), max_pooling=st.one_of(
        st.sampled_from([True, False]), st.just(True)), deep=st.one_of(st.
        sampled_from([True, False]), st.just(False)), noise_sigma=st.one_of
        (st.floats(min_value=0, max_value=1, allow_nan=None, allow_infinity
        =None, width=64, exclude_min=True, exclude_max=True), st.just(0.01)
        ), weight_constraint=st.one_of(st.sampled_from([True, False]), st.
        just(True)))
    @hy.settings(deadline=None, suppress_health_check=[hy.HealthCheck.
        filter_too_much, hy.HealthCheck.too_slow])
    def test_main(self, nb_epoch, data_augmentation, noise, maxout, dropout,
        l1_reg, l2_reg, max_pooling, deep, noise_sigma, weight_constraint):
        print("NEW_TESTING")
        hy.assume(not (l1_reg and l2_reg))
        hy.assume(not (weight_constraint and l2_reg))
        main(nb_epoch, data_augmentation, noise, maxout, dropout, l1_reg,
            l2_reg, max_pooling, deep, noise_sigma, weight_constraint)
