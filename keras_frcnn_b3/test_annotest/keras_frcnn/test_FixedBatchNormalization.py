import unittest
import hypothesis as hy
import hypothesis.strategies as st
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization


class Test_FixedBatchNormalization(unittest.TestCase):

    @hy.given(epsilon=st.just(0.001), axis=st.one_of(st.sampled_from([-1, 1,
        3]), st.just(-1)), weights=st.just(None), beta_init=st.just('zero'),
        gamma_init=st.just('one'), gamma_regularizer=st.just(None),
        beta_regularizer=st.just(None))
    @hy.settings(deadline=None)
    def test_init(self, epsilon, axis, weights, beta_init, gamma_init,
        gamma_regularizer, beta_regularizer):
        FixedBatchNormalization(epsilon, axis, weights, beta_init,
            gamma_init, gamma_regularizer, beta_regularizer)

    @hy.given(st_for_data=st.data())
    @hy.settings(deadline=None)
    def test_get_config(self, st_for_data):
        cc_epsilon = st_for_data.draw(st.just(0.001))
        cc_axis = st_for_data.draw(st.one_of(st.sampled_from([-1, 1, 3]),
            st.just(-1)))
        cc_weights = st_for_data.draw(st.just(None))
        cc_beta_init = st_for_data.draw(st.just('zero'))
        cc_gamma_init = st_for_data.draw(st.just('one'))
        cc_gamma_regularizer = st_for_data.draw(st.just(None))
        cc_beta_regularizer = st_for_data.draw(st.just(None))
        obj = FixedBatchNormalization(cc_epsilon, cc_axis, cc_weights,
            cc_beta_init, cc_gamma_init, cc_gamma_regularizer,
            cc_beta_regularizer)
        obj.get_config()
