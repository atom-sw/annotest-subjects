import unittest
import hypothesis as hy
import hypothesis.strategies as st
from densenet import DenseNet
from hypothesis.strategies._internal.utils import defines_strategy


@defines_strategy()
def integer_lists_an(min_len=1, max_len=None, min_value=1, max_value=None):
    if max_len is None:
        max_len = min_len + 2
    if max_value is None:
        max_value = min_value + 5
    return st.lists(st.integers(min_value, max_value), min_size=min_len, max_size=max_len)


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(input_shape=st.one_of(st.tuples(st.integers(min_value=20,
        max_value=70), st.integers(min_value=20, max_value=70), st.integers
        (min_value=1, max_value=3)), st.just(None)), dense_blocks=st.one_of
        (st.integers(min_value=1, max_value=5), st.just(3)), dense_layers=
        st.one_of(st.one_of(st.sampled_from([-1]), st.integers(min_value=1,
        max_value=5), integer_lists_an(min_len=2, max_len=5, min_value=2,
        max_value=5)), st.just(-1)), growth_rate=st.one_of(st.integers(
        min_value=1, max_value=20), st.just(12)), nb_classes=st.one_of(st.
        integers(min_value=2, max_value=22), st.just(None)), dropout_rate=
        st.one_of(st.floats(min_value=0, max_value=1, allow_nan=None,
        allow_infinity=None, width=64, exclude_min=True, exclude_max=True),
        st.just(None)), bottleneck=st.one_of(st.sampled_from([True, False]),
        st.just(False)), compression=st.one_of(st.floats(min_value=0,
        max_value=1, allow_nan=None, allow_infinity=None, width=64,
        exclude_min=True, exclude_max=False), st.just(1.0)), weight_decay=
        st.one_of(st.floats(min_value=0.0001, max_value=0.01, allow_nan=
        None, allow_infinity=None, width=64, exclude_min=False, exclude_max
        =False), st.just(0.0001)), depth=st.one_of(st.integers(min_value=10,
        max_value=100), st.just(40)))
    @hy.settings(deadline=None, suppress_health_check=[hy.HealthCheck.
        filter_too_much, hy.HealthCheck.too_slow])
    def test_DenseNet(self, input_shape, dense_blocks, dense_layers,
        growth_rate, nb_classes, dropout_rate, bottleneck, compression,
        weight_decay, depth):
        hy.assume(not nb_classes == None)
        hy.assume(not (compression <= 0.0 or compression > 1.0))
        hy.assume(not (type(dense_layers) is list and len(dense_layers) !=
            dense_blocks))
        hy.assume(not input_shape is None)
        DenseNet(input_shape, dense_blocks, dense_layers, growth_rate,
            nb_classes, dropout_rate, bottleneck, compression, weight_decay,
            depth)