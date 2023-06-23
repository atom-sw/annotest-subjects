from __future__ import print_function
import unittest
import hypothesis as hy
import hypothesis.strategies as st
from model_zoo import full_model


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(vocab_size=st.integers(min_value=900, max_value=1100),
        max_len=st.integers(min_value=10, max_value=100), embed_size=st.
        integers(min_value=200, max_value=400), nb_hidden_states=st.
        sampled_from([512]), nb_regs=st.sampled_from([10]), nb_feats=st.
        just(4096), common_dim=st.just(300), batch_size=st.just(32), lr=st.
        just(0.01))
    @hy.settings(deadline=None)
    def test_full_model(self, vocab_size, max_len, embed_size,
        nb_hidden_states, nb_regs, nb_feats, common_dim, batch_size, lr):
        full_model(vocab_size, max_len, embed_size, nb_hidden_states,
            nb_regs, nb_feats, common_dim, batch_size, lr)
