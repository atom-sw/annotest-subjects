import unittest
import hypothesis as hy
import hypothesis.strategies as st
from train import Trainer, generator_Trainer_init_c2i, generator_Trainer_init_data, generator_Trainer_init_ema, generator_Trainer_init_model, generator_Trainer_init_optimizer, generator_Trainer_init_w2i


class Test_Trainer(unittest.TestCase):

    @hy.given(epoch=st.sampled_from([1]), starting_epoch=st.sampled_from([0
        ]), batch_size=st.sampled_from([10]), st_for_data=st.data())
    @hy.settings(deadline=None)
    def test_init(self, epoch, starting_epoch, batch_size, st_for_data):
        model = generator_Trainer_init_model()
        data = generator_Trainer_init_data()
        w2i = generator_Trainer_init_w2i()
        c2i = generator_Trainer_init_c2i()
        optimizer = generator_Trainer_init_optimizer()
        ema = generator_Trainer_init_ema()
        Trainer(model, data, w2i, c2i, optimizer, ema, epoch,
            starting_epoch, batch_size)

    @hy.given(st_for_data=st.data())
    @hy.settings(deadline=None)
    def test_train(self, st_for_data):
        cc_epoch = st_for_data.draw(st.sampled_from([1]))
        cc_starting_epoch = st_for_data.draw(st.sampled_from([0]))
        cc_batch_size = st_for_data.draw(st.sampled_from([10]))
        cc_model = generator_Trainer_init_model()
        cc_data = generator_Trainer_init_data()
        cc_w2i = generator_Trainer_init_w2i()
        cc_c2i = generator_Trainer_init_c2i()
        cc_optimizer = generator_Trainer_init_optimizer()
        cc_ema = generator_Trainer_init_ema()
        obj = Trainer(cc_model, cc_data, cc_w2i, cc_c2i, cc_optimizer,
            cc_ema, cc_epoch, cc_starting_epoch, cc_batch_size)
        obj.train()
