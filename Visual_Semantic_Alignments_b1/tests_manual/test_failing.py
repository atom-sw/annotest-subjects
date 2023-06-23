from model_zoo import full_model


def test_failing():
    vocab_size = 900
    max_len = 10
    embed_size = 200
    nb_hidden_states = 512
    nb_regs = 10
    nb_feats = 4096
    common_dim = 300
    batch_size = 32
    lr = 0.01

    full_model(vocab_size, max_len, embed_size, nb_hidden_states,
               nb_regs, nb_feats, common_dim, batch_size, lr)
