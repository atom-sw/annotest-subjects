import os
from argparse import Namespace

import torch
from torch import optim

from model.BiDAF import BiDAF
from train import Trainer
from util.ema import EMA
from util.process_data import load_processed_json, DataSet, load_glove_weights

bug_data_dir = (os.path.expanduser('~') + "/annotest_subjects_data/BiDAF_PyTorch_b1/")


def generator_Trainer_init_model():
    args = Namespace()
    args.epoch = 1
    args.batch_size = 10
    args.lr = 0.5
    args.word_embd_dim = 100
    args.char_embd_dim = 8
    args.start_epoch = 0
    args.test = False
    args.resume = "~/checkpoints/Epoch-11.model"

    home = os.path.expanduser("~")
    train_json, train_shared_json = load_processed_json(bug_data_dir + './data/squad/data_train.json',
                                                        bug_data_dir + './data/squad/shared_train.json')
    test_json, test_shared_json = load_processed_json(bug_data_dir + './data/squad/data_test.json',
                                                      bug_data_dir + './data/squad/shared_test.json')
    train_data = DataSet(train_json, train_shared_json)
    test_data = DataSet(test_json, test_shared_json)
    # make *_to_index combining both training and test set
    w2i_train, c2i_train = train_data.get_word_index()
    w2i_test, c2i_test = test_data.get_word_index()
    word_vocab = sorted(list(set(list(w2i_train.keys()) + list(w2i_test.keys()))))
    w2i = {w: i for i, w in enumerate(word_vocab, 3)}  # 0:NULL, 1: UNK, 2: ENT
    char_vocab = sorted(list(set(list(c2i_train.keys()) + list(c2i_test.keys()))))
    c2i = {c: i for i, c in enumerate(char_vocab, 3)}
    NULL = "-NULL-"
    UNK = "-UNK-"
    ENT = "-ENT-"
    w2i[NULL] = 0
    w2i[UNK] = 1
    w2i[ENT] = 2
    c2i[NULL] = 0
    c2i[UNK] = 1
    c2i[ENT] = 2
    # load pre-trained GloVe
    glove_path = os.path.join(home, "data", "glove")
    glove = torch.from_numpy(load_glove_weights(glove_path, args.word_embd_dim, len(w2i), w2i)).type(torch.FloatTensor)
    # set up arguments
    args.word_vocab_size = len(w2i)
    args.char_vocab_size = len(c2i)
    args.pretrained = True
    args.pretrained_embd = glove
    ## for CNN
    args.filters = [[1, 5]]
    args.out_chs = 100

    model = BiDAF(args)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # check if resume
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model


def generator_Trainer_init_data():
    args = Namespace()
    args.epoch = 1
    args.batch_size = 10
    args.lr = 0.5
    args.word_embd_dim = 100
    args.char_embd_dim = 8
    args.start_epoch = 0
    args.test = False
    args.resume = "~/checkpoints/Epoch-11.model"

    train_json, train_shared_json = load_processed_json(bug_data_dir + './data/squad/data_train.json',
                                                        bug_data_dir + './data/squad/shared_train.json')
    train_data = DataSet(train_json, train_shared_json)
    return train_data


def generator_Trainer_init_w2i():
    args = Namespace()
    args.epoch = 1
    args.batch_size = 10
    args.lr = 0.5
    args.word_embd_dim = 100
    args.char_embd_dim = 8
    args.start_epoch = 0
    args.test = False
    args.resume = "~/checkpoints/Epoch-11.model"

    train_json, train_shared_json = load_processed_json(bug_data_dir + './data/squad/data_train.json',
                                                        bug_data_dir + './data/squad/shared_train.json')
    test_json, test_shared_json = load_processed_json(bug_data_dir + './data/squad/data_test.json',
                                                      bug_data_dir + './data/squad/shared_test.json')
    train_data = DataSet(train_json, train_shared_json)
    test_data = DataSet(test_json, test_shared_json)
    # make *_to_index combining both training and test set
    w2i_train, c2i_train = train_data.get_word_index()
    w2i_test, c2i_test = test_data.get_word_index()
    word_vocab = sorted(list(set(list(w2i_train.keys()) + list(w2i_test.keys()))))
    w2i = {w: i for i, w in enumerate(word_vocab, 3)}  # 0:NULL, 1: UNK, 2: ENT
    NULL = "-NULL-"
    UNK = "-UNK-"
    ENT = "-ENT-"
    w2i[NULL] = 0
    w2i[UNK] = 1
    w2i[ENT] = 2
    return w2i


def generator_Trainer_init_c2i():
    args = Namespace()
    args.epoch = 1
    args.batch_size = 10
    args.lr = 0.5
    args.word_embd_dim = 100
    args.char_embd_dim = 8
    args.start_epoch = 0
    args.test = False
    args.resume = "~/checkpoints/Epoch-11.model"

    train_json, train_shared_json = load_processed_json(bug_data_dir + './data/squad/data_train.json',
                                                        bug_data_dir + './data/squad/shared_train.json')
    test_json, test_shared_json = load_processed_json(bug_data_dir + './data/squad/data_test.json',
                                                      bug_data_dir + './data/squad/shared_test.json')
    train_data = DataSet(train_json, train_shared_json)
    test_data = DataSet(test_json, test_shared_json)
    # make *_to_index combining both training and test set
    w2i_train, c2i_train = train_data.get_word_index()
    w2i_test, c2i_test = test_data.get_word_index()
    word_vocab = sorted(list(set(list(w2i_train.keys()) + list(w2i_test.keys()))))
    w2i = {w: i for i, w in enumerate(word_vocab, 3)}  # 0:NULL, 1: UNK, 2: ENT
    char_vocab = sorted(list(set(list(c2i_train.keys()) + list(c2i_test.keys()))))
    c2i = {c: i for i, c in enumerate(char_vocab, 3)}
    NULL = "-NULL-"
    UNK = "-UNK-"
    ENT = "-ENT-"
    w2i[NULL] = 0
    w2i[UNK] = 1
    w2i[ENT] = 2
    c2i[NULL] = 0
    c2i[UNK] = 1
    c2i[ENT] = 2
    return c2i


def generator_Trainer_init_optimizer():
    args = Namespace()
    args.epoch = 1
    args.batch_size = 10
    args.lr = 0.5
    args.word_embd_dim = 100
    args.char_embd_dim = 8
    args.start_epoch = 0
    args.test = False
    args.resume = "~/checkpoints/Epoch-11.model"

    # loading data
    home = os.path.expanduser("~")
    train_json, train_shared_json = load_processed_json(bug_data_dir + './data/squad/data_train.json',
                                                        bug_data_dir + './data/squad/shared_train.json')
    test_json, test_shared_json = load_processed_json(bug_data_dir + './data/squad/data_test.json',
                                                      bug_data_dir + './data/squad/shared_test.json')
    train_data = DataSet(train_json, train_shared_json)
    test_data = DataSet(test_json, test_shared_json)

    # make *_to_index combining both training and test set
    w2i_train, c2i_train = train_data.get_word_index()
    w2i_test, c2i_test = test_data.get_word_index()
    word_vocab = sorted(list(set(list(w2i_train.keys()) + list(w2i_test.keys()))))
    w2i = {w: i for i, w in enumerate(word_vocab, 3)}  # 0:NULL, 1: UNK, 2: ENT
    char_vocab = sorted(list(set(list(c2i_train.keys()) + list(c2i_test.keys()))))
    c2i = {c: i for i, c in enumerate(char_vocab, 3)}
    NULL = "-NULL-"
    UNK = "-UNK-"
    ENT = "-ENT-"
    w2i[NULL] = 0
    w2i[UNK] = 1
    w2i[ENT] = 2
    c2i[NULL] = 0
    c2i[UNK] = 1
    c2i[ENT] = 2

    # load pre-trained GloVe
    glove_path = os.path.join(home, "data", "glove")
    glove = torch.from_numpy(load_glove_weights(glove_path, args.word_embd_dim, len(w2i), w2i)).type(torch.FloatTensor)

    # set up arguments
    args.word_vocab_size = len(w2i)
    args.char_vocab_size = len(c2i)
    args.pretrained = True
    args.pretrained_embd = glove
    ## for CNN
    args.filters = [[1, 5]]
    args.out_chs = 100

    model = BiDAF(args)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # check if resume
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return optimizer


def generator_Trainer_init_ema():
    args = Namespace()
    args.epoch = 1
    args.batch_size = 10
    args.lr = 0.5
    args.word_embd_dim = 100
    args.char_embd_dim = 8
    args.start_epoch = 0
    args.test = False
    args.resume = "~/checkpoints/Epoch-11.model"

    home = os.path.expanduser("~")
    train_json, train_shared_json = load_processed_json(bug_data_dir + './data/squad/data_train.json',
                                                        bug_data_dir + './data/squad/shared_train.json')
    test_json, test_shared_json = load_processed_json(bug_data_dir + './data/squad/data_test.json',
                                                      bug_data_dir + './data/squad/shared_test.json')
    train_data = DataSet(train_json, train_shared_json)
    test_data = DataSet(test_json, test_shared_json)

    # make *_to_index combining both training and test set
    w2i_train, c2i_train = train_data.get_word_index()
    w2i_test, c2i_test = test_data.get_word_index()
    word_vocab = sorted(list(set(list(w2i_train.keys()) + list(w2i_test.keys()))))
    w2i = {w: i for i, w in enumerate(word_vocab, 3)}  # 0:NULL, 1: UNK, 2: ENT
    char_vocab = sorted(list(set(list(c2i_train.keys()) + list(c2i_test.keys()))))
    c2i = {c: i for i, c in enumerate(char_vocab, 3)}
    NULL = "-NULL-"
    UNK = "-UNK-"
    ENT = "-ENT-"
    w2i[NULL] = 0
    w2i[UNK] = 1
    w2i[ENT] = 2
    c2i[NULL] = 0
    c2i[UNK] = 1
    c2i[ENT] = 2

    # load pre-trained GloVe
    glove_path = os.path.join(home, "data", "glove")
    glove = torch.from_numpy(load_glove_weights(glove_path, args.word_embd_dim, len(w2i), w2i)).type(torch.FloatTensor)

    # set up arguments
    args.word_vocab_size = len(w2i)
    args.char_vocab_size = len(c2i)
    args.pretrained = True
    args.pretrained_embd = glove
    ## for CNN
    args.filters = [[1, 5]]
    args.out_chs = 100

    model = BiDAF(args)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # check if resume
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # exponential moving average
    ema = EMA(0.999)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            ema.register(name, parameter.data)
    return ema


def test_failing():
    cc_epoch = 1
    cc_starting_epoch = 0
    cc_batch_size = 10
    cc_model = generator_Trainer_init_model()
    cc_data = generator_Trainer_init_data()
    cc_w2i = generator_Trainer_init_w2i()
    cc_c2i = generator_Trainer_init_c2i()
    cc_optimizer = generator_Trainer_init_optimizer()
    cc_ema = generator_Trainer_init_ema()
    obj = Trainer(cc_model, cc_data, cc_w2i, cc_c2i, cc_optimizer,
                  cc_ema, cc_epoch, cc_starting_epoch, cc_batch_size)
    obj.train()
