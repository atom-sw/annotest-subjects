from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim
import os.path
import numpy as np
import random
from tqdm import tqdm
import datetime

from model.BiDAF import BiDAF
from util.ema import EMA
from util.process_data import get_idx_tensor, DataSet, load_processed_json, load_glove_weights

from annotest import an_language as an

bug_data_dir = (os.path.expanduser('~') + "/annotest_subjects_data/BiDAF_PyTorch_b1/")


@an.exclude()
@an.generator()
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


@an.exclude()
@an.generator()
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


@an.exclude()
@an.generator()
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


@an.exclude()
@an.generator()
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


@an.exclude()
@an.generator()
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


@an.exclude()
@an.generator()
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


class Trainer(object):

	@an.arg("model", an.obj(generator_Trainer_init_model))
	@an.arg("data", an.obj(generator_Trainer_init_data))
	@an.arg("w2i", an.obj(generator_Trainer_init_w2i))
	@an.arg("c2i", an.obj(generator_Trainer_init_c2i))
	@an.arg("optimizer", an.obj(generator_Trainer_init_optimizer))
	@an.arg("ema", an.obj(generator_Trainer_init_ema))
	@an.arg("epoch", an.sampled([1]))
	@an.arg("starting_epoch", an.sampled([0]))
	@an.arg("batch_size", an.sampled([10]))
	def __init__(self, model, data, w2i, c2i, optimizer, ema, epoch, starting_epoch, batch_size):
		self.model = model
		self.data = data
		self.optimizer = optimizer
		self.ema = ema
		self.num_epoch = epoch
		self.start_from = starting_epoch
		self.word_to_index = w2i
		self.char_to_index = c2i
		self.batch_size = batch_size

	def train(self):
		self.model.train()
		for epoch in tqdm(range(self.start_from, self.num_epoch)):
			print(">>>>>>>>>>>>>Processing epoch:", epoch)
			batches = self.data.get_batches(self.batch_size, shuffle = True)
			p1_EM, p2_EM = 0, 0
			num_data_processed = 0
			for i, batch in enumerate(batches):
				# each batch consists of tuples of (ctx_word_lv, ctx_char_lv, query_word_lv, query_char_lv, answer)
				max_ctx_sent_len = max([len(tupl[0]) for tupl in batch])
				max_ctx_word_len = max([len(word) for tupl in batch for word in tupl[1]])
				max_query_sent_len = max([len(tupl[2]) for tupl in batch])
				max_query_word_len = max([len(word) for tupl in batch for word in tupl[3]])

				# padding to make batch equal lengthy
				ctx_word_lv, ctx_char_lv, query_word_lv, query_char_lv, answer = get_idx_tensor(batch, 
																								self.word_to_index, 
																								self.char_to_index, 
																								max_ctx_sent_len, 
																								max_ctx_word_len,
																								max_query_sent_len,
																								max_query_word_len)
				# forward
				ans_start = answer[:, 0]
				ans_end = answer[:, 1] - 1
				p1, p2 = self.model(ctx_word_lv, ctx_char_lv, query_word_lv, query_char_lv)
				loss_p1 = nn.NLLLoss(p1, ans_start)  # repo_bug 1/3
				loss_p2 = nn. NLLLoss(p2, ans_end)  # repo_bug 2/3
				loss = torch.add(loss_p1, loss_p2)  # repo_bug 3/3
				p1_EM += torch.sum(ans_start == torch.max(p1, 1)[1]).item()
				p2_EM += torch.sum(ans_start == torch.max(p2, 1)[1]).item()
				num_data_processed += len(batch)

				# print training process
				if (i + 1) % 50 == 0:
					loss_info = "[{}] Epoch {} completed {:.1f}%, loss_p1: {:.3f}, loss_p2: {:.3f}"
					print(loss_info.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
										   epoch, 100 * i / len(batches),
										   loss_p1.data[0], loss_p2.data[0]))

					EM_info = "p1 EM: {:.3f}% ({}/{}), p2 EM: {:.3f}% ({}/{})"
					print(EM_info.format(100 * p1_EM / num_data_processed, p1_EM, num_data_processed,
										 100 * p2_EM / num_data_processed, p2_EM, num_data_processed))

				# backward
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				for name, parameter in model.named_parameters():
					if parameter.requires_grad:
						parameter.data = ema(name, parameter,data)

			# end of one epoch
			print(">>>>>>>>>>>>>Epoch", epoch, "result")
			print('p1 EM: {:.3f}, p2 EM: {:.3f}'.format(100 * p1_EM / num_data_processed,
													    100 * p2_EM / num_data_processed))
			filename = '{}/Epoch-{}.model'.format('~/checkpoints', epoch)
			torch.save({'epoch': epoch + 1, 
						'state_dict': model.state_dict(), 
						'optimizer': optimizer.state_dict(),
						'p1_EM': 100 * p1_EM / num_data_processed,
						'p2_EM': 100 * p2_EM / num_data_processed
						}, filename=filename)


