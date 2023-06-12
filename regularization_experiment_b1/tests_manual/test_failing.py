from go import main


def test_failing():
    nb_epoch = 1
    data_augmentation = False
    noise = True
    maxout = False
    dropout = True
    l1_reg = False
    l2_reg = False
    max_pooling = True
    deep = False
    noise_sigma = 0.01
    weight_constraint = True

    main(nb_epoch, data_augmentation, noise, maxout, dropout, l1_reg,
         l2_reg, max_pooling, deep, noise_sigma, weight_constraint)
