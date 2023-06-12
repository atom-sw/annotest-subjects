'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.

to simply test the code, run THEANO_FLAGS=device=gpu,floatX=float32 python cifar10_cnn.py -e 1
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GaussianNoise, MaxoutDense
from keras.regularizers import l1, l2, activity_l1, activity_l2, l1l2
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optparse
import sys
import os
import data_loader
import time
import numpy

batch_size = 32
nb_classes = 10
nb_epoch = 100
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3
sigma = 0.01
l1_weight = 0.01
l2_weight = 0.05
l1_activation = 0.05
l2_activation = 0.05

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/CIFAR-10/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

output_path = "outputs_good_model/"
# specify the purpose of tge experiment on the directory name
output_directory = "l2_weight_constraint/"

def parse_arg():

    parser = optparse.OptionParser('usage%prog  [-e epoch] [-a data_augmentation] [-n noise] [-m maxout] [-d dropout] [-l l1] [-r l2] [-p max_pooling] [-x deep] [-o noise_sigma] [-w weight_constraint]')
    parser.add_option('-e', dest='epoch')
    parser.add_option('-a', dest='data_augmentation')
    parser.add_option('-n', dest='noise')
    parser.add_option('-m', dest='maxout')
    parser.add_option('-d', dest='dropout')
    parser.add_option('-l', dest='l1')
    parser.add_option('-r', dest='l2')
    parser.add_option('-p', dest='max_pooling')
    parser.add_option('-x', dest='deep')
    parser.add_option('-o', dest='noise_sigma')
    parser.add_option('-w', dest='weight_constraint')

    (options, args) = parser.parse_args()
    return options

def main(nb_epoch=50, data_augmentation=False, noise=False, maxout=False, dropout=True, l1_reg=False, l2_reg=True, max_pooling=True, deep=False, noise_sigma=0.01, weight_constraint=True):
    # l1 and l2 regularization shouldn't be true in the same time
    if l1_reg and l2_reg:
        print("No need to run l1 and l2 regularization in the same time")
        quit()
    if weight_constraint and l2_reg:
        print("No need to run weight_constraint and l2 regularization in the same time")
        quit()
    # print settings for this experiment
    print("number of epoch: {0}".format(nb_epoch))
    print("data augmentation: {0}".format(data_augmentation))
    print("noise: {0}".format(noise))
    print("sigma: {0}".format(sigma))
    print("maxout: {0}".format(maxout))
    print("dropout: {0}".format(dropout))
    print("l1: {0}".format(l1_reg))
    print("l2: {0}".format(l2_reg))
    print("max_pooling: {0}".format(max_pooling))
    print("deep: {0}".format(deep))
    print("weight_constraint: {0}".format(weight_constraint))
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # split the validation dataset
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    # X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    # X_valid /= 255
    X_test /= 255

    ##### try loading data using data_loader.py ####
    # data_loader.download_and_extract(data_path, data_url)
    # class_names = data_loader.load_class_names()
    # print(class_names)
    # images_train, cls_train, labels_train = data_loader.load_training_data()
    # images_test, cls_test, labels_test = data_loader.load_test_data()
    # X_train, Y_train = images_train, labels_train
    # X_test, Y_test = images_test, labels_test
    # X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(X_train)))
    # print("- Validation-set:\t\t{}".format(len(X_valid)))
    print("- Test-set:\t\t{}".format(len(X_test)))

    # Create the model
    model = Sequential()
    if noise:
        model.add(GaussianNoise(noise_sigma, input_shape=(32, 3, 3)))  # repo_bug
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', border_mode='same'))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    if dropout:

        model.add(Dropout(0.2))
    if l2_reg:
        model.add(Dense(1024, activation='relu', W_regularizer=l2(l2_weight)))
    elif weight_constraint:
        model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    else:
        model.add(Dense(1024, activation='relu'))

    if dropout:
        model.add(Dropout(0.2))
    if maxout:
        model.add(MaxoutDense(512, nb_feature=4, init='glorot_uniform'))
    else:
        model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='softmax'))
    # Compile model
    lrate = 0.01
    decay = lrate/nb_epoch
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())




    start_time = time.time()
    if not data_augmentation:
        # his = model.fit(X_train, Y_train,
        #           batch_size=batch_size,
        #           nb_epoch=nb_epoch,
        #           validation_data=(X_valid, Y_valid),
        #           shuffle=True)
        # numpy.random.seed(seed)
        his = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=nb_epoch, batch_size=64)
# Final evaluation of the model
    else:
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=True,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        his = model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test))

    # evaluate our model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print('training time', time.time() - start_time)

    file_path = os.path.join(output_path, output_directory)
    print("outputs should be store at %s" % file_path)
    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        print("creat output directory fro storing output")
        # Check if the download directory exists, otherwise create it.
        os.makedirs(file_path)
    # wirte test accuracy to a file
    output_file_name = os.path.join(file_path, 'train_val_loss_with_dropout__{0}_data_augmentation_{1}_noise_{2}_sigma{12}_maxout_{3}_dropout_{4}_l1_{5}_l2_{6}_sigma_{7}_l1weight_{8}_l2weight_{9}_max_pooling_{10}_deep_{11}.txt'.format(nb_epoch, data_augmentation, noise, maxout, dropout, l1_reg, l2_reg, sigma, l1_weight, l2_weight, max_pooling, deep, sigma))
    print("save file at {}".format(output_file_name)    )
    with open(output_file_name, "w") as text_file:
        text_file.write('Test score: {}\n'.format(score[0]))
        text_file.write('Test accuracy: {}\n'.format(score[1]))
        text_file.write('Training time: {}\n'.format(time.time() - start_time))
    text_file.close()

    # visualize training history
    train_loss = his.history['loss']
    val_loss = his.history['val_loss']
    plt.plot(range(1, len(train_loss)+1), train_loss, color='blue', label='train loss')
    plt.plot(range(1, len(val_loss)+1), val_loss, color='red', label='val loss')
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.xlabel('#epoch')
    plt.ylabel('loss')

    output_fig_name = os.path.join(file_path, 'train_val_loss_with_dropout__{0}_data_augmentation_{1}_noise_{2}_sigma{12}_maxout_{3}_dropout_{4}_l1_{5}_l2_{6}_sigma_{7}_l1weight_{8}_l2weight_{9}_max_pooling_{10}_deep_{11}.png'.format(nb_epoch, data_augmentation, noise, maxout, dropout, l1_reg, l2_reg, sigma, l1_weight, l2_weight, max_pooling, deep, sigma))
    plt.savefig(output_fig_name, dpi=300)
    plt.show()

if __name__ == '__main__':
    opts = parse_arg()
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['nb_epoch'] = int(opts.epoch)
        kwargs['data_augmentation'] = True if opts.data_augmentation == 'True' else False
        kwargs['noise'] = True if opts.noise == 'True' else False
        kwargs['maxout'] = True if opts.maxout == 'True' else False
        kwargs['dropout'] = True if opts.dropout == 'True' else False
        kwargs['l1_reg'] = True if opts.l1 == 'True' else False
        kwargs['l2_reg'] = True if opts.l2 == 'True' else False
        kwargs['max_pooling'] = True if opts.max_pooling == 'True' else False
        kwargs['deep'] = True if opts.deep == 'True' else False
        kwargs['noise_sigma'] = float(opts.noise_sigma)
        kwargs['weight_constraint'] = True if opts.weight_constraint == 'True' else False

    main(**kwargs)
