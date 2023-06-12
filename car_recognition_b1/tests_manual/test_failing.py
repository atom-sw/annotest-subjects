import os

from car_recognition_aws.matlabtester import model


def test_failing():
    bug_id_data_directory_path = (os.path.expanduser('~') +
                                  "/annotest_subjects_data/car_recognition_b1")
    learningRate = 5e-324
    optimazerLastLayer = 'ADAGRAD'
    noOfEpochs = 1
    batchSize = 64
    savedModelName = 'SomeName'
    srcImagesDir = bug_id_data_directory_path + '/car_ims'
    labelsFile = bug_id_data_directory_path + '/cars_annos.mat'
    model(learningRate, optimazerLastLayer, noOfEpochs, batchSize,
          savedModelName, srcImagesDir, labelsFile)
