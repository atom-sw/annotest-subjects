import unittest
import hypothesis as hy
import hypothesis.strategies as st
import os
from car_recognition_aws.matlabtester import model


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(learningRate=st.floats(min_value=0, max_value=0.002,
        allow_nan=None, allow_infinity=None, width=64, exclude_min=True,
        exclude_max=False), optimazerLastLayer=st.sampled_from(['ADAGRAD',
        'ADADELTA', 'ADAM', 'RMSPROP', 'MOM']), noOfEpochs=st.sampled_from(
        [1]), batchSize=st.sampled_from([64]), savedModelName=st.
        sampled_from(['SomeName']), srcImagesDir=st.sampled_from([os.path.
        expanduser('~') +
        '/annotest_subjects_data/car_recognition_b1/car_ims']), labelsFile=
        st.sampled_from([os.path.expanduser('~') +
        '/annotest_subjects_data/car_recognition_b1/cars_annos.mat']))
    @hy.settings(deadline=None)
    def test_model(self, learningRate, optimazerLastLayer, noOfEpochs,
        batchSize, savedModelName, srcImagesDir, labelsFile):
        model(learningRate, optimazerLastLayer, noOfEpochs, batchSize,
            savedModelName, srcImagesDir, labelsFile)
