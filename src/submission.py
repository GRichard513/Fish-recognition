import numpy as np
np.random.seed(1989)
import matplotlib as mlt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import glob
from scipy.misc import *
import skimage.measure as sm
import multiprocessing
import random
from subprocess import check_output
import os
import time
import cv2
import warnings
warnings.filterwarnings("ignore")
new_style = {'grid': False}
plt.rc('axes', **new_style)

# sklearn
import sklearn as skt
from sklearn import cluster
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

# keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras import __version__ as keras_version

import load_data
import feature_extractor
import model
import submit

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))

    # Input data files are available in the "../input/" directory.
    # Load data
    train_data, train_target, train_id = load_data.load_train()
    test_data, test_id = load_data.load_test()
    train_target = np_utils.to_categorical(train_target, 8)

    # define number of fold for cross validation
    num_folds = 3

    # features exctraction
    fe = feature_extractor.FeatureExtractor()
    fe.fit(train_data, train_target)
    train_data = fe.transform(train_data)
    test_data = fe.transform(test_data)

    # model initialization
    model = model.model()

    # input image dimensions
    nfolds=10
    batch_size = 32
    nb_epoch = 8
    random_state = 51
    first_rl = 96

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        # model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = '_' + str(np.round(score,3)) + '_flds_' + str(nfolds) + '_eps_' + str(nb_epoch) + '_fl_' + str(first_rl)

    # prediction submission
    batch_size = 24
    num_fold = 0
    yfull_test = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = submit.merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    submit.create_submission(test_res, test_id, info_string)
