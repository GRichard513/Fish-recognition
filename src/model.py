from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras import __version__ as keras_version

class model():
    def __init__(self):
        self.model = Sequential()
        self.model.add(ZeroPadding2D((1, 1), input_shape=(3, 64, 64), dim_ordering='th'))
        self.model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
        self.model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
        self.model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(96, activation='relu',init='he_uniform'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(24, activation='relu',init='he_uniform'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(8, activation='softmax'))

        sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.88, nesterov=False)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')

        #return model

    def fit(self, X, y, batch_size, nb_epoch, shuffle, verbose, validation_data, callbacks):
        return self.model.fit(X, y, batch_size=batch_size, verbose=verbose, nb_epoch=nb_epoch, shuffle=shuffle, validation_data=validation_data,callbacks=callbacks)

    def predict(self, X,  batch_size, verbose):
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
