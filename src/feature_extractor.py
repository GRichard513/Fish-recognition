import numpy as np
import time


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y_df):
        pass

    def transform(self, train_data):
        start_time = time.time()

        # read and normalize data
        print('Convert to numpy...')
        train_data = np.array(train_data, dtype=np.uint8)

        print('Reshape...')
        train_data = train_data.transpose((0, 3, 1, 2))

        print('Convert to float...')
        train_data = train_data.astype('float32')
        train_data = train_data / 255

        print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
        return train_data
