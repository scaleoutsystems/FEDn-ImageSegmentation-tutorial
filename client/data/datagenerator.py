import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=32, dim=(256, 256), n_channels=4,
                 shuffle=True, original_size=(256, 256), binary_class=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size

        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.original_size = original_size
        self.binary_class = binary_class

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.original_size, self.n_channels))
        if self.binary_class:
            y_channel = 2
        else:
            y_channel = 4
        y = np.empty((self.batch_size, *self.original_size))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            im_frame1 = np.expand_dims(np.array(Image.open(ID + '_flair.png')), -1)
            im_frame2 = np.expand_dims(np.array(Image.open(ID + '_T1.png')), -1)
            im_frame3 = np.pad(np.expand_dims(np.array(Image.open(ID + '_T1ce.png')), -1), ((8, 8), (8, 8), (0, 0)))
            im_frame4 = np.expand_dims(np.array(Image.open(ID + '_T2.png')), -1)
            X[i,] = np.concatenate((im_frame1, im_frame2, im_frame3, im_frame4), -1) / 2000

            # Store class
            y[i] = np.int32(np.array(Image.open(ID + '_seg.png')) / 10000)

        if self.binary_class:
            y = np.sign(y)
            y = to_categorical(y, num_classes=2)
        else:
            y = to_categorical(y, num_classes=5)[:, :, :, [0, 1, 2, 4]]

        if not self.dim == self.original_size:
            X = tf.keras.preprocessing.image.smart_resize(
                X, self.dim, interpolation='bilinear')

            y = tf.keras.preprocessing.image.smart_resize(
                y, self.dim, interpolation='bilinear')
            y = np.round(y)

        return X, y

    def data_sample(self, id):

        ID = self.list_IDs[id]
        im_frame1 = np.expand_dims(np.array(Image.open(ID + '_flair.png')), -1)
        im_frame2 = np.expand_dims(np.array(Image.open(ID + '_T1.png')), -1)
        im_frame3 = np.pad(np.expand_dims(np.array(Image.open(ID + '_T1ce.png')), -1), ((8, 8), (8, 8), (0, 0)))
        im_frame4 = np.expand_dims(np.array(Image.open(ID + '_T2.png')), -1)
        X = np.concatenate((im_frame1, im_frame2, im_frame3, im_frame4), -1) / 2000

        y = np.int32(np.array(Image.open(ID + '_seg.png')) / 10000)

        if self.binary_class:
            y = np.sign(y)
            y = to_categorical(y, num_classes=2)
        else:
            y = to_categorical(y, num_classes=5)[:, :, [0, 1, 2, 4]]

        y = np.expand_dims(y, 0)

        if not self.dim == self.original_size:
            X = tf.keras.preprocessing.image.smart_resize(
                X, self.dim, interpolation='bilinear')

            y = tf.keras.preprocessing.image.smart_resize(
                y, self.dim, interpolation='bilinear')
            y = np.round(y)

        return X, y

    def count_pixels(self):

        class_counts = np.zeros(4)
        l = len(self.list_IDs)
        for j, ID in enumerate(self.list_IDs):
            y_ = np.int32(np.array(Image.open(ID + '_seg.png')) / 10000)
            for i, cl in enumerate([0, 1, 2, 4]):
                class_counts[i] += np.sum(y_ == cl)

            if j % 1000 == 0:
                print(j, "/", l)

        return class_counts


