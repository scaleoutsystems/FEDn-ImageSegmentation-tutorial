import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Conv2DTranspose, \
    Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K


class unet(object):

    def __init__(self, img_size, Nclasses, class_weights, model_name='myWeightsAug.h5', Nfilter_start=64, depth=3,
                 batch_size=3):
        self.img_size = img_size
        self.Nclasses = Nclasses
        self.class_weights = class_weights
        self.model_name = model_name
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        self.batch_size = batch_size

        self.model = Sequential()
        inputs = Input(img_size)

        def dice(y_true, y_pred, eps=1e-5):

            num = 2. * K.sum(self.class_weights * K.sum(y_true * y_pred, axis=[0, 1, 2]))
            den = K.sum(self.class_weights * K.sum(y_true + y_pred, axis=[0, 1, 2])) + eps

            return num / den

        def dice_alt(y_true, y_pred, eps=1e-15):

            num = K.sum(y_true * y_pred, axis=[0, 1, 2])
            den = K.sum(y_true + y_pred, axis=[0, 1, 2]) + eps

            return K.sum(self.class_weights / np.sum(self.class_weights) * 2. * num / den)

        def diceLoss(y_true, y_pred):
            return 1 - dice(y_true, y_pred)

        def dice_parts(y_true, y_pred, eps=1e-5):
            num = K.sum(y_true * y_pred, axis=[0, 1, 2])
            den = K.sum(y_true + y_pred, axis=[0, 1, 2])
            div = 2. * num / (den + eps)

            return div

        def dice1(y_true, y_pred, eps=1e-10):
            return dice_parts(y_true, y_pred, eps=1e-5)[1]

        def dice2(y_true, y_pred, eps=1e-10):
            return dice_parts(y_true, y_pred, eps=1e-5)[2]

        def dice3(y_true, y_pred, eps=1e-10):
            return dice_parts(y_true, y_pred, eps=1e-5)[3]

        def diceneg(y_true, y_pred, eps=1e-10):
            return dice_parts(y_true, y_pred, eps=1e-5)[0]

        def bceLoss(y_true, y_pred):
            bce = K.sum(-self.class_weights * K.sum(y_true * K.log(y_pred), axis=[0, 1, 2]))
            return bce

            # This is a help function that performs 2 convolutions, each followed by batch normalization

        # and ReLu activations, Nf is the number of filters, filter size (3 x 3)
        def convs(layer, Nf):
            x = Conv2D(Nf, (3, 3), kernel_initializer='he_normal', padding='same')(layer)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(Nf, (3, 3), kernel_initializer='he_normal', padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x

        # This is a help function that defines what happens in each layer of the encoder (downstream),
        # which calls "convs" and then Maxpooling (2 x 2). Save each layer for later concatenation in the upstream.
        def encoder_step(layer, Nf):
            y = convs(layer, Nf)
            x = MaxPooling2D(pool_size=(2, 2))(y)
            return y, x

        # This is a help function that defines what happens in each layer of the decoder (upstream),
        # which contains transpose convolution (filter size (3 x 3), stride (2,2) batch normalization, concatenation with
        # corresponding layer (y) from encoder, and lastly "convs"
        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv2DTranspose(filters=Nf, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                kernel_initializer='he_normal')(layer)
            x = BatchNormalization()(x)
            x = concatenate([x, layer_to_concatenate])
            x = convs(x, Nf)
            return x

        layers_to_concatenate = []
        x = inputs

        # Make encoder with 'self.depth' layers,
        # note that the number of filters in each layer will double compared to the previous "step" in the encoder
        for d in range(self.depth - 1):
            y, x = encoder_step(x, self.Nfilter_start * np.power(2, d))
            layers_to_concatenate.append(y)

        # Make bridge, that connects encoder and decoder using "convs" between them.
        # Use Dropout before and after the bridge, for regularization. Use dropout probability of 0.2.
        x = Dropout(0.2)(x)
        x = convs(x, self.Nfilter_start * np.power(2, self.depth - 1))
        x = Dropout(0.2)(x)

        # Make decoder with 'self.depth' layers,
        # note that the number of filters in each layer will be halved compared to the previous "step" in the decoder
        for d in range(self.depth - 2, -1, -1):
            y = layers_to_concatenate.pop()
            x = decoder_step(x, y, self.Nfilter_start * np.power(2, d))

            # Make classification (segmentation) of each pixel, using convolution with 1 x 1 filter
        final = Conv2D(filters=self.Nclasses, kernel_size=(1, 1), activation='softmax')(x)

        # Create model
        self.model = Model(inputs=inputs, outputs=final)
        self.model.compile(loss=diceLoss, optimizer=Adam(learning_rate=1e-4),
                           metrics=['accuracy', dice, diceneg, dice1, dice2, dice3])


