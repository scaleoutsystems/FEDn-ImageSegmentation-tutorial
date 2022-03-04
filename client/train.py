import sys
import tensorflow as tf
import numpy as np
import yaml
import os
from fedn.utils.kerashelper import KerasHelper
from models.unet import unet
from data.datagenerator import DataGenerator
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(model, data_generator, settings):
    print("-- RUNNING TRAINING --", flush=True)

    # Train one round
    history = model.model.fit(data_generator, epochs=int(settings['epochs']), verbose=2)


    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    dim = tuple(np.int32(settings['image_dimensions'].split(',')))
    # Create datageneraor

    path = '../data/train_set'

    # Print out if GPU is available
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    d, counts = np.unique(np.array(["_".join(x.split("_")[:-1]) for x in os.listdir(path)]), return_counts=True)
    data_ids = [os.path.join(path,s) for s in d if s.startswith("Subject")]

    data_generator = DataGenerator(data_ids,
                                   batch_size=int(settings['batch_size']),
                                   dim=dim,
                                   n_channels=4,
                                   shuffle=True,
                                   original_size=(256,256),
                                   binary_class=bool(settings['binary_class']))

    # if not exists! compute and store class weights
    if os.path.isfile('class_weights_trainset.npy'):
        class_weights = np.load('class_weights_trainset.npy')

    else:
        pixel_counts = data_generator.count_pixels()
        class_weights = 1 / np.sum(1 / pixel_counts) / pixel_counts
        np.save('class_weights_trainset.npy', class_weights)


    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    model = unet(img_size=(*dim, 4),
                 Nclasses=4,
                 class_weights=class_weights,
                 Nfilter_start=np.int32(settings['Nfilter_start']),
                 depth=np.int32(settings['depth']))

    model.model.set_weights(weights)

    model = train(model,data_generator,settings)
    helper.save_model(model.model.get_weights(),sys.argv[2])
