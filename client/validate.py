import sys
import json
import os
import yaml
import numpy as np

import tensorflow as tf
from fedn.utils.kerashelper import KerasHelper
from models.unet import unet
from data.datagenerator import DataGenerator
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def validate(data_path, weights, settings, dataset_label):
    print("-- " + dataset_label + " RUNNING VALIDATION --", flush=True)

    dim = tuple(np.int32(settings['image_dimensions'].split(',')))

    d = np.unique(np.array(["_".join(x.split("_")[:-1]) for x in os.listdir(data_path)]))
    data_ids = [os.path.join(data_path, s) for s in d if s.startswith("Subject")]
    data_generator = DataGenerator(data_ids,
                                    batch_size=int(settings['batch_size']),
                                    dim=dim,
                                    n_channels=4,
                                    shuffle=True,
                                    original_size=(256, 256),
                                    binary_class=bool(settings['binary_class']))

    # if not exists! compute and store class weights
    if os.path.isfile('class_weights_' + dataset_label + '.npy'):
        class_weights = np.load('class_weights_' + dataset_label + '.npy')

    else:
        pixel_counts = data_generator.count_pixels()
        class_weights= 1 / np.sum(1 / pixel_counts) / pixel_counts
        np.save('class_weights_' + dataset_label + '.npy', class_weights)

    model = unet(img_size=(*dim, 4),
                 Nclasses=4,
                 class_weights=class_weights,
                 Nfilter_start=np.int32(settings['Nfilter_start']),
                 depth=np.int32(settings['depth']))

    model.model.set_weights(weights)

    results = model.model.evaluate(data_generator)

    report = {dataset_label + '_loss': results[0],
              dataset_label + '_accuracy': results[1],
              dataset_label + '_dice': results[2],
              dataset_label + '_dice_background': results[3],
              dataset_label + '_dice_class1': results[4],
              dataset_label + '_dice_class2': results[5],
              dataset_label + '_dice_class3': results[6],
              }

    print("-- " + dataset_label + " VALIDATION COMPLETE! --", flush=True)
    return report


if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise (e)



    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])


    train_report = validate('../data/train_set', weights, settings, 'trainset')
    val_report = validate('../data/validation_set', weights, settings, 'valset')
    report = {}
    report.update(train_report)
    report.update(val_report)

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))



