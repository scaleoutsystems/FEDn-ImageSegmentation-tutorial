import yaml
import numpy as np
from fedn.utils.kerashelper import KerasHelper
from client.models.unet import unet

if __name__ == '__main__':

    with open('client/settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise (e)

    dim = tuple(np.int32(settings['image_dimensions'].split(',')))

    if bool(settings['binary_class']):
        model = unet(img_size=(*dim, 4),
                     Nclasses=4,
                     class_weights=np.ones(2),
                     Nfilter_start=np.int32(settings['Nfilter_start']),
                     depth=np.int32(settings['depth']))

    else:
        model = unet(img_size=(*dim, 4),
                     Nclasses=4,
                     class_weights=np.ones(4),
                     Nfilter_start=np.int32(settings['Nfilter_start']),
                     depth=np.int32(settings['depth']))

    weights = model.model.get_weights()

    outfile_name = "initial_weights.npz"
    helper = KerasHelper()
    helper.save_model(weights, outfile_name)
