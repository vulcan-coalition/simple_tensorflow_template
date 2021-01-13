import argparse
from network import Model
import os
import yaml
import numpy as np

import dataformat as dataformat
# import gen_data as gen_data

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='weights', help="outputs path")
parser.add_argument('--resume', action="store_true")
opts = parser.parse_args()


dir_path = os.path.dirname(os.path.realpath(__file__))


def get_config(config):
    with open(os.path.join(dir_path, config), 'r') as stream:
        return yaml.load(stream)


# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
model = Model(config)

generator = dataformat.get_trainer_generator(config['batch_size'], config['input_dims'], config['output_dims'])

model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(dir_path, opts.output_path, "outputs", model_name)
model.resume(os.path.join(output_directory, 'checkpoints'))


if __name__ == '__main__':

    for i, mb in enumerate(generator):
        predictions = model.predict(mb[0])
        label_tensor = mb[1].numpy()

        print(predictions, label_tensor)

        if i > 5:
            break
