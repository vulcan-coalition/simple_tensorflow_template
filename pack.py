import argparse
from network import Trainer
import os
import yaml
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml', help='Path to the config file.')
parser.add_argument('--input_path', type=str, default='weights', help="inputs path")
opts = parser.parse_args()


dir_path = os.path.dirname(os.path.realpath(__file__))
artifacts_path = os.path.join(dir_path, "artifacts")


def get_config(config):
    with open(os.path.join(dir_path, config), 'r') as stream:
        return yaml.load(stream)


# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
trainer = Trainer(config)

model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(dir_path, opts.input_path, "outputs", model_name)
trainer.resume(os.path.join(output_directory, 'checkpoints'))


if __name__ == '__main__':

    model = trainer.get_model()
    saved_model_dir = os.path.join(artifacts_path, model_name)
    tf.saved_model.save(model, saved_model_dir)

    imported = tf.saved_model.load(saved_model_dir)
    print(list(imported.signatures.keys()))
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open(os.path.join(artifacts_path, model_name + '.tflite'), 'wb') as f:
        f.write(tflite_model)
