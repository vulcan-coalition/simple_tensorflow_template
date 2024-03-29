import argparse
from network import Trainer
import os
import sys
import shutil
import yaml
import tensorflow as tf


import dataformat as dataformat
# import gen_data as gen_data

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='weights', help="outputs path")
parser.add_argument('--resume', action="store_true", default=False)
opts = parser.parse_args()


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def prepare_directory(output_dir):
    if not os.path.exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        os.makedirs(output_dir)
    else:
        print("Clearing directory: {}".format(output_dir))
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                os.remove(os.path.join(root, file))
    shutil.copy(opts.config, os.path.join(output_dir, 'config.yaml'))  # copy config file to output folder


# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
trainer = Trainer(config)

generator = dataformat.get_trainer_generator(config['batch_size'], config['input_dims'], config['output_dims'])

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path, "outputs", model_name)

if opts.resume:
    trainer.resume(output_directory)
else:
    prepare_directory(output_directory)

    # Start training
max_iter = config['max_iter']
sum_loss = 0
sum_count = 0

writer = tf.summary.create_file_writer(os.path.join(opts.output_path, "logs"))
for mb in generator:
    data_tensor = mb[0]
    label_tensor = mb[1]

    # Main training code
    loss, iterations = trainer.update(data_tensor, label_tensor, config)
    sum_loss = sum_loss + loss.numpy()
    sum_count = sum_count + 1

    # Dump training stats in log file
    if (iterations + 1) % config['log_iter'] == 0:
        print("Iteration: %08d/%08d, loss: %.8f" % (iterations + 1, max_iter, sum_loss / sum_count))
        sum_loss = 0
        sum_count = 0

    with writer.as_default():
        tf.summary.scalar('Loss/train', loss, iterations)

    # Save network weights
    if (iterations + 1) % config['snapshot_save_iter'] == 0:
        trainer.save(output_directory)

    if iterations >= max_iter:
        sys.exit('Finish training')
