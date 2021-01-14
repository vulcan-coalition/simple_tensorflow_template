import argparse
from network import Trainer
import os
import yaml
import tensorflow as tf
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml', help='Path to the config file.')
parser.add_argument('--input_path', type=str, default='weights', help="inputs path")
parser.add_argument('--test_load', action="store_true", default=True, help="Dont do this if the model is large.")
opts = parser.parse_args()


dir_path = os.path.dirname(os.path.realpath(__file__))
artifacts_path = os.path.join(dir_path, "artifacts")


def prepare_directory(output_dir):
    if not os.path.exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        os.makedirs(output_dir)
    else:
        print("Clearing directory: {}".format(output_dir))
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                os.remove(os.path.join(root, file))


def get_config(config):
    with open(os.path.join(dir_path, config), 'r') as stream:
        return yaml.load(stream)


# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
trainer = Trainer(config)

model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(dir_path, opts.input_path, "outputs", model_name)
trainer.resume(output_directory)


if __name__ == '__main__':

    model = trainer.get_model()
    saved_model_dir = os.path.join(artifacts_path, model_name)
    prepare_directory(saved_model_dir)

    # In the module the function declare @tf.function, here we specify its signature.
    tf.saved_model.save(model, saved_model_dir, signatures=model.call.get_concrete_function(
        tf.TensorSpec(shape=[None, config["input_dims"]], dtype=tf.float32)))

    imported = tf.saved_model.load(saved_model_dir)
    print(list(imported.signatures.keys()))
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open(os.path.join(artifacts_path, model_name + '.tflite'), 'wb') as f:
        f.write(tflite_model)

    if opts.test_load:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("input details", input_details)
        print("output details", output_details)

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
