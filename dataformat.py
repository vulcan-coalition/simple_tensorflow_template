import os
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "data")


def get_trainer_generator(max_batch_size, input_dims, output_dims):
    while True:
        yield tf.random.normal([max_batch_size, input_dims]), tf.ones([max_batch_size, output_dims])


if __name__ == '__main__':
    gen = get_trainer_generator(8, 1, 1)
    data, labels = next(gen)
    print(data, labels)
