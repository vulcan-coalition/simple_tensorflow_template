import tensorflow as tf
import os
from modules import *


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def compute_mse_loss(predictions, targets):
    return tf.reduce_mean(tf.square(predictions - targets))


class Trainer:

    def __init__(self, hyperparameters):
        self.input_dims = hyperparameters['input_dims']
        self.output_dims = hyperparameters['output_dims']
        self.layer_sizes = hyperparameters['layers']

        self.model = Multilayer_linear(self.input_dims, self.layer_sizes, self.output_dims)

        # Setup the optimizers
        lr = hyperparameters['lr']

        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=hyperparameters['step_size'],
            decay_rate=hyperparameters['weight_decay'],
            staircase=True)
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.opt, model=self.model)

    def update(self, b_data, b_labels, hyperparameters):
        with tf.GradientTape() as tape:
            loss_values = compute_mse_loss(self.model(b_data, training=True), b_labels)
            grads = tape.gradient(loss_values, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        self.checkpoint.step.assign_add(1)
        return loss_values, self.checkpoint.step.numpy()

    def predict(self, b_data):
        predictions = self.model(b_data, training=False)
        return predictions

    def resume(self, checkpoint_dir):
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def save(self, checkpoint_dir):
        self.checkpoint.save(checkpoint_dir)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    print("assert that the model works.")

    params = {
        "max_iter": 1000,
        "lr": 0.01,
        "step_size": 1000,
        "weight_decay": 0.5,
        "num_layers": 2,
        "num_classes": 8,
        "layers": [8, 8],
        "input_dims": 8,
        "output_dims": 4
    }

    network = Trainer(params)

    x = tf.ones([2, 8])
    y = tf.zeros([2, 4])
    step = 0
    step = network.resume("./weights/test_network/")
    print("resume from", step)
    for i in range(1, 1000):
        loss, _ = network.update(x, y, params)
        if i % 100 == 0:
            print(i, loss.numpy())
            network.save("./weights/test_network/")
