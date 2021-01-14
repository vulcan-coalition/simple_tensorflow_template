import tensorflow as tf


class Linear(tf.keras.Model):
    def __init__(self, input_dims, output_dims):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.W = tf.Variable(
            initial_value=w_init(shape=(input_dims, output_dims), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.B = tf.Variable(initial_value=b_init(shape=(output_dims,), dtype="float32"), trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.W) + self.B)


class Multilayer_linear(tf.keras.Model):
    def __init__(self, input_dims, layer_sizes, output_dims):
        super(Multilayer_linear, self).__init__()

        current_size = input_dims
        layers = []
        for s in layer_sizes:
            layers.append(Linear(current_size, s))
            current_size = s
        layers.append(Linear(current_size, output_dims))

        self.model = tf.keras.Sequential(layers)

    # need to specify this signature for exporting to tensorflow lite
    @tf.function
    def call(self, inputs):
        return self.model(inputs)


if __name__ == '__main__':
    print("assert that model works.")
    model = Linear(16, 8)
    x = tf.random.normal([2, 16])
    y = model(x)
    print(y)
