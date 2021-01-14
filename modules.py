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
        return tf.matmul(inputs, self.W) + self.B


if __name__ == '__main__':
    print("assert that model works.")
    model = Linear(16, 8)
    x = tf.random.normal([2, 16])
    y = model(x)
    print(y)
