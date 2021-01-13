import tensorflow as tf


class Linear(tf.keras.Model):
    def __init__(self, input_dims, output_dims):
        super(Linear, self).__init__()
        self.W = tf.Variable(tf.random.normal([input_dims, output_dims]), name='weight')
        self.B = tf.Variable(tf.zeros([output_dims]), name='bias')

    def call(self, inputs):
        return tf.matmul(inputs, self.W) + self.B


if __name__ == '__main__':
    print("assert that model works.")
    model = Linear(16, 8)
    x = tf.random.normal([2, 16])
    y = model(x)
    print(y)
