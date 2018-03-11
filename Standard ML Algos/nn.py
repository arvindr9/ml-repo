from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

def fc_net(X, dims, training):
    for i in range(len(dims) - 1):
        out_dim = dims[i + 1]
        X = tf.layers.dense(X, out_dim)
        X = tf.layers.batch_normalization(X, training=training)
        X = tf.nn.relu(X)
    return X

def main():
    mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)
    X = tf.placeholder(tf.float32, [None, 784])
    is_training = tf.placeholder(tf.bool)
    y = fc_net(X, [784, 50, 10], is_training)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict = {X: batch_xs, y_: batch_ys, is_training:True})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels, is_training: False}))

if __name__ == '__main__':
    main()
