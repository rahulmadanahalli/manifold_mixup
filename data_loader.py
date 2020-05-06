import tensorflow as tf
import numpy as np

def pre_processing_mnist(x, y):
    return x.astype('float32') / 255, tf.one_hot(y, 10)

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, y_train = pre_processing_mnist(x_train, y_train)
    x_test, y_test = pre_processing_mnist(x_test, y_test)

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    return x_train, y_train, x_val, y_val, x_test, y_test

def generate_two_spirals_dataset(n_points, noise=.5):
    """
     Returns the two spirals dataset.
     https://glowingpython.blogspot.com/2017/04/solving-two-spirals-problem-with-keras.html
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points), np.ones(n_points)))
    )

def pre_processing_spiral(x, y):
    return x.astype('float32'), tf.one_hot(y.astype(int), 2)

def split_dataset(x, y, perc):
    num = round(len(x) * perc)
    return x[:num], y[:num], x[num:], y[num:]

def get_two_spirals_data(n_points, noise=0.5):
    x, y = generate_two_spirals_dataset(n_points, noise=noise)
    indices = tf.random.shuffle(tf.range(x.shape[0]))
    x = x[indices]
    y = y[indices]

    x, y = pre_processing_spiral(x, y)

    x_train, y_train, x_test, y_test = split_dataset(x, y, 0.9)
    x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, 0.9)
    return x_train, y_train, x_val, y_val, x_test, y_test
