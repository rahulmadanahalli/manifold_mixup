import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils
from collections import defaultdict


def plot_bottleneck_representation(bottleneck_representation, y):
    bottleneck_x, bottleneck_y = tf.transpose(bottleneck_representation)
    plt.scatter(bottleneck_x, bottleneck_y, c=y)
    plt.show()

def show_b2_model_hidden_representation(model, x, y, bottleneck_layer_name='bottleneck'):
    y = utils.one_hot_to_index_vector(y)
    bottleneck_representation = utils.get_activation_from_layer(model, bottleneck_layer_name, x)

    bottleneck_rep_0_4, y_0_4 = utils.get_hidden_representation_for_class(bottleneck_representation, y, range(5), 250)
    bottleneck_rep_0_9, y_0_9 = utils.get_hidden_representation_for_class(bottleneck_representation, y, range(10), 250)

    plot_bottleneck_representation(bottleneck_rep_0_4, y_0_4)
    plot_bottleneck_representation(bottleneck_rep_0_9, y_0_9)

def compare_svd_for_b12_models(models, x, y, classes=range(10), bottleneck_layer_name='bottleneck'):
    y = utils.one_hot_to_index_vector(y)
    class_to_svd = defaultdict(dict)
    for model in models:
        bottleneck_representation = utils.get_activation_from_layer(model, bottleneck_layer_name, x)
        for c in classes:
            bottleneck_rep_c, _ = utils.get_hidden_representation_for_class(bottleneck_representation, y, [c])
            s, _, _ = tf.linalg.svd(bottleneck_rep_c)
            class_to_svd[c][model.name] = s

    for c in classes:
        for model in models:
            plt.plot(class_to_svd[c][model.name], label=model.name)
        plt.legend()
        plt.title("Singular Values of hidden representations for class {}".format(c))
        plt.show()

def plot_spiral_dataset(x, y):
    y = utils.one_hot_to_index_vector(y)
    plt.title('Spiral dataset')
    plt.plot(x[y == 0, 0], x[y == 0, 1], '.', label='class 1')
    plt.plot(x[y == 1, 0], x[y == 1, 1], '.', label='class 2')
    plt.legend()
    plt.show()

def plot_spiral_model_confidence(model, x_train, y_train, title='spiral model'):
    xi = np.arange(-15, 15, 0.1)
    xj = np.arange(-15, 15, 0.1)
    x_sample = np.array([[j,i] for i in xi for j in xj])
    y = model(x_sample)

    # get P(Y=1|X)
    confidence = tf.transpose(tf.math.softmax(y))[1].numpy()
    confidence = confidence.reshape((len(xi), len(xj)))
    x, y = np.meshgrid(xi, xj)

    plt.pcolormesh(x, y, confidence)

    def get_dim(x, y, dim=0, label_class=0, subset=500):
        return x[y==label_class, dim][:subset]

    y_train_class = utils.one_hot_to_index_vector(y_train)
    x_d0_l0 = get_dim(x_train, y_train_class, dim=0, label_class=0)
    x_d1_l0 = get_dim(x_train, y_train_class, dim=1, label_class=0)
    x_d0_l1 = get_dim(x_train, y_train_class, dim=0, label_class=1)
    x_d1_l1 = get_dim(x_train, y_train_class, dim=1, label_class=1)

    plt.title(title)
    plt.plot(x_d0_l0, x_d1_l0, '.', label='class 0')
    plt.plot(x_d0_l1, x_d1_l1, '.', label='class 1')
    plt.colorbar()
    plt.show()