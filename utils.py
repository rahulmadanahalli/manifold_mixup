import tensorflow as tf
import numpy as np

def one_hot_to_index_vector(v):
    return np.argmax(v, axis=1)

def get_activation_from_layer(model, layer_name, inputs):
    activation = inputs
    for layer in model.layers:
        activation = layer(activation)
        if layer.name == layer_name:
            break
    return activation.numpy()

def get_hidden_representation_for_class(bottleneck_representation, y, c, subset=None):
    indices = [i for i, y_i in enumerate(y) if y_i in c]
    indices = indices[:subset] if subset else indices
    y = tf.gather(y, indices)
    bottleneck_representation = tf.gather(bottleneck_representation, indices)
    return bottleneck_representation, y

