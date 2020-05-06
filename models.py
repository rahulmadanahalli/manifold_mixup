import numpy as np
import tensorflow as tf
from tensorflow.python.eager import backprop
import tensorflow_probability as tfp
import enum


LEAKY_RELU_ALPHA = 0.3


class MixupMode(enum.Enum):
    NO_MIXUP = 'no_mixup'
    INPUT_MIXUP = 'input_mixup'
    MANIFOLD_MIXUP = 'manifold_mixup'


class ManifoldMixupModel(tf.keras.models.Sequential):
    def __init__(self, layers_tuple, mixup_alpha=None, name=None):
        super(ManifoldMixupModel, self).__init__([layer for layer, _ in layers_tuple], name=name)
        self.alpha = mixup_alpha
        self.eligible_mixup_layers = [layer for layer, eligible_for_mixup in layers_tuple if eligible_for_mixup]

    def select_mixup_layer_at_random(self):
        if len(self.eligible_mixup_layers) == 0:
            return None
        mixed_layer_ind = np.random.choice(len(self.eligible_mixup_layers))
        return self.eligible_mixup_layers[mixed_layer_ind]

    def mixup(self, lmbda, inputs_a, inputs_b):
        return lmbda * inputs_a + (1 - lmbda) * inputs_b

    def manifold_mixup(self, inputs, y_true):
        assert self.alpha is not None
        alpha = tf.constant(self.alpha)
        dist = tfp.distributions.Beta(alpha, alpha)
        lmbda = dist.sample(1)
        indices = tf.random.shuffle(tf.range(inputs.shape[0]))
        x_mixup = self.mixup(lmbda, inputs, tf.gather(inputs, indices))
        y_mixup = self.mixup(lmbda, y_true, tf.gather(y_true, indices))
        return x_mixup, y_mixup

    def call(self, inputs, training=False, mixup_mode=False, y_true=None):
        can_mixup = training and mixup_mode
        k = self.select_mixup_layer_at_random() if can_mixup else None

        layer_activation = inputs
        for layer in self.layers:
            if can_mixup and layer == k:
                layer_activation, y_true = self.manifold_mixup(layer_activation, y_true)
            layer_activation = layer(layer_activation)

        if can_mixup:
            return layer_activation, y_true
        return layer_activation

    def train_step(self, x, y_true):
        with backprop.GradientTape() as tape:
            y_pred, y_mixed_true = self(x, training=True, mixup_mode=True, y_true=y_true)
            loss = self.loss(y_mixed_true, y_pred)

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss

    def get_accuracy(self, inputs, y_true):
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(y_true, self(inputs))
        return m.result().numpy()

    def step_decay_schedule(self, epoch):
        if epoch % 10 == 0:
            self.optimizer.lr = self.optimizer.lr * 0.1

    def fit(self, x_train, y_train, batch_size=None, epochs=1, step_decay=True, validation_data=None):
        if batch_size:
            x_train = np.array_split(x_train, len(x_train) / batch_size)
            y_train = np.array_split(y_train, len(y_train) / batch_size)

        x_val, y_val = validation_data[0], validation_data[1]
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for epoch in range(1, epochs + 1):
            epoch_loss_avg.reset_states()
            epoch_accuracy.reset_states()
            for x, y in zip(x_train, y_train):
                loss_value = self.train_step(x, y)
                epoch_loss_avg.update_state(loss_value)
                epoch_accuracy.update_state(y, self(x))

            validation_accuracy = self.get_accuracy(x_val, y_val)
            print("Epoch {:03d}: Loss: {:.3f}, Training Accuracy: {:.3%}, Validation Accuracy: {:.3%}".format(
                epoch, epoch_loss_avg.result(), epoch_accuracy.result(), validation_accuracy
            ))
            if step_decay:
                self.step_decay_schedule(epoch)



def create_mnist_model_bottleneck_2(mixup_mode=MixupMode.NO_MIXUP):
    mixup_alpha = 1.0
    include_input_mixup = mixup_mode is MixupMode.INPUT_MIXUP
    include_manifold_mixup = mixup_mode is MixupMode.MANIFOLD_MIXUP
    return ManifoldMixupModel(
        [
            (tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)), include_input_mixup),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(2, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA), name='bottleneck'),
             False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), include_manifold_mixup),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), include_manifold_mixup),
            (tf.keras.layers.Dense(10, activation='softmax', name='output'), False),
        ],
        mixup_alpha=mixup_alpha if mixup_mode is not MixupMode.NO_MIXUP else None,
        name="bottleneck_2_{}".format(mixup_mode.value)
    )

def create_mnist_model_bottleneck_12(mixup_mode=MixupMode.NO_MIXUP):
    mixup_alpha = 2.0
    include_input_mixup = mixup_mode is MixupMode.INPUT_MIXUP
    include_manifold_mixup = mixup_mode is MixupMode.MANIFOLD_MIXUP
    return ManifoldMixupModel(
        [
            (tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)), include_input_mixup),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA), name='bottleneck'), include_manifold_mixup),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(10, activation='softmax', name='output'), False),
        ],
        mixup_alpha=mixup_alpha if mixup_mode is not MixupMode.NO_MIXUP else None,
        name="bottleneck_12_{}".format(mixup_mode.value)
    )

def create_spiral_model(mixup_mode=MixupMode.NO_MIXUP):
    mixup_alpha = 1.0
    include_input_mixup = mixup_mode is MixupMode.INPUT_MIXUP
    include_manifold_mixup = mixup_mode is MixupMode.MANIFOLD_MIXUP
    return ManifoldMixupModel(
        [
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA), input_shape=(2,)), include_input_mixup),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), False),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), include_manifold_mixup),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), include_manifold_mixup),
            (tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)), include_manifold_mixup),
            (tf.keras.layers.Dense(2, activation='softmax'), include_manifold_mixup),
        ],
        mixup_alpha=mixup_alpha if mixup_mode is not MixupMode.NO_MIXUP else None,
        name="spiral_{}".format(mixup_mode.value)
    )

def train_model(model, training, validation, epochs=24, batch_size=100, step_decay=True, save_to_file=None):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_crossentropy'],
    )
    print('# Fit model on training data')
    x_train, y_train = training
    x_val, y_val = validation
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        step_decay=step_decay,
        validation_data=(x_val, y_val)
    )
    if save_to_file:
        model.save_weights(save_to_file)
    return model
