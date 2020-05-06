import models
from models import MixupMode
import data_loader
import visualizer
import time

TRAINED_MODELS_DIR = './results/trained_models'

def get_path_for_trained_models(model):
    return "{}/{}_{}.h5".format(TRAINED_MODELS_DIR, model.name, time.time_ns())

def show_bottleneck_representation_demo(baseline_weights_file=None, manifold_mixup_weights_file=None):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~ BOTTLENECK REPRESENTATION DEMO ~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_mnist_data()

    mnist_model_b2 = models.create_mnist_model_bottleneck_2(mixup_mode=MixupMode.NO_MIXUP)
    mnist_model_b2_with_mixup = models.create_mnist_model_bottleneck_2(mixup_mode=MixupMode.MANIFOLD_MIXUP)

    print("baseline mnist 2-node bottelneck model")
    if baseline_weights_file:
        print("Loading model from file {}...".format(baseline_weights_file))
        mnist_model_b2.load_weights(baseline_weights_file)
    else:
        print("Training...")
        models.train_model(
            mnist_model_b2,
            (x_train, y_train),
            (x_val, y_val),
            save_to_file=get_path_for_trained_models(mnist_model_b2)
        )

    print("Test Accuracy: {:.3f}".format(mnist_model_b2.get_accuracy(x_test, y_test)))

    print("mnist 2-node bottleneck model with manifold mixup")
    if manifold_mixup_weights_file:
        print("Loading model from file {}...".format(manifold_mixup_weights_file))
        mnist_model_b2_with_mixup.load_weights(manifold_mixup_weights_file)
    else:
        print("Training...")
        models.train_model(
            mnist_model_b2_with_mixup,
            (x_train, y_train),
            (x_val, y_val),
            save_to_file=get_path_for_trained_models(mnist_model_b2_with_mixup)
        )

    print("Test Accuracy: {:.3f}".format(mnist_model_b2_with_mixup.get_accuracy(x_test, y_test)))

    visualizer.show_b2_model_hidden_representation(mnist_model_b2, x_train, y_train)
    visualizer.show_b2_model_hidden_representation(mnist_model_b2_with_mixup, x_train, y_train)

def show_svd_demo(baseline_weights_file=None, input_mixup_weights_file=None, manifold_mixup_weights_file=None):
    print("~~~~~~~~~~~~~~")
    print("~~ SVD DEMO ~~")
    print("~~~~~~~~~~~~~~")

    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_mnist_data()

    mnist_model_b12 = models.create_mnist_model_bottleneck_12(mixup_mode=MixupMode.NO_MIXUP)
    mnist_model_b12_with_input_mixup = models.create_mnist_model_bottleneck_12(mixup_mode=MixupMode.INPUT_MIXUP)
    mnist_model_b12_with_manifold_mixup = models.create_mnist_model_bottleneck_12(mixup_mode=MixupMode.MANIFOLD_MIXUP)

    print("baseline mnist 12-node bottleneck model")
    if baseline_weights_file:
        print("Loading model from file {}...".format(baseline_weights_file))
        mnist_model_b12.load_weights(baseline_weights_file)
    else:
        print("Training...")
        models.train_model(
            mnist_model_b12,
            (x_train, y_train),
            (x_val, y_val),
            epochs=6,
            save_to_file=get_path_for_trained_models(mnist_model_b12)
        )

    print("Test Accuracy: {:.3f}".format(mnist_model_b12.get_accuracy(x_test, y_test)))

    print("mnist 12-node bottleneck model with input mixup")
    if input_mixup_weights_file:
        print("Loading model from file {}...".format(input_mixup_weights_file))
        mnist_model_b12_with_input_mixup.load_weights(input_mixup_weights_file)
    else:
        print("Training...")
        models.train_model(
            mnist_model_b12_with_input_mixup,
            (x_train, y_train),
            (x_val, y_val),
            save_to_file=get_path_for_trained_models(mnist_model_b12_with_input_mixup)
        )

    print("Test Accuracy: {:.3f}".format(mnist_model_b12_with_input_mixup.get_accuracy(x_test, y_test)))

    print("mnist 12-node bottleneck model with manifold mixup")
    if manifold_mixup_weights_file:
        print("Loading model from file {}...".format(manifold_mixup_weights_file))
        mnist_model_b12_with_manifold_mixup.load_weights(manifold_mixup_weights_file)
    else:
        print("Training...")
        models.train_model(
            mnist_model_b12_with_manifold_mixup,
            (x_train, y_train),
            (x_val, y_val),
            save_to_file=get_path_for_trained_models(mnist_model_b12_with_manifold_mixup)
        )

    print("Test Accuracy: {:.3f}".format(mnist_model_b12_with_manifold_mixup.get_accuracy(x_test, y_test)))

    visualizer.compare_svd_for_b12_models(
        [mnist_model_b12, mnist_model_b12_with_input_mixup, mnist_model_b12_with_manifold_mixup],
        x_train,
        y_train
    )

def show_spiral_demo(baseline_model_weights_file=None, mixup_model_weights_file=None):
    print("~~~~~~~~~~~~~~~~~")
    print("~~ SPIRAL DEMO ~~")
    print("~~~~~~~~~~~~~~~~~")

    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_two_spirals_data(2000, noise=0.9)
    # visualizer.plot_spiral_dataset(x_train, y_train)

    spiral_model = models.create_spiral_model(models.MixupMode.NO_MIXUP)
    spiral_model_with_mixup = models.create_spiral_model(models.MixupMode.MANIFOLD_MIXUP)

    print("baseline spiral model")
    if baseline_model_weights_file:
        print("Loading model from file {}...".format(baseline_model_weights_file))
        spiral_model.load_weights(baseline_model_weights_file)
    else:
        print("Training...")
        models.train_model(
            spiral_model,
            (x_train, y_train),
            (x_val, y_val),
            batch_size=20,
            save_to_file=get_path_for_trained_models(spiral_model)
        )

    print("Test Accuracy: {:.3f}".format(spiral_model.get_accuracy(x_test, y_test)))

    print("Spiral model with manifold mixup")
    if mixup_model_weights_file:
        print("Loading model from file {}...".format(mixup_model_weights_file))
        spiral_model_with_mixup.load_weights(mixup_model_weights_file)
    else:
        print("Training...")
        models.train_model(
            spiral_model_with_mixup,
            (x_train, y_train),
            (x_val, y_val),
            batch_size=20,
            save_to_file=get_path_for_trained_models(spiral_model_with_mixup)
        )

    print("Test Accuracy: {:.3f}".format(spiral_model_with_mixup.get_accuracy(x_test, y_test)))

    visualizer.plot_spiral_model_confidence(spiral_model, x_train, y_train, title='no mixup')
    visualizer.plot_spiral_model_confidence(spiral_model_with_mixup, x_train, y_train, title='manifold mixup')

def main():
    show_bottleneck_representation_demo(
        baseline_weights_file='./{}/{}'.format(TRAINED_MODELS_DIR, 'bottleneck_2_no_mixup_1588680905625058000.h5'),
        manifold_mixup_weights_file='./{}/{}'.format(TRAINED_MODELS_DIR, 'bottleneck_2_manifold_mixup_1588704453481012000.h5')
    )

    show_svd_demo(
        baseline_weights_file='./{}/{}'.format(TRAINED_MODELS_DIR, 'bottleneck_12_no_mixup_1588699264167078000.h5'),
        input_mixup_weights_file='./{}/{}'.format(TRAINED_MODELS_DIR, 'bottleneck_12_input_mixup_1588699435781023000.h5'),
        manifold_mixup_weights_file='./{}/{}'.format(TRAINED_MODELS_DIR, 'bottleneck_12_manifold_mixup_1588702304001298000.h5'),
    )

    show_spiral_demo(
        baseline_model_weights_file='./{}/{}'.format(TRAINED_MODELS_DIR, 'spiral_no_mixup_1588698972067361000.h5'),
        mixup_model_weights_file='./{}/{}'.format(TRAINED_MODELS_DIR, 'spiral_manifold_mixup_1588682681909873000.h5'),
    )


if __name__ == "__main__":
    main()
