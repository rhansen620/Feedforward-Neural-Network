"""
The main code for the feedforward networks assignment.
See README.md for details.
"""
from typing import Tuple, Dict

import tensorflow

def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """
    #deep model
    deep_model = tensorflow.keras.Sequential()
    deep_model.add(tensorflow.keras.layers.Dense(8, input_shape=(n_inputs,)))
    deep_model.add(tensorflow.keras.layers.Dense(8))
    deep_model.add(tensorflow.keras.layers.Dense(8))
    deep_model.add(tensorflow.keras.layers.Dense(n_outputs, activation='linear'))
    deep_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-2),
              loss="mse", metrics=["mae"])
    #wide model
    wide_model = tensorflow.keras.Sequential()
    wide_model.add(tensorflow.keras.layers.Dense(10, input_shape=(n_inputs,)))
    wide_model.add(tensorflow.keras.layers.Dense(10))
    wide_model.add(tensorflow.keras.layers.Dense(n_outputs,activation= 'linear'))
    wide_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-2),
              loss="mse", metrics=["mae"])
    return deep_model, wide_model


def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """
    #relu model
    relu_model = tensorflow.keras.Sequential()
    relu_model.add(tensorflow.keras.layers.Dense(10, activation='relu', input_shape=(n_inputs,)))
    relu_model.add(tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid'))
    relu_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3),
              loss="hinge")
    #tanh model
    tanh_model = tensorflow.keras.Sequential()
    tanh_model.add(tensorflow.keras.layers.Dense(10, activation='tanh', input_shape=(n_inputs,)))
    tanh_model.add(tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid'))
    tanh_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3),
              loss="hinge")
    return relu_model, tanh_model


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """
    #dropout model
    dropout_model = tensorflow.keras.Sequential()
    dropout_model.add(tensorflow.keras.layers.Dense(8, input_shape=(n_inputs,)))
    dropout_model.add(tensorflow.keras.layers.Dropout(.2))
    dropout_model.add(tensorflow.keras.layers.Dense(n_outputs,
        activation=tensorflow.keras.activations.softmax))
    dropout_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3),
              loss="categorical_crossentropy")
    #nodropout model
    nodropout_model = tensorflow.keras.Sequential()
    nodropout_model.add(tensorflow.keras.layers.Dense(8, input_shape=(n_inputs,)))
    nodropout_model.add(tensorflow.keras.layers.Dense(n_outputs,
        activation=tensorflow.keras.activations.softmax))
    nodropout_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3),
              loss="categorical_crossentropy")
    return dropout_model, nodropout_model


def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                Dict,
                                                tensorflow.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """
    #early stopping model
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
    early_model = tensorflow.keras.Sequential()
    early_model.add(tensorflow.keras.layers.Dense(12, input_shape=(n_inputs,)))
    early_model.add(tensorflow.keras.layers.Dense(n_outputs,activation='sigmoid'))
    early_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-2),
              loss="binary_crossentropy")
    earlystopping_dict = {"callbacks":[early_stopping]}
    #no early stopping model
    noearly_model = tensorflow.keras.Sequential()
    noearly_model.add(tensorflow.keras.layers.Dense(12, input_shape=(n_inputs,)))
    noearly_model.add(tensorflow.keras.layers.Dense(n_outputs,activation='sigmoid'))
    noearly_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-2),
              loss="binary_crossentropy")
    return early_model, earlystopping_dict, noearly_model, {}
    