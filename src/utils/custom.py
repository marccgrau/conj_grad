# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

keras = tf.keras
K = keras.backend

__all__ = [
    "as_Kfloat",
    "symmetric_sigmoid",
    "symmetric_sigmoid_h",
    "get_random_numbers",
    "CustomInitializer",
    "mse",
    "crossentropy",
    "bin_crossentropy",
    "bin_accuracy",
    "accuracy",
]


# noinspection PyPep8Naming
def as_Kfloat(x: float) -> np.float16 | np.float32 | np.float64:
    """
    Returns the given float in the current precision.
    @param x:
    @return:
    """
    return np.float64(x).astype(K.floatx())


# noinspection PyTypeChecker
@tf.function(jit_compile=True)
def symmetric_sigmoid(x: tf.Tensor) -> tf.Tensor:
    """
    Activation function for a NN layer: the symmetric sigmoid.

    @param x: the tensor to work on
    @return: element-wise applied symmetric sigmoid
    """
    return (2 / (1 + K.exp(-2 * x))) - 1


h = as_Kfloat(0.05)


# noinspection PyTypeChecker
@tf.function(jit_compile=True)
def symmetric_sigmoid_h(x: tf.Tensor) -> tf.Tensor:
    return (1 - h) * symmetric_sigmoid(x) + h * x


def get_random_numbers(shape) -> tf.Tensor:
    return K.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=K.floatx())

@tf.function
def mse(y_true, y_pred):
    return tf.reduce_mean(keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred))


@tf.function
def crossentropy(y_true, y_pred):
    return tf.reduce_mean(
        keras.losses.categorical_crossentropy(
            y_true=y_true, y_pred=y_pred, from_logits=True
        )
    )


@tf.function
def bin_crossentropy(y_true, y_pred):
    return tf.reduce_mean(
        keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)
    )


@tf.function
def accuracy(y_true, y_pred):
    return tf.reduce_mean(keras.metrics.categorical_accuracy(y_true, y_pred))


@tf.function
def bin_accuracy(y_true, y_pred):
    return tf.reduce_mean(keras.metrics.binary_accuracy(y_true, y_pred))
