from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils.data_utils import get_file

from tensorflow.keras.datasets import mnist

from src.configs.configs import DataConfig

keras = tf.keras
K = keras.backend


def _load_mnist(data_config: DataConfig):
    """
    Loads the MNIST dataset.
    @param data_config:
    @return: x_train, y_train, x_test, y_test
    """
    path = data_config.path
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path)
    x_train = np.reshape(x_train, (60000, 28, 28, 1)).astype("float32")
    x_test = np.reshape(x_test, (10000, 28, 28, 1)).astype("float32")
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    assert x_train.shape == (60000, 28, 28, 1)
    assert x_test.shape == (10000, 28, 28, 1)
    assert y_train.shape == (60000, 10)
    assert y_test.shape == (10000, 10)
    return x_train, y_train, x_test, y_test

def fetch_data(
    data_config: DataConfig
) -> tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
    """
    Returns the reference/training data from file.

    It will create the data if it's not already in the expected location.
    @param data_config:
    @return: train_data, test_data
    """
    data_config.path.mkdir(parents=True, exist_ok=True)

    if "MNIST" in data_config.name:
        x_train, y_train, x_test, y_test = _load_mnist(data_config)
    else:
        # TODO: add other datasets
        pass

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if all(v is not None for v in (x_test, y_test)):
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    else:
        ds_test = None
    return ds_train, ds_test
