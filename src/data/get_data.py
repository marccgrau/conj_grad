from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_file

from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

from src.configs.configs import DataConfig

keras = tf.keras
K = keras.backend

_IMAGE_SIZE_IMAGENET = (224, 224)
_IMAGE_SIZE_CIFAR = (32, 32)

def _load_imagenet(data_config: DataConfig): 
    train_ds, test_ds = tfds.load(
        "imagenet2012", 
        split=["train", "validation"], 
        shuffle_files=True, 
        ) 
    # Imagenet dataset has arbitrary sizes, we need to resize all of them. 
    resizer = tf.keras.layers.Resizing(*_IMAGE_SIZE_IMAGENET) 
    
    def resize(img, label): 
        return resizer(img), label 
    
    def one_hot(image, label):
        # Casts to an Int and performs one-hot ops
        label = tf.one_hot(tf.cast(label, tf.int32), data_config.num_classes)
        # Recasts it to Float32
        label = tf.cast(label, tf.float32)
        return image, label
    
    def prepare(split): 
        split = split.map( lambda item: (item["image"], item["label"]), num_parallel_calls=tf.data.AUTOTUNE, ) 
        split = split.map(resize, num_parallel_calls=tf.data.AUTOTUNE) 
        split = split.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE) 
        return split 
    
    train_ds = prepare(train_ds) 
    test_ds = prepare(test_ds) 
    return train_ds, test_ds

def _load_cifar10(data_config: DataConfig): 
    train_ds, test_ds = tfds.load(
        "cifar10", 
        split=["train", "validation"], 
        shuffle_files=True, 
        as_supervised=True,
        ) 
    # Imagenet dataset has arbitrary sizes, we need to resize all of them. 
    resizer = tf.keras.layers.Resizing(*_IMAGE_SIZE_CIFAR) 
    
    def resize(img, label): 
        return resizer(img), label 
    
    def one_hot(image, label):
        # Casts to an Int and performs one-hot ops
        label = tf.one_hot(tf.cast(label, tf.int32), data_config.num_classes)
        # Recasts it to Float32
        label = tf.cast(label, tf.float32)
        return image, label
    
    def prepare(split): 
        split = split.map( lambda item: (item["image"], item["label"]), num_parallel_calls=tf.data.AUTOTUNE, ) 
        split = split.map(resize, num_parallel_calls=tf.data.AUTOTUNE) 
        split = split.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE) 
        return split 
    
    train_ds = prepare(train_ds) 
    test_ds = prepare(test_ds) 
    return train_ds, test_ds

def _load_cifar100(data_config: DataConfig): 
    train_ds, test_ds = tfds.load(
        "cifar100", 
        split=["train", "validation"], 
        shuffle_files=True, 
        as_supervised=True,
        ) 
    # Imagenet dataset has arbitrary sizes, we need to resize all of them. 
    resizer = tf.keras.layers.Resizing(*_IMAGE_SIZE_CIFAR) 
    
    def resize(img, label): 
        return resizer(img), label 
    
    def one_hot(image, label):
        # Casts to an Int and performs one-hot ops
        label = tf.one_hot(tf.cast(label, tf.int32), data_config.num_classes)
        # Recasts it to Float32
        label = tf.cast(label, tf.float32)
        return image, label
    
    def prepare(split): 
        split = split.map( lambda item: (item["image"], item["label"]), num_parallel_calls=tf.data.AUTOTUNE, ) 
        split = split.map(resize, num_parallel_calls=tf.data.AUTOTUNE) 
        split = split.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE) 
        return split 
    
    train_ds = prepare(train_ds) 
    test_ds = prepare(test_ds) 
    return train_ds, test_ds

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
    elif "IMAGENET" in data_config.name:
        ds_train, ds_test = _load_imagenet(data_config)
        return ds_train, ds_test
    elif "CIFAR10" in data_config.name:
        ds_train, ds_test = _load_cifar10(data_config)
        return ds_train, ds_test
    elif "CIFAR100" in data_config.name:
        ds_train, ds_test = _load_cifar100(data_config)
        return ds_train, ds_test
    else:
        # TODO: add other datasets
        pass

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if all(v is not None for v in (x_test, y_test)):
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    else:
        ds_test = None
    return ds_train, ds_test
