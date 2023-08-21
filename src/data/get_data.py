from typing import Optional
import tensorflow as tf
import tensorflow_datasets as tfds
from src.configs.configs import DataConfig

keras = tf.keras
K = keras.backend

_IMAGE_SIZE_IMAGENET = (224, 224)
_IMAGE_SIZE_CIFAR = (32, 32)
_IMAGE_SIZE_MNIST = (28, 28)


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
        split = split.map(
            lambda item: (item["image"], item["label"]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        split = split.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
        split = split.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        return split

    train_ds = prepare(train_ds)
    test_ds = prepare(test_ds)
    return train_ds, test_ds


def _load_cifar10(data_config: DataConfig):
    train_ds, test_ds = tfds.load(
        "cifar10",
        split=["train", "test"],
        data_dir=data_config.path,
        shuffle_files=True,
        as_supervised=True,
    )
    # ensure all images have same size
    resizer = tf.keras.layers.Resizing(*_IMAGE_SIZE_CIFAR)

    def resize(img, label):
        img /= 255
        return resizer(img), label

    def one_hot(image, label):
        # Casts to an Int and performs one-hot ops
        label = tf.one_hot(tf.cast(label, tf.int32), data_config.num_classes)
        # Recasts it to Float32
        label = tf.cast(label, tf.float32)
        return image, label

    def prepare(split):
        split = split.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
        split = split.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        return split

    train_ds = prepare(train_ds)
    test_ds = prepare(test_ds)
    return train_ds, test_ds


def _load_cifar100(data_config: DataConfig):
    train_ds, test_ds = tfds.load(
        "cifar100",
        split=["train", "test"],
        data_dir=data_config.path,
        shuffle_files=True,
        as_supervised=True,
    )
    # ensure all images have same size
    resizer = tf.keras.layers.Resizing(*_IMAGE_SIZE_CIFAR)

    def resize(img, label):
        img /= 255
        return resizer(img), label

    def one_hot(image, label):
        # Casts to an Int and performs one-hot ops
        label = tf.one_hot(tf.cast(label, tf.int32), data_config.num_classes)
        # Recasts it to Float32
        label = tf.cast(label, tf.float32)
        return image, label

    def prepare(split):
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
    @return: train_ds, test_ds
    """
    train_ds, test_ds = tfds.load(
        "mnist",
        split=["train", "test"],
        data_dir=data_config.path,
        shuffle_files=True,
        as_supervised=True,
    )
    # ensure all images have same size
    resizer = tf.keras.layers.Resizing(*_IMAGE_SIZE_MNIST)

    def resize(img, label):
        img /= 255
        return resizer(img), label

    def one_hot(image, label):
        # Casts to an Int and performs one-hot ops
        label = tf.one_hot(tf.cast(label, tf.int32), data_config.num_classes)
        # Recasts it to Float32
        label = tf.cast(label, tf.float32)
        return image, label

    def prepare(split):
        split = split.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
        split = split.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        return split

    train_ds = prepare(train_ds)
    test_ds = prepare(test_ds)

    return train_ds, test_ds


def _load_fashion_mnist(data_config: DataConfig):
    """
    Loads the Fashion-MNIST dataset.
    @param data_config:
    @return: train_ds, test_ds
    """
    train_ds, test_ds = tfds.load(
        "fashion_mnist",
        split=["train", "test"],
        data_dir=data_config.path,
        shuffle_files=True,
        as_supervised=True,
    )
    # ensure all images have same size
    resizer = tf.keras.layers.Resizing(*_IMAGE_SIZE_MNIST)

    def resize(img, label):
        img /= 255
        return resizer(img), label

    def one_hot(image, label):
        # Casts to an Int and performs one-hot ops
        label = tf.one_hot(tf.cast(label, tf.int32), data_config.num_classes)
        # Recasts it to Float32
        label = tf.cast(label, tf.float32)
        return image, label

    def prepare(split):
        split = split.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
        split = split.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        return split

    train_ds = prepare(train_ds)
    test_ds = prepare(test_ds)

    return train_ds, test_ds


def fetch_data(
    data_config: DataConfig,
) -> tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
    """
    Returns the reference/training data from file.

    It will create the data if it's not already in the expected location.
    @param data_config:
    @return: train_data, test_data
    """
    data_config.path.mkdir(parents=True, exist_ok=True)

    lowercase_name = data_config.name.lower()

    if lowercase_name == "mnist":
        ds_train, ds_test = _load_mnist(data_config)
        return ds_train, ds_test
    elif lowercase_name == "fashion_mnist":
        ds_train, ds_test = _load_fashion_mnist(data_config)
        return ds_train, ds_test
    elif lowercase_name == "imagenet":
        ds_train, ds_test = _load_imagenet(data_config)
        return ds_train, ds_test
    elif lowercase_name == "cifar100":
        ds_train, ds_test = _load_cifar100(data_config)
        return ds_train, ds_test
    elif lowercase_name == "cifar10":
        ds_train, ds_test = _load_cifar10(data_config)
        return ds_train, ds_test
    else:
        # TODO: add other datasets
        return
