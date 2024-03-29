from src.models.basic_cnn import BasicCNN
from src.models.resnet import ResNetTypeI, ResNetTypeII, ResNetCIFAR
from src.models.cifar_models import CIFARCNN
from src.models.flat_cnn import FlatCNN
from src.models.flat_cnn_cifar100 import FlatCNNCifar100
from src.models.flat_mlp import FlatMLP
import tensorflow as tf
from typing import Optional


def get_model(
    model_name: str,
    num_classes: int,
    num_units_mlp: Optional[int] = None,
    num_base_filters: Optional[int] = None,
    model_size: str = "small",
    seed: int = 42,
) -> tf.keras.Model:
    """_summary_

    Args:
        model_name (str): choose architecture
        num_classes (int): number of classes to predict
        num_units_mlp (Optional[int], optional): MLP hidden layer neurons. Defaults to None.
        num_base_filters (Optional[int], optional): CNN filters. Defaults to None.
        model_size (str, optional): Large or small model to construct. Defaults to "small".
        seed (int, optional): seed for initialization. Defaults to 42.


    Returns:
        tf.keras.Model: model
    """
    if model_name == "FlatMLP":
        return flat_mlp(
            num_classes=num_classes,
            num_units_mlp=num_units_mlp,
            model_size=model_size,
            seed=seed,
        )
    elif model_name == "FlatCNN":
        return flat_cnn(
            num_classes=num_classes,
            num_base_filters=num_base_filters,
            model_size=model_size,
            seed=seed,
        )
    elif model_name == "FlatCNNCifar100":
        return flat_cnn_cifar100(
            num_classes=num_classes,
            num_base_filters=num_base_filters,
            model_size=model_size,
            seed=seed,
        )
    elif model_name == "BasicCNN":
        return basic_cnn(num_classes=num_classes)
    elif model_name == "CIFARCNN":
        return cifar_cnn(num_classes=num_classes)
    elif model_name == "ResNetTypeI_18":
        return resnet_18(num_classes=num_classes)
    elif model_name == "ResNetTypeI_34":
        return resnet_34(num_classes=num_classes)
    elif model_name == "ResNetTypeII_50":
        return resnet_50(num_classes=num_classes)
    elif model_name == "ResNetTypeII_101":
        return resnet_101(num_classes=num_classes)
    elif model_name == "ResNetTypeII_152":
        return resnet_152(num_classes=num_classes)
    elif model_name == "ResNetCIFAR":
        return resnet_cifar(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not found")


def flat_cnn(
    num_classes: int,
    num_base_filters: int,
    model_size: str,
    seed: int,
) -> tf.keras.Model:
    return FlatCNN(
        num_classes=num_classes,
        num_base_filters=num_base_filters,
        model_size=model_size,
        seed=seed,
    )


def flat_cnn_cifar100(
    num_classes: int,
    num_base_filters: int,
    model_size: str,
    seed: int,
):
    return FlatCNNCifar100(
        num_classes=num_classes,
        num_base_filters=num_base_filters,
        model_size=model_size,
        seed=seed,
    )


def flat_mlp(
    num_classes: int,
    num_units_mlp: int,
    model_size: str,
    seed: int,
) -> tf.keras.Model:
    return FlatMLP(
        num_classes=num_classes,
        num_units_mlp=num_units_mlp,
        model_size=model_size,
        seed=seed,
    )


def basic_cnn(num_classes: int) -> tf.keras.Model:
    return BasicCNN(num_classes=num_classes)


def cifar_cnn(num_classes: int) -> tf.keras.Model:
    return CIFARCNN(num_classes=num_classes)


def resnet_18(num_classes: int) -> tf.keras.Model:
    return ResNetTypeI(layer_params=[2, 2, 2, 2], num_classes=num_classes)


def resnet_34(num_classes: int) -> tf.keras.Model:
    return ResNetTypeI(layer_params=[3, 4, 6, 3], num_classes=num_classes)


def resnet_50(num_classes: int) -> tf.keras.Model:
    return ResNetTypeII(layer_params=[3, 4, 6, 3], num_classes=num_classes)


def resnet_101(num_classes: int) -> tf.keras.Model:
    return ResNetTypeII(layer_params=[3, 4, 23, 3], num_classes=num_classes)


def resnet_152(num_classes: int) -> tf.keras.Model:
    return ResNetTypeII(layer_params=[3, 8, 36, 3], num_classes=num_classes)


def resnet_cifar(num_classes: int) -> tf.keras.Model:
    return ResNetCIFAR(layer_params=[1, 1, 1], num_classes=num_classes)
