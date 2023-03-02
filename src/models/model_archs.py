from src.models.basic_cnn import BasicCNN
from src.models.resnet import ResNetTypeI, ResNetTypeII

def basic_cnn(num_classes):
    return BasicCNN(num_classes=num_classes)


def resnet_18(num_classes):
    return ResNetTypeI(layer_params=[2, 2, 2, 2], num_classes=num_classes)


def resnet_34(num_classes):
    return ResNetTypeI(layer_params=[3, 4, 6, 3], num_classes=num_classes)


def resnet_50(num_classes):
    return ResNetTypeII(layer_params=[3, 4, 6, 3], num_classes=num_classes)


def resnet_101(num_classes):
    return ResNetTypeII(layer_params=[3, 4, 23, 3], num_classes=num_classes)


def resnet_152(num_classes):
    return ResNetTypeII(layer_params=[3, 8, 36, 3], num_classes=num_classes)