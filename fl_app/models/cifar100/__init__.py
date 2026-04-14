"""CIFAR-100 модели: WRN-16-4 и SE-ResNet."""

from .wrn import WideResNet
from .se_resnet import CifarSEResNet

__all__ = ["WideResNet", "CifarSEResNet"]
