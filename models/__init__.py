from .AlexNet import AlexNet
from .ResNet34 import ResNet34
from .ResNet50 import ResNet50, ResNet101, ResNet152
from efficientnet_pytorch import EfficientNet
efficientnet_b0 = EfficientNet.from_name("efficientnet-b0")
efficientnet_b1 = EfficientNet.from_name("efficientnet-b1")
efficientnet_b2 = EfficientNet.from_name("efficientnet-b2")
efficientnet_b3 = EfficientNet.from_name("efficientnet-b3")
efficientnet_b4 = EfficientNet.from_name("efficientnet-b4")
efficientnet_b5 = EfficientNet.from_name("efficientnet-b5")
efficientnet_b6 = EfficientNet.from_name("efficientnet-b6")
efficientnet_b7 = EfficientNet.from_name("efficientnet-b7")