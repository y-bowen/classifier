# coding:utf8
from torch import nn
from .BasicModule import BasicModule


class AlexNet(BasicModule):
    '''
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    '''

    def __init__(self, input_channel=3, num_classes=2):
        super(AlexNet, self).__init__()

        self.model_name = 'alexnet'

        # self.features = nn.Sequential(
        #     nn.Conv2d(input_channel, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.mp5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)

        )

    def forward(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        print("x-->{}".format(x.size()))
        x = self.conv1(x)
        print("conv1-->{}".format(x.size()))
        x = self.relu1(x)
        print("relu1-->{}".format(x.size()))
        x = self.mp1(x)
        print("mp1-->{}".format(x.size()))
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        print("2-->{}".format(x.size()))
        x = self.conv3(x)
        x = self.relu3(x)
        print("3-->{}".format(x.size()))
        x = self.conv4(x)
        x = self.relu4(x)
        print("4-->{}".format(x.size()))
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.mp5(x)
        print("5-->{}".format(x.size()))
        x = self.classifier(x)
        return x