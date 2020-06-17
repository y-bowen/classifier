# coding:utf-8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class BatteryCap(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # 训练集和验证集的文件命名不一样
        # if self.test:
            # imgs = sorted(imgs, key=lambda x: int(x.split('.')[0].split('/')[-1]))
        # else:
            # imgs = sorted(imgs, key=lambda x: int(x.split('.')[0]))

        self.imgs_num = len(imgs)

        # shuffle imgs   看作者知乎代码加的
        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        # 划分训练、验证集，训练：验证 = 7：3
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * self.imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * self.imgs_num):]  # 训练集中的30%用作验证集

        if transforms is None:
            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean=[140.10686],
                                    std=[80.187306])  # 怎么来的？？

            # 测试集和验证集不用数据增强
            if self.test or not train:
                self.transforms = T.Compose([
                    # T.Resize(1300),
                    # T.CenterCrop(224),
                    T.ToTensor(),
                    # normalize
                ])
            # 训练集需要数据增强
            else:
                self.transforms = T.Compose([
                    # T.Resize(1300),
                    # T.RandomResizedCrop(224),  # 有改动 RandomReSizedCrop -> RandomResizedCrop
                    # T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    # normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        如果是测试集，没有图片的id，如1000.jpg返回1000
        """
        img_path = self.imgs[index]
        if self.test:
            # TODO 测试部分还有问题
            label = self.imgs[index].split('.')[-2].split('/')[-1]
            # label = index
        else:
            label = 1 if 'POS' in img_path.split('/')[-1] else 0  # 合格1 不合格0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        '''
                返回数据集中所有图片的个数
                :return:
                '''
        return len(self.imgs)
