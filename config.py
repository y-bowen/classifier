# coding:utf-8
import warnings
import torch as t


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port = 8097  # visdom 端口
    # model = 'AlexNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致   SqueezeNet -> ResNet34
    model = 'ResNet50'
    inputchannel = 1  # 输入图片通道
    train_data_root = './data/train/'  # 训练集存放路径   default: './data/train/'
    test_data_root = './data/test1'  # 测试集存放路径     default: './data/test1'
    load_model_path = "./checkpoints/resnet50_06-1_13.56.09.pth"  # 加载预训练的模型的路径，为None代表不加载  default: None

    batch_size = 4  # batch size   default: 32  4
    use_gpu = True  # user GPU or not
    num_workers = 2  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 20  # default: 10  100
    lr = 0.001  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay   default: 0.5
    weight_decay = 0e-5  # 损失函数  default: 0e-5    1e-5


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于个人喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')  # 原来没有这一行

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()