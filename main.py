import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from config import opt
from data import BatteryCap
from utils import Visualizer
import torch as t
from torchnet import meter
import models

def train(**kwargs):
    """
    训练
    """
    # 根据命令行参数更新配置
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)

    # step1: configure model 模型
    model = getattr(models, opt.model)(opt.inputchannel)  # 最后的()不要忘
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)  # 这一行和书中相比，改过

    # step2: data  数据
    train_dataset = BatteryCap(opt.train_data_root, train=True)  # 训练集
    train_dataloader = DataLoader(train_dataset,
                                  opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    val_dataset = BatteryCap(opt.train_data_root, train=False)  # 交叉验证集
    val_dataloader = DataLoader(val_dataset,
                                opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)

    # step3: criterion and optimizer   目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)
    # step4: meters  统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    # train  训练
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            # train model 训练模型参数
            input_batch = data.to(opt.device)
            label_batch = label.to(opt.device)

            optimizer.zero_grad()  # 梯度清零
            score = model(input_batch)
            loss = criterion(score, label_batch)
            loss.backward()  # 反向传播
            optimizer.step()  # 优化

            # meters update and visualize  更新统计指标及可视化
            loss_meter.add(loss.item())

            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), label_batch.detach())

            if (ii + 1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])  # 先不可视化了!!!
                print('   loss: ', loss_meter.value()[0])

                # 如果需要的话，进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()
        model.save()

        # validate and visualize  计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()), lr=lr))

        # update learning rate  如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

        print('第', str(epoch), '个迭代已结束')
        print("验证集准确率为： ", str(val_accuracy))
        print('---' * 50)


@t.no_grad()
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息，用以辅助训练
    """
    # 把模型设为验证模式
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))

    # 把模型恢复为训练模式
    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    print('accuracy: ', str(accuracy))

    return confusion_matrix, accuracy

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

@t.no_grad()  # pytorch>=0.5
def test(**kwargs):
    """
    测试（inference）
    """
    opt._parse(kwargs)

    # configure model  模型
    model = getattr("models", opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # data  数据
    test_data = BatteryCap(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)

    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        test_input = data.to(opt.device)
        test_score = model(test_input)
        probability = t.nn.functional.softmax(test_score, dim=1)[:, 1].detach().tolist()  # 这里改过，github代码有误
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results

    write_csv(results, opt.result_file)

    return results


def help():
    """
    打印帮助的信息
    """
    print("""
        usage : python {0} <function> [--args=value]
        <function> := train | test | help
        example: 
                python {0} train --env='env0701' --lr=0.01
                python {0} test --dataset='path/to/dataset/root/'
                python {0} help
        avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))  # DefaultConfig类
    print(source)


if __name__ == '__main__':
    # import fire

    # fire.Fire()
    loader = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()])
    data = Image.open("./data/train/POS-4441756.bmp")
    data = loader(data)
    data = data.view(1, 1, 224, 224)
    c = torch.nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
    data = c(data)
    print(data.size())
