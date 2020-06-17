import glob
import os
import uuid

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from tqdm import tqdm

from config import opt
from data import BatteryCap
from utils import Visualizer
import torch as t
from torchnet import meter
import models
import numpy as np


def train(**kwargs):
    """
    训练
    """
    # 根据命令行参数更新配置
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)

    # step1: configure model 模型
    model = getattr(models, opt.model)()  # 最后的()不要忘
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
            print("网络输出的：", score)
            print("-------------------------------")
            print("label", label)
            print("-------------------------------")
            print("softmax后：", t.nn.functional.softmax(score.detach(), dim=1).detach().tolist())
            print("-------------------------------")
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


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


@t.no_grad()  # pytorch>=0.5
def test(**kwargs):
    """
    测试（inference）
    """
    opt._parse(kwargs)

    # configure model  模型
    model = getattr(models, opt.model)().eval()
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
        # print(test_score)
        # print(t.nn.functional.softmax(test_score, dim=1))
        probability = t.nn.functional.softmax(test_score, dim=1).detach()  # 这里改过，github代码有误
        # print("probability:", probability)
        label = probability.max(dim=1)
        batch_results = [(path_, similarity.item(), "不合格" if category.item() == 0 else "合格") for
                         path_, similarity, category in zip(path, label[0], label[1])]
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


def transforms():
    for i in tqdm(glob.glob(os.path.join(opt.train_data_root, '*.bmp'))):
        uid = uuid.uuid1()
        profix = "NEG"
        img = Image.open(i)
        if 'POS' in i.split('/')[-1]:
            profix = "POS"
        p = profix + "-" + str(uid) + "1.bmp"
        data = T.RandomHorizontalFlip(p=1)(img)
        data.save(os.path.join("./data/train", p))
        p = profix + "-" + str(uid) + "2.bmp"
        data = T.RandomVerticalFlip(p=1)(img)
        data.save(os.path.join("./data/train", p))
        p = profix + "-" + str(uid) + "3.bmp"
        data = img.rotate(90)
        data.save(os.path.join("./data/train", p))
        p = profix + "-" + str(uid) + "4.bmp"
        data = img.rotate(270)
        data.save(os.path.join("./data/train", p))
        p = profix + "-" + str(uid) + "5.bmp"
        data = T.RandomRotation(90)(img)
        data.save(os.path.join("./data/train", p))
        p = profix + "-" + str(uid) + "6.bmp"
        data = T.RandomRotation(270)(img)
        data.save(os.path.join("./data/train", p))


def compute_mean_std():
    channel = 0
    std = 0
    filename = os.listdir(opt.train_data_root)
    for i in tqdm(glob.glob(os.path.join(opt.train_data_root, '*.bmp'))):
        img = Image.open(i)
        # h, w = img.size
        # pixels_num += h * w  # 统计单个通道的像素数量
        img = np.array(img)
        # channel += np.sum(img)
        channel += img.mean()
        std += img.std()
        # channel_square += np.sum(np.power(img, 2.0))

    mean = channel / len(filename)

    """   
    S^2
    = sum((x-x')^2 )/N = sum(x^2+x'^2-2xx')/N
    = {sum(x^2) + sum(x'^2) - 2x'*sum(x) }/N
    = {sum(x^2) + N*(x'^2) - 2x'*(N*x') }/N
    = {sum(x^2) - N*(x'^2) }/N
    = sum(x^2)/N - x'^2
    """

    std = std / len(filename)

    print("mean is %f" % (mean))
    print("std is %f" % (std))


def flask():
    from flask import Flask
    from flask import request, Response, render_template
    from gevent import pywsgi
    import io
    import base64
    from flask import jsonify, json

    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        # 获取图片文件 name = upload
        return render_template('index.html', name="demo")

    @app.route('/upload/', methods=['POST'])
    def upload():
        # 获取图片文件 name = upload
        img = request.files['file']
        score_thr = 0.9
        data = Image.open(img)
        data = T.ToTensor()(data)
        img = t.Tensor(1, 1, 1300, 1300)
        img[0] = data
        model = getattr(models, opt.model)()
        model.load(opt.load_model_path)
        device = t.device('cpu')
        model.to(device)
        model.eval()
        score = model(img)
        probability = t.nn.functional.softmax(score, dim=1)[:, 1].detach().tolist()
        # result = inference_detector(model, img)
        # a = np.zeros(shape=(0, 5))
        # result = [result[i] if i == 14 else a for i in range(len(result))]
        # scores = result[14][:, -1]
        # inds = scores > score_thr
        # result[14] = result[14][inds, :]
        # img = show_result(img, result, model.CLASSES, score_thr=score_thr, wait_time=1, show=False)
        # img = Image.fromarray(np.uint8(data))
        # imgByteArr = io.BytesIO()
        # img.save(imgByteArr, format='JPEG')
        # imgByteArr = imgByteArr.getvalue()
        # # 返回图片
        # # m = Model(base64.b64encode(imgByteArr),len(result[14]))
        # # resp = Response(base64.b64encode(imgByteArr), mimetype="application/text")
        # # return jsonify(
        # #     img=base64.b64encode(imgByteArr),
        # #     num=len(result[14]))
        # print(type(base64.b64encode(imgByteArr)))
        # return json.dumps({'img': str(base64.b64encode(imgByteArr), encoding="utf-8"), 'num': len(result[14])}), 200, {
        #     'ContentType': 'application/json'}
        return json.dumps({'type': "合格" if probability[0] >= score_thr else "不合格"}), 200, {
            'ContentType': 'application/json'}

    app.debug = True
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()


if __name__ == '__main__':
    import fire

    fire.Fire()
    # data = t.randn(1, 12, 3, 3)
    # data = t.nn.AvgPool2d(3)(data)
    # print(data.size())

    # data = t.nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1, stride=1, padding=0, bias=False)
    # print(list(data.parameters())[0].data.numpy().shape)
