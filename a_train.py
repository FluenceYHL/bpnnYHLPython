# -*- coding:utf-8 -*-

# self
import images.fileFilter
# package
import numpy as np
import imageio
import math
import warnings
import cv2
from sympy import *
from numba import jit
import matplotlib.pyplot as plt
from skimage import data, color, filters


@jit
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


class Neural_Networks():
    def __init__(self, inputnode=784, hiddennode=200, outputnode=10, learningrate=0.1):
        self.inpt = inputnode  # 785
        self.hid = hiddennode  # 200
        self.out = outputnode  # 10
        self.lrt = learningrate
        # 初始化权值
        self.o_thre = np.random.rand(1, self.out)  # 785 random
        self.h_thre = np.random.rand(1, self.hid)  # 200
        self.whid = np.random.normal(
            0.0, self.out ** -0.5, (self.inpt, self.hid))
        self.wout = np.random.normal(
            0.0, self.hid ** -0.5, (self.hid, self.out))
        # 图像处理
        infile = '1.png'
        outfile = str(infile)

    # 图片归一化  返回1*784的向量
    def convert_to_binary(self, myimage):
        thresh = filters.threshold_li(myimage)
        dst = (myimage <= thresh) * 1.0
        dst = np.asarray(dst)
        dst = dst.reshape(1, self.inpt)
        return dst

    def forward_back(self, inpt, target):
        hid_in = np.dot(inpt, self.whid)  # 1*200
        hid_out = sigmoid(hid_in)  # (hid_in - self.h_thre)        #1*200
        out_in = np.dot(hid_out, self.wout)  # 1*10
        target = target.reshape(1, 10)
        # inpt = inpt.reshape(1, self.inpt)
        hid_out = hid_out.reshape(1, self.hid)
        out_out = sigmoid(out_in)  # (out_in - self.o_thre)        #1*10
        out_out = out_out.reshape(1, 10)
        out_daoshu = (target[0] - out_out[0]) * \
            out_out[0] * (1 - out_out[0])
        self.wout += self.lrt * hid_out.T * out_daoshu
        sigma = np.dot(self.wout, out_daoshu)  # 100，1
        hid_daoshu = hid_out * (1 - hid_out) * sigma  # 100，1
        self.whid += self.lrt * inpt.T * hid_daoshu

    # 权值更新
    def bp_learning(self, training_data):
        # count = 0
        for data in training_data:
            target = np.zeros(self.out) + 0.01
            target[int(data[0])] = 0.99
            inpt = np.asfarray(data[1:])
            inpt = self.convert_to_binary(inpt.reshape(28, 28))
            self.forward_back(inpt, target)

    # 单步权值更新
    def one_step_training(self, data, mytarget):
        target = np.zeros(self.out) + 0.01
        target[int(mytarget)] = 0.99
        inpt = self.convert_to_binary(data)
        # image = inpt.reshape(28, 28)
        # plt.imshow(image)
        self.forward_back(inpt, target)

    # 判断输入图片所属类别  返回int值（0-9）
    def judge(self, inpt):
        hid_in = np.dot(inpt, self.whid)  # 1*200
        hid_out = sigmoid(hid_in)  # (hid_in - self.h_thre)        #1*200
        out_in = np.dot(hid_out, self.wout)  # 1*10
        out_out = sigmoid(out_in)  # (out_in - self.o_thre)        #1*10
        out_out = out_out.reshape(1, 10)
        out = np.argmax(out_out)
        return out

    # 测试minst数据集 返回正确率
    def mni_test(self, input):
        correct = 0
        for data in input:
            target = np.zeros(self.out) + 0.01
            target[int(data[0])] = 0.99
            inpt = np.asfarray(data[1:])
            inpt = inpt.reshape(28, 28)
            inpt = self.convert_to_binary(inpt)
            out_out = self.judge(inpt)
            if int(out_out) == int(data[0]):
                correct += 1
        rate = correct / len(input)
        return rate


def train():
    input_node = 784
    hidden_node = 300
    output_node = 10
    learning_rate = 0.13  # ,96%,3times
    x = []  # learning rate
    y = []  # correct rate
    # for learning_rate in np.arange(0.12, 0.12):
    nn = Neural_Networks(input_node, hidden_node, output_node, learning_rate)
    print('Learning rate is :' + str(nn.lrt))
    testdata = np.load('./npys/test_mnist.npy')
    rate_before = nn.mni_test(testdata)

    data = np.load('./npys/train_mnist.npy')
    training_time = 3  # 使用training data训练3次
    for i in range(training_time):
        nn.bp_learning(data)
        print(str(i + 1) + ' training step is over')

    np.save("./npys/whid.npy", nn.whid)
    np.save("./npys/wout.npy", nn.wout)
    rate_after = nn.mni_test(testdata)
    print('Before training, the correct rate is ' + str(rate_before) + '\nAfter training, the correct rate is ' + str(
        rate_after))
    x.append(learning_rate)
    y.append(rate_after)
    return x, y, n


def bpnnTrain():
    input_node = 784
    hidden_node = 100
    output_node = 10
    learning_rate = 0.02
    bpnn = Neural_Networks(input_node, hidden_node, output_node, learning_rate)
    answers = np.load("./npys/answers.npy")
    fileList = images.fileFilter.filter('./images/handled')
    # print(fileList)
    for it in range(40):
        count = 0
        for path in fileList:
            img = cv2.imread(path, 0)
            bpnn.one_step_training(img, answers[count])
            count = count + 1

        print(str(it) + '  is over')

    count = 0
    correct = 0
    for path in fileList:
        img = cv2.imread(path, 0)
        inpt = bpnn.convert_to_binary(img)
        target = np.zeros(output_node) + 0.01
        target[answers[count]] = 0.99
        out_out = bpnn.judge(inpt)
        if int(out_out) == answers[count]:
            correct += 1
        count = count + 1
    print('正确率　　:  ' + str(correct / count))
    np.save("./npys/whid.npy", bpnn.whid)
    np.save("./npys/wout.npy", bpnn.wout)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    bpnnTrain()
    # x, y, nn = train()
    # plt.plot(x, y, 'r')
    # plt.xlabel('learning rate')
    # plt.ylabel('correct rate')
    # plt.title('test plot')
    # plt.show()
