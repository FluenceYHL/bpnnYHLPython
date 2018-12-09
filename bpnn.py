# package
import cv2
import math
import time
import numpy
import datetime
import warnings
import matplotlib.pyplot as plt
# self
import classfiers.samples
import images.fileFilter


def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


def dSigmoid(x):
    return x * (1.0 - x)

# 可以继续优化 —— SGD 优化器, 学习率震荡


class BPNN():
    def __init__(self, inputSize=784, hideSize=100, outputSize=10, lrt=0.02):
        self.inputSize = inputSize
        self.hideSize = hideSize
        self.outputSize = outputSize
        self.lrt = lrt
        self.whid = numpy.random.uniform(-0.5,
                                         0.5, (self.inputSize, self.hideSize))
        self.wout = numpy.random.uniform(-0.5,
                                         0.5, (self.hideSize, self.outputSize))
        self.hid_out = numpy.zeros(self.inputSize)
        self.out_out = numpy.zeros(self.outputSize)

    def forward(self, inpt):
        self.hid_out = sigmoid(numpy.dot(inpt, self.whid))
        self.out_out = sigmoid(numpy.dot(self.hid_out, self.wout))
        return self.out_out

    def backward(self, inpt, target):
        out_daoshu = (target - self.out_out) * dSigmoid(self.out_out)
        self.wout += self.lrt * \
            self.hid_out.reshape(self.hideSize, 1) * out_daoshu
        hid_daoshu = dSigmoid(self.hid_out) * numpy.dot(self.wout, out_daoshu)
        self.whid += self.lrt * inpt.reshape(self.inputSize, 1) * hid_daoshu

    def recognize(self, inpt):
        return numpy.argmax(self.forward(inpt))


class Mnist():
    def __init__(self):
        self.bpnn = BPNN(784, 100, 10, 0.13)

    def train_and_test(self):
        train_features, train_labels, test_features, test_labels = classfiers.samples.Samples(
        ).get_npy('./npys/train_mnist.npy', './npys/test_mnist.npy')
        train_features.resize(train_features.shape[:2])
        test_features.resize(test_features.shape[:2])

        start = datetime.datetime.now()
        effiency = []
        for i in range(20):
            cnt = 0
            for inpt in train_features:
                target = numpy.zeros(10)
                target[int(train_labels[cnt])] = 1
                self.bpnn.forward(inpt)
                self.bpnn.backward(inpt, target)
                cnt = cnt + 1
            cnt = 0
            correct = 0
            for inpt in test_features:
                judge = self.bpnn.recognize(inpt)
                if(judge == int(test_labels[cnt])):
                    correct = correct + 1
                cnt = cnt + 1
            accuracy = correct / len(test_features)
            print("准确率  :  " + str(accuracy))
            effiency.append([i, accuracy])
        plt.plot(numpy.array(effiency)[:, 0], numpy.array(effiency)[:, 1])
        plt.show()

        print(datetime.datetime.now() - start)
        numpy.save('./npys/yhl_whid.npy', self.bpnn.whid)
        numpy.save('./npys/yhl_wout.npy', self.bpnn.wout)


class Self():
    def __init__(self, inputSize, hideSize, outputSize, lrt):
        self.inputSize = inputSize
        self.hideSize = hideSize
        self.outputSize = outputSize
        self.lrt = lrt
        self.bpnn = BPNN(inputSize, hideSize, outputSize, lrt)

    def train_and_test(self, train_file, train_answers, test_file, test_answers, iters=1):
        train_labels = numpy.load(train_answers)
        test_labels = numpy.load(test_answers)
        file_list = images.fileFilter.filter(train_file)
        test_list = images.fileFilter.filter(test_file)

        for it in range(iters):
            count = 0
            for path in file_list:
                img = cv2.imread(path, 0)
                img.resize(self.inputSize)
                target = numpy.zeros(self.outputSize)
                target[int(train_labels[count])] = 1
                count = count + 1
                self.bpnn.forward(img)
                self.bpnn.backward(img, target)
            count = 0
            correct = 0
            for path in test_list:
                img = cv2.imread(path, 0)
                img.resize(self.inputSize)
                judge = self.bpnn.recognize(img)
                if(judge == test_labels[count]):
                    correct = correct + 1
                count = count + 1
            accuracy = correct / len(test_labels)
            print("准确率  :  " + str(accuracy))

        numpy.save("./npys/whid.npy", self.bpnn.whid)
        numpy.save("./npys/wout.npy", self.bpnn.wout)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    two = Self(784, 100, 10, 0.01)
    two.train_and_test('./images/handled', './npys/answers.npy',
                       './images/handled', './npys/answers.npy', 40)
