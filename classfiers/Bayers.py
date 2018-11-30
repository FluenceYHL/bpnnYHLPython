# package
import cv2
import math
import time
import numpy
import datetime
# self
import samples


class Naive_Bayers():
    def __init__(self, labels=10, feature_len=784, feature_value=2):
        self.labels = labels
        self.feature_len = feature_len
        self.feature_value = feature_value
        self.prior = numpy.zeros(self.labels, dtype=float)  # 　先验概率
        self.conditional = numpy.zeros(
            (self.labels, self.feature_len, self.feature_value))  # 条件概率
        self.category = set([])  # 　所有分类

    def set_argment(self, labels, feature_len, feature_value):
        self.labels = labels
        self.feature_len = feature_len
        self.feature_value = feature_value

    def train(self, train_features, train_labels):
        # 根据数据集调整自身参数
        length = len(train_labels)
        train_features.reshape(
            train_features.shape[0], train_features.shape[1])
        value = set(train_features[0].reshape(784))
        self.category = set(train_labels)
        self.set_argment(len(self.category),
                         train_features.shape[1], len(value))
        # 计算每种标签的先验概率
        self.prior = numpy.zeros(self.labels, dtype=float)
        for it in train_labels:
            self.prior[int(it)] += 1
        for it in self.category:
            self.prior[int(it)] = (self.prior[int(it)] + 1) / \
                (length + self.labels)  # 分子　+ 1 / 分母　+ 类别 (拉普拉斯平滑)

        # 计算每种标签 k, 第　i 个特征,　取值为 train_features[i][j] 的条件概率
        self.conditional = numpy.zeros(
            (self.labels, self.feature_len, self.feature_value))
        for i in range(length):
            k = int(train_labels[i])
            for j in range(self.feature_len):
                self.conditional[k][j][train_features[i][j]] += 1
        for it in self.category:  # 对条件概率拉普拉斯平滑
            k = int(it)
            for i in range(self.feature_len):
                res = sum(self.conditional[k][i])
                self.conditional[k][i] = (self.conditional[k][i] + 1) / \
                    (res + self.feature_value)
        numpy.save('./npys/bayers_category.npy',
                   numpy.array(list(self.category)))
        numpy.save('./npys/bayers_prior.npy', self.prior)
        numpy.save('./npys/bayers_conditional.npy', self.conditional)

    # 输入特征为　one, 识别为　k 的后验概率
    def get(self, k, one):
        res = 0.00
        for i in range(self.feature_len):
            res += math.log(self.conditional[k][i][int(one[i])])
        return res + math.log(self.prior[k])

    # 输入特征为 one, 从诸多种类中，找出后验概率最大的
    def predict(self, one):
        max_value = self.get(0, one)
        best = -1
        for it in self.category:
            k = int(it)
            cur = self.get(k, one)
            if(cur > max_value):
                max_value = cur
                best = k
        return best

    # test_features 测试数据集 ; test_labels 测试答案
    def test(self, test_features, test_labels):
        test_features.resize(test_features.shape[:2])
        print(test_features.shape)
        correct = 0
        n, m = test_features.shape
        for i in range(n):
            cur = self.predict(test_features[i])
            if(cur == test_labels[i]):
                correct += 1
        print('正确率  ' + str(correct / n))

    #　不训练，直接加载，迅速识别
    def load(self):
        self.category = set(numpy.load('./npys/bayers_category.npy'))
        self.prior = numpy.load('./npys/bayers_prior.npy')
        self.conditional = numpy.load('./npys/bayers_conditional.npy')


if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = samples.Samples().get_npy(
        '../npys/train_mnist.npy', '../npys/test_mnist.npy')
    start = datetime.datetime.now()

    bayers = Naive_Bayers()
    bayers.train(train_features, train_labels)
    print(datetime.datetime.now() - start)
    # bayers.load()
    bayers.test(test_features, test_labels)
