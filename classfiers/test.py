# package
import cv2
import math
import time
import numpy
# self
import samples


class Naive_Bayers():
    def __init__(self):
        self.labels = 10
        self.feature_len = 784
        self.feature_value = 2
        self.prior = numpy.zeros(self.labels, dtype=float)
        self.conditional = numpy.zeros(
            (self.labels, self.feature_len, self.feature_value))
        self.category = set([])

    def set_argment(self, labels, feature_len, feature_value):
        self.labels = labels
        self.feature_len = feature_len
        self.feature_value = feature_value

    def train(self, train_features, train_labels):
        length = len(train_labels)
        train_features.reshape(
            train_features.shape[0], train_features.shape[1])
        self.category = set(train_labels)
        value = set(train_features[0].reshape(784))
        self.set_argment(len(self.category),
                         train_features.shape[1], len(value))

        self.prior = numpy.zeros(self.labels, dtype=float)
        for it in train_labels:
            self.prior[int(it)] += 1
        for it in self.category:
            self.prior[int(it)] = (self.prior[int(it)] + 1) / \
                (length + self.labels)

        self.conditional = numpy.zeros(
            (self.labels, self.feature_len, self.feature_value))
        for i in range(length):
            k = int(train_labels[i])
            for j in range(self.feature_len):
                self.conditional[k][j][train_features[i][j]] += 1
        for it in self.category:
            k = int(it)
            for i in range(self.feature_len):
                res = sum(self.conditional[k][i])
                self.conditional[k][i] = (self.conditional[k][i] + 1) / \
                    (res + self.feature_value)
        numpy.save('./npys/bayers_prior.npy', self.prior)
        numpy.save('./npys/bayers_conditional.npy', self.conditional)

        # 记得保存　conditional 等信息

    def get(self, k, one):
        res = 0.00
        for i in range(self.feature_len):
            res += math.log(self.conditional[k][i][int(one[i])])
        return res + math.log(self.prior[k])
        # res = 1.00
        # for i in range(self.feature_len):
        #     res *= self.conditional[k][i][int(one[i])]
        # return self.prior[k] * res

    def predict(self, one):
        max_value = self.get(0, one)
        best = -1
        # print(self.category)
        for it in self.category:
            k = int(it)
            cur = self.get(k, one)
            # print(str(cur) + '\t' + str(max_value))
            # time.sleep(1)
            if(cur > max_value):
                max_value = cur
                best = k
        return best

    def mni_test(self, test_features, test_labels):
        # numpy.reshape(test_features, (10000, 784))  # python 智障吧
        test_features.resize(10000, 784)
        print(test_features.shape)
        print(test_labels.shape)
        correct = 0
        n, m = test_features.shape
        self.category = set(test_labels)
        for i in range(n):
            cur = self.predict(test_features[i])
            # print(cur)
            # print(str(i) + '\tcur = ' + str(cur) + '\t' + str(test_labels[i]))
            if(cur == test_labels[i]):
                correct += 1
        print('正确率  ' + str(correct / n))

    def mni_load(self):
        self.prior = numpy.load('./npys/bayers_prior.npy')
        self.conditional = numpy.load('./npys/bayers_conditional.npy')


if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = samples.Samples().get_npy(
        '../npys/train_mnist.npy', '../npys/test_mnist.npy')

    bayers = Naive_Bayers()
    # bayers.train(train_features, train_labels)
    bayers.mni_load()
    bayers.mni_test(test_features, test_labels)

    cv2.waitKey()
    cv2.destroyAllWindows()
