# package
import cv2
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

    def set_argment(self, labels, feature_len, feature_value):
        self.labels = labels
        self.feature_len = feature_len
        self.feature_value = feature_value

    def train(self, train_features, train_labels):
        length = len(train_labels)
        train_features.reshape(
            train_features.shape[0], train_features.shape[1])
        category = set(train_labels)
        value = set(train_features[0].reshape(784))
        self.set_argment(len(category), train_features.shape[1], len(value))

        self.prior = numpy.zeros(self.labels, dtype=float)
        for it in train_labels:
            self.prior[int(it)] += 1
        for it in category:
            self.prior[int(it)] = (self.prior[int(it)] + 1) / \
                (length + self.labels)

        self.conditional = numpy.zeros(
            (self.labels, self.feature_len, self.feature_value))
        for i in range(length):
            k = int(train_labels[i])
            for j in range(self.feature_len):
                self.conditional[k][j][train_features[i][j]] += 1
        for it in category:
            k = int(it)
            for i in range(self.feature_len):
                res = sum(self.conditional[k][i])
                self.conditional[k][i] = (self.conditional[k][i] + 1) / \
                    (res + self.feature_value)
        print(self.conditional[0])

    def mni_test(self, test_features, test_labels):
        print(mni_test.shape)
        print(test_labels.shape)


if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = samples.Samples().get_npy(
        '../npys/train_mnist.npy', '../npys/test_mnist.npy')

    bayers = Naive_Bayers()
    bayers.train(train_features, train_labels)
    bayers.mni_test(test_features, test_labels)

    cv2.waitKey()
    cv2.destroyAllWindows()
