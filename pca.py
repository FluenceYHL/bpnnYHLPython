# self
import images.fileFilter
# package
import numpy
import cv2


def getDigit(num):
    answers = numpy.load('./npys/answers.npy')
    index = numpy.argwhere(answers == num)
    fileList = images.fileFilter.filter('./images/handled')
    featureOfNum = []
    for it in index:
        img = cv2.imread(fileList[int(it)], 0)
        featureOfNum.append(img.reshape(784))
    return numpy.array(featureOfNum)


def getK(feature_values, percentage=0.9):
    res = 0
    index = 0
    features_sorted = numpy.sort(-feature_values)
    sum_value = sum(feature_values)
    for it in feature_values:
        index += 1
        res += it
        if(res >= sum_value * percentage):
            return index


def pca(features, percentage=0.98):
    mean_value = numpy.mean(features, axis=0)
    features = features - mean_value
    cov_matrix = numpy.cov(features, rowvar=0)
    feature_values, feature_vectors = numpy.linalg.eig(numpy.mat(cov_matrix))
    k = getK(feature_values, percentage)
    indexs = numpy.argsort(-feature_values)
    k_indexs = indexs[:k]
    primary_components = feature_vectors[:, k_indexs]
    return k, mean_value, numpy.dot(features, primary_components)


class YHL_pca():
    def __init__(self, percentage=0.998):
        self.percentage = percentage

    def __getK(self, feature_values):
        res = 0
        index = 0
        features_sorted = numpy.sort(-feature_values)
        sum_value = sum(feature_values)
        for it in feature_values:
            index += 1
            res += it
            if(res >= sum_value * self.percentage):
                return index

    def pca(self, features, percentage=0.998, k=0):
        self.percentage = percentage
        mean_value = numpy.mean(features, axis=0)
        features = features - mean_value
        cov_matrix = numpy.cov(features, rowvar=0)
        feature_values, feature_vectors = numpy.linalg.eig(
            numpy.mat(cov_matrix))
        if(k == 0):
            k = self.__getK(feature_values)
        indexs = numpy.argsort(-feature_values)
        k_indexs = indexs[:k]
        primary_components = feature_vectors[:, k_indexs]
        return k, mean_value, numpy.dot(features, primary_components)


if __name__ == '__main__':
    one = YHL_pca()
    print(one.pca(getDigit(1), 0.998, 40))
