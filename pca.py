# self
import images.fileFilter
# package
import numpy
import cv2

# 取出所有数字为　num 的图片
def getDigit(num):
    answers = numpy.load('./npys/answers.npy')
    index = numpy.argwhere(answers == num)
    fileList = images.fileFilter.filter('./images/handled')
    featureOfNum = []
    for it in index:
        img = cv2.imread(fileList[int(it)], 0)
        featureOfNum.append(img.reshape(784))
    return numpy.array(featureOfNum)

# 取出目录下所有图片
def allDigits(fileDir):
    answers = numpy.load('./npys/answers.npy')
    fileList = images.fileFilter.filter(fileDir)
    featureOfNum = []
    for it in fileList:
        img = cv2.imread(it, 0)
        featureOfNum.append(img.reshape(784))
    return numpy.array(featureOfNum)


class YHL_pca():
    def __init__(self, percentage=0.998):
        self.percentage = percentage

    # 根据比例, 提取出前　K 个特征向量, 本函数返回　k
    def __getK(self, feature_values):
        res = 0
        index = 0
        features_sorted = numpy.sort(-feature_values)  # 从大到小排序
        sum_value = sum(feature_values)
        for it in feature_values:
            index += 1
            res += it
            if(res >= sum_value * self.percentage):
                return index

    def pca(self, features, percentage=0.998, k=0):
        self.percentage = percentage               # 更新本次　PCA 的比率
        mean_value = numpy.mean(features, axis=0)
        features = features - mean_value		   # 减去均值, 更方便计算协方差
        cov_matrix = numpy.cov(features, rowvar=0)  # 求协方差矩阵
        feature_values, feature_vectors = numpy.linalg.eig(
            numpy.mat(cov_matrix))  # 求特征值，特征向量
        if(k == 0):
            k = self.__getK(feature_values)        # 如果没有指定　k, 就按照特征比率提取
        indexs = numpy.argsort(-feature_values)    # 对特征值从大到小排序，返回索引
        k_indexs = indexs[:k]					   # 提取　k 个特征值
        primary_components = feature_vectors[:, k_indexs]  # 找出对应的　k 个特征向量
        print(primary_components.shape)
        print(features.shape)
        return k, mean_value, primary_components, numpy.dot(features, primary_components)


if __name__ == '__main__':
    one = YHL_pca()
    k, mean_value, primary_components, features = one.pca(
        allDigits('./images/handled'), 0.998, 40)
    print(features.shape)
    print(mean_value.shape)
