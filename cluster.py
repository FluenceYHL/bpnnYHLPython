# self
import merge
# package
import os
import cv2
import image
import math
import numpy
import images.fileFilter
from operator import itemgetter

# 求两点之间的欧氏距离


def distance(l, r):
    return math.sqrt((l[0] - r[0])**2 + (l[1] - r[1])**2)


def getImages(myImage):
    myImage = cv2.resize(myImage, (70, 70))
    myImage = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)

    points = []  # 存储黑色像素点
    if(myImage.ndim == 2):  # 如果是二值化的图像
        width = myImage.shape[1]
        height = myImage.shape[0]
        for i in range(height):
            for j in range(width):
                if(myImage[i, j] == 0):  # 如果黑色, 加入
                    points.append([i, j])

        # 建立一个最大标号为　len(points) 的并查集
        one = merge.mergeSet(len(points))
        i = 0
        for l in points:
            j = 0
            for r in points:
                if(distance(l, r) < 2):
                    one.merge(i, j)     # < threshold 就建立连接　属于同一连通分量的阈值
                j = j + 1
            i = i + 1
        # 在这里可以过滤点数过小的点, 噪音, 处理负号, 以及断层修复的问题
        clusters = one.cluster()
        for k in list(clusters.keys()):
            if(len(clusters[k]) < 10):  # threshold2 过滤点数过小的点集合, 噪音
                clusters.pop(k)
                continue
        print('聚类个数　　:  ' + str(len(clusters)))

        # cv2.imshow('YHL.png', myImage)
        # cv2.waitKey()
        imgs = []
        #　对每一个连通分量
        for it in clusters.values():
            meanX = 0
            meanY = 0
            floor = -1
            ceil = 1e6
            for p in it:
                meanX += points[p][1]               # x 均值
                meanY += points[p][0]               # y 均值
                floor = max(floor, points[p][0])  # 图像下限
                ceil = min(ceil, points[p][0])      # 图像上限
            meanX /= len(it)
            meanY /= len(it)

            radius = (floor - ceil) / 2  # 　竖向半径
            assert(radius > 0)

            img = numpy.zeros(
                (int(radius * 2 + 12), int(radius * 2 + 12)), numpy.uint8)
            img.fill(255)  # 生成一张白色图片

            h = img.shape[1]
            w = img.shape[0]
            for p in it:
                x = int(0.5 * h + (points[p][1] - meanX))  # 以原图片中心构建新图像
                y = int(0.5 * w + (points[p][0] - meanY))
                img[y, x] = 0
                if(x < h - 1):  # 　一定的修补
                    img[y, x + 1] = 0
                if(y < w - 1):          # 一定的修补
                    img[y + 1, x] = 0

            img = cv2.resize(img, (28, 28))
            imgs.append([meanY, meanX, img])   # 存储这整张图片所有的数字

        imgs = sorted(imgs, key=itemgetter(1, 0))

        # cnt = 0
        # # global myCount
        # for pair in imgs:
        #     cnt = cnt + 1
        #     print('均值　 :  ' + str(pair[0]))
        #     # myCount = myCount + 1
        #     # print('myCount  =  ' + str(myCount))
        #     cv2.imshow(str(cnt) + '.png', pair[2])
        #     # cv2.imwrite('./images/handled/' + str(myCount) + '.png', pair[1])
        #     cv2.waitKey()
        return imgs


def getImage(path):
    print(path)
    myImage = cv2.imread(path, cv2.IMREAD_COLOR)
    return getImages(myImage)


if __name__ == '__main__':
    fileDir = './images/origin'
    fileList = images.fileFilter.filter(fileDir)
    for it in fileList:
        getImage(it)
    cv2.destroyAllWindows()
