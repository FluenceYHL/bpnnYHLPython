# package
import os
import cv2
import numpy
import random


def make_noise(origin, probablity=0.25):
    noise = origin
    n, m = noise.shape
    for i in range(n):
        for j in range(m):
            if(random.randint(0, 24) % 6 == 0):
                noise[i][j] = -noise[i][j]
    return noise


def paint(input, name='yhl.png'):
    img = numpy.zeros((500, 500), dtype=numpy.uint8) + 255
    for i in range(1, 10):
        cv2.line(img, (i * 50, 0), (i * 50, 500), (54, 54, 54))
        cv2.line(img, (0, i * 50), (500, i * 50), (54, 54, 54))
    # 画矩形
    n, m = input.shape
    for i in range(0, n):
        for j in range(0, m):
            if(input[i][j] > 0):
                cv2.rectangle(img, (j * 50, i * 50),
                              ((j + 1) * 50, (i + 1) * 50), (0, 0, 0), -1)
    cv2.imshow(name, img)


def sigmoid(net):
    return 1 if(net > 0) else -1


class Hopfield():
    def __init__(self, origin):
        self.m, self.n = origin.shape
        self.feature_len = self.m * self.n
        self.weights = numpy.zeros(
            (self.feature_len, self.feature_len), dtype=numpy.float)

    def memorize(self, samples):
        for origin in samples:
            print(origin)
            origin.resize(self.feature_len)
            for i in range(self.feature_len):
                for j in range(i):
                    self.weights[i][j] += origin[i] * origin[j]
        for i in range(self.feature_len):
            for j in range(i):
                self.weights[j][i] = self.weights[i][j]
        self.weights /= self.feature_len

    def repair(self, noise):
        noise.resize(self.feature_len, 1)
        cnt = 0
        while(True):
            changed = False
            paint(noise.reshape(self.m, self.n), 'noise' + str(cnt) + '.png')
            cv2.waitKey()
            cnt += 1
            for i in range(self.feature_len):
                ans = sigmoid(self.weights[i].dot(noise))
                if(ans != noise[i]):
                    changed = True
                noise[i] = ans
            if(changed == False):
                break
        noise.resize(self.m, self.n)
        paint(noise, 'repaired.png')
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    origin = numpy.load('./npys/hopfield/origin/4.npy')
    one = Hopfield(origin)

    fileList = os.listdir('./npys/hopfield/origin')
    print(fileList)
    samples = []
    for it in fileList:
        origin = numpy.load('./npys/hopfield/origin/' + it)
        samples.append(origin)
    samples = numpy.array(samples)
    print(samples.shape)
    one.memorize(samples)

    noise = numpy.load('./npys/hopfield/noise/4.npy')
    one.repair(noise)
    for i in range(10):
        noise = numpy.load('./npys/hopfield/noise/4_' + str(i) + '.npy')
        one.repair(noise)

    cv2.waitKey()
    cv2.destroyAllWindows()

    # fileList = os.listdir('./npys/hopfield/noise')
    # print(fileList)
    # for it in fileList:
    #     noise = numpy.load('./npys/hopfield/noise/' + it)
    #     for i in range(10):
    #         for j in range(10):
    #             if(noise[i][j] == 0):
    #                 noise[i][j] = -1
    #     numpy.save('./npys/hopfield/noise/' + it, noise)
    #     paint(noise)
    #     cv2.waitKey()

    # 直接对样本进行噪声处理
    # for it in samples:
    #     noise = make_noise(it)
    #     one.repair(noise)

    # origin = numpy.load('./npys/hopfield/origin/4.npy')
    # for i in range(10):
    #     noise = make_noise(origin)
    #     paint(noise, 'yhl.png')
    #     cv2.waitKey()
    #     numpy.save('./npys/hopfield/noise/4_' + str(i) + '.npy', noise)

    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # origin = numpy.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    #                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    #                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    #                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    #                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    #                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    #                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    #                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    #                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    # numpy.save('./npys/hopfield/origin/0.npy', origin)
    # origin = numpy.load('./npys/hopfield/origin/0.npy')
    # paint(origin, 'yhl.png')
    # cv2.waitKey()
    # cv2.destroyAllWindows()
