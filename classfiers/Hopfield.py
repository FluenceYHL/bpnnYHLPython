# package
import cv2
import numpy
import random


def make_noise(origin, probablity=0.25):
    noise = origin
    n, m = noise.shape
    for i in range(n):
        for j in range(m):
            if(random.randint(0, 24) % 6 == 0):
                noise[i][j] = 1 - noise[i][j]
    return noise


def paint(input, name):
    img = numpy.zeros((500, 500), dtype=numpy.uint8) + 255
    for i in range(1, 10):
        cv2.line(img, (i * 50, 0), (i * 50, 500), (54, 54, 54))
        cv2.line(img, (0, i * 50), (500, i * 50), (54, 54, 54))
    # 画矩形
    n, m = input.shape
    for i in range(0, n):
        for j in range(0, m):
            if(input[i][j] == 1):
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
            (self.feature_len, self.feature_len), dtype=numpy.uint8)

    def memorize(self, origin):
        origin.resize(self.feature_len)
        for i in range(self.feature_len):
            for j in range(i):
                self.weights[i][j] += origin[i] * origin[j]
                self.weights[j][i] = self.weights[i][j]

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
        cv2.destroyAllWindows()


if __name__ == '__main__':
    origin = numpy.load('./npys/hopfield/origin/4.npy')
    one = Hopfield(origin)
    one.memorize(origin)

    noise = numpy.load('./npys/hopfield/noise/4.npy')
    one.repair(noise)
    for i in range(10):
        noise = numpy.load('./npys/hopfield/noise/4_' + str(i) + '.npy')
        one.repair(noise)

    cv2.waitKey()
    cv2.destroyAllWindows()

    # origin = numpy.load('./npys/hopfield/origin/4.npy')
    # for i in range(10):
    #     noise = make_noise(origin)
    #     paint(noise, 'yhl.png')
    #     cv2.waitKey()
    #     numpy.save('./npys/hopfield/noise/4_' + str(i) + '.npy', noise)

    # cv2.waitKey()
    # cv2.destroyAllWindows()
