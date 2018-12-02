# package
import cv2
import numpy
import random


def sigmoid(net):
    return 1 if(net > 0) else -1


def make_noise(origin):
    noise = origin
    n, m = noise.shape
    for i in range(n):
        for j in range(m):
            if(random.randint(0, 24) % 6 == 0):
                noise[i][j] = 1 - noise[i][j]
    paint(noise)
    return noise


def paint(input):
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
    cv2.imshow('yhl.png', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    origin = numpy.load('./npys/hopfield/origin/4.npy')
    paint(origin)

    noise = numpy.load('./npys/hopfield/noise/4.npy')
    paint(noise)

# img = numpy.array([[0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
#                [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
#                [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
#                [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
#                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
#                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
#                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
#                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])
# numpy.save('./npys/hopfield/4.npy', img)
