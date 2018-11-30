# package
import cv2
import numpy
# self


class Samples():
    def binary_value(self, imgs):
        new_imgs = []
        for it in imgs:
            ret, img = cv2.threshold(it, 127, 1, cv2.THRESH_BINARY)
            new_imgs.append(img)
        return numpy.array(new_imgs, dtype=numpy.uint8)

    def get_npy(self, train_file, test_file):
        train_data = numpy.load(train_file)
        train_traits = self.binary_value(train_data[:, 1:])
        train_labels = train_data[:, 0]

        test_data = numpy.load('../npys/test_mnist.npy')
        test_traits = self.binary_value(test_data[:, 1:])
        test_labels = test_data[:, 0]
        return train_traits, train_labels, test_traits, test_labels


if __name__ == '__main__':
    train_traits, train_labels, test_traits, test_labels = Samples().get_npy(
        '../npys/train_mnist.npy', '../npys/test_mnist.npy')

    print(train_traits.shape)
    # img = train_traits[0].reshape(28, 28)
    # cv2.imshow('YHL.png', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
