# package
import numpy
import sys
sys.path.append("..")
# self
import pca


if __name__ == '__main__':
    train_data = numpy.load('../npys/train_mnist.npy')
    one = pca.YHL_pca()
    k, mean_value, primary_components, features = one.pca(train_data, 0.95)
    m, n = features.shape

    test_data = numpy.load('../npys/test_mnist.npy')
    print(test_data.shape)
    m2, n2 = test_data.shape
    for l in test_data:
        result = []
        index = 0
        for r in train_data:
            cnt = 0
            for j in range(n):
                if(l[j] != r[j]):
                    cnt = cnt + 1
            result.append([cnt, index])
            index = index + 1
        result = result.sort()
        print(str(result[0][0]))
        break
