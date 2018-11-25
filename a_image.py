# coding=utf-8

# self
import a_train as zzr
import cluster
import images.fileFilter
# package
import image
import shutil
import os
import glob as gb
import cv2
import numpy
import matplotlib.pyplot as plt
import warnings


# 把测试结果转化为npy类型的文件
def test_output(filename):
    f = open(filename)
    line = f.readline()
    output = []
    while (line != ''):
        output.append(int(line))
        line = f.readline()
    numpy.save("./npys/answers.npy", output)
    return output


# 遍历一个文件夹下的所有图片  打印预测正确率
def traverse(f, mode=1):
    test_output('./images/answers.txt')
    output = numpy.load("./npys/answers.npy")
    print(output)
    fileList = images.fileFilter.filter(f)
    count = 0
    right = 0
    for path in fileList:
        if(mode == 2):
            imgs = cluster.getImage(path)
            result = ''
            for img in imgs:
                out = nn.judge(nn.convert_to_binary(img[1]))
                print(out)
                result += str(out)
            print(result)
        else:
            img = cv2.imread(path, 0)
            guess = nn.judge(nn.convert_to_binary(img))
            print(path)
            if(guess == output[count]):
                right = right + 1
            count = count + 1

    print("正确率  :  " + str(right / count))


# 使用训练好的模型
if __name__ == '__main__':
    input_node = 784
    hidden_node = 100
    output_node = 10
    learning_rate = 0.1
    nn = zzr.Neural_Networks(input_node, hidden_node,
                             output_node, learning_rate)
    nn.whid = numpy.load("./npys/whid.npy")
    nn.wout = numpy.load("./npys/wout.npy")
    warnings.filterwarnings("ignore")
    # traverse('./images/origin', 2)
    traverse('./images/handled', 1)


# # coding=utf-8

# # self
# import a_train as zzr
# import cluster
# # package
# import image
# import shutil
# import os
# import glob as gb
# import cv2
# import numpy
# import matplotlib.pyplot as plt
# import warnings

# # 把测试结果转化为npy类型的文件


# def test_output(filename):
#     f = open(filename)
#     line = f.readline()
#     output = []
#     while (line != ''):
#         output.append(int(line))
#         line = f.readline()
#     numpy.save("test_output.npy", output)
#     return output


# # 遍历一个文件夹下的所有图片  打印预测正确率
# def traverse(f):
#     # test_output("count.txt")
#     output = numpy.load("test_output.npy")
#     fs = os.listdir(f)
#     count = 0
#     right = 0
#     for f1 in fs:
#         path = os.path.join(f, f1)
#         if not os.path.isdir(path):
#             print(path)
#             imgs = cluster.getImage(path)
#             result = ''
#             for img in imgs:
#                 out = nn.judge(nn.convert_to_binary(img[1]))
#                 print(out)
#                 result += str(out)
#             print(result)
#             # myimage = cv2.imread(path, cv2.IMREAD_COLOR)
#             # myimage = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
#             # myimage = nn.convert_to_binary(myimage)
#             # my_1 = numpy.zeros((1, 784))
#             # my_1 = numpy.full(my_1.shape, 1)
#             # myimage = my_1 - myimage
#             # out = nn.judge(myimage)
#             # if (output[count] == int(out)):
#             #     right += 1
#             # print('I guess it is ' + str(out))
#             # count += 1

#         else:
#             print('文件夹：%s' % path)
#             traverse(path)
#     print("Right:" + str(right) + "   All have:" + str(count))


# # 使用训练好的模型
# if __name__ == '__main__':
#     input_node = 784
#     hidden_node = 100
#     output_node = 10
#     learning_rate = 0.1  # ,96%,3times
#     nn = zzr.Neural_Networks(input_node, hidden_node,
#                              output_node, learning_rate)
#     nn.whid = numpy.load("whid.npy")
#     nn.wout = numpy.load("wout.npy")
#     warnings.filterwarnings("ignore")
#     traverse('./images')
