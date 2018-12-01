# self
import components
# package
import image
import cv2
import numpy as np
import a_train as zzr
import cluster
import warnings


class doodleBoard():
    def __init__(self):
        self.quit = False
        self.Drawing = True
        #　加载神经网络
        self.bpnn = zzr.Neural_Networks()
        self.bpnn.whid = np.load("./npys/whid.npy")
        self.bpnn.wout = np.load("./npys/wout.npy")
        # 加载　cv 涂鸦板
        self.Img = np.zeros((500, 700, 3), np.uint8)
        self.Img = np.full(self.Img.shape, 255, np.uint8)
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.__paint)

    def __predict(self, threshold=0.5):
        imgs = cluster.getImages(self.Img)
        guess = ''
        for pair in imgs:
            out = self.bpnn.judge(
                self.bpnn.convert_to_binary(pair[2]))
            res = []
            # cv2.imshow('yhl2.png', pair[2])
            if(np.max(out) < threshold):
                for i in range(3):
                    pair[2] = np.rot90(pair[2])
                    # cv2.imshow('yhl.png', pair[2])
                    # cv2.waitKey()
                    out = self.bpnn.judge(
                        self.bpnn.convert_to_binary(pair[2]))
                    index = np.argmax(out)
                    res.append([1 - out[index], index])
                res.sort()
                print(res)
                guess += str(res[0][1])
            else:
                guess += str(np.argmax(out))
        print('我猜是　　:  ' + guess)
        components.speaker().speak(int(guess))
        # components.noticeBoard(480, 560, '基尔兽').display(guess)

    def __paint(self, Event, X, Y, Flags, mmm):
        Color = (0, 0, 0)
        global IX, IY
        if Event == cv2.EVENT_LBUTTONDOWN:
            self.Drawing = True
            IX, IY = X, Y
        elif Event == cv2.EVENT_MOUSEMOVE and Flags == cv2.EVENT_FLAG_LBUTTON:
            if self.Drawing:
                cv2.line(self.Img, (IX, IY), (X, Y), Color, 30)
                IX, IY = X, Y
        elif Event == cv2.EVENT_LBUTTONUP:
            self.__predict()
            self.Drawing = False

    def run(self):
        while (not self.quit):
            self.Img = np.zeros((500, 700, 3), np.uint8)
            self.Img = np.full(self.Img.shape, 255, np.uint8)
            while (1):
                cv2.imshow('Image', self.Img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('n'):
                    self.Drawing = False
                    break
                elif k == ord('q'):
                    self.quit = True
                    break


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    one = doodleBoard()
    one.run()
    cv2.destroyAllWindows()
