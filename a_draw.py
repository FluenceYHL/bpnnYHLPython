import image
import cv2
import numpy as np
import a_train as zzr
import cluster


def drawCircle(Event, X, Y, Flags, mmm):
    Color = (0, 0, 0)  # black
    global IX, IY, Drawing, Mode
    if Event == cv2.EVENT_LBUTTONDOWN:
        Drawing = True
        IX, IY = X, Y
    elif Event == cv2.EVENT_MOUSEMOVE and Flags == cv2.EVENT_FLAG_LBUTTON:
        if Drawing:
            cv2.line(Img, (IX, IY), (X, Y), Color, 30)
            IX, IY = X, Y
    elif Event == cv2.EVENT_LBUTTONUP:
        Drawing = False


quit = False

while (not quit):
    Img = np.zeros((500, 700, 3), np.uint8)
    Img = np.full(Img.shape, 255, np.uint8)
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', drawCircle)

    nn = zzr.Neural_Networks()
    nn.whid = np.load("./npys/whid.npy")
    nn.wout = np.load("./npys/wout.npy")

    while (1):
        cv2.imshow('Image', Img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('n'):
            # Img = cv2.resize(Img, (28, 28), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('./zzr/try.png', Img)

            imgs = cluster.getImage('./zzr/try.png')
            guess = ''
            for pair in imgs:
                out = nn.judge(nn.convert_to_binary(pair[1]))
                guess += str(out)
            print('我猜是　　:  ' + guess)
            break
        elif k == ord('q'):
            quit = True
            break
cv2.destroyAllWindows()
