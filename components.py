# -*- coding: utf-8-*-

# self
# package
import tkinter
import sys
import pyttsx3


class noticeBoard():
    def __init__(self, w, h, name):
        self.dialog = tkinter.Tk()
        self.w = w
        self.h = h
        self.dialog = tkinter.Tk()
        self.dialog.title(name)
        self.dialog.geometry(str(h) + 'x' + str(w))

    def display(self, value):
        result = tkinter.Label(self.dialog, fg='Green', bg='#DDDDDD',
                               font=('Arial', 50), text=str(value), anchor='center', width=480, height=560).pack()
        self.dialog.mainloop()


class speaker():
    # https://blog.csdn.net/ctwy291314/article/details/81098998
    # https://blog.csdn.net/qq_24822271/article/details/82836339
    # https://github.com/mozillazg/python-pinyin
    # https://blog.csdn.net/weixin_39012047/article/details/82012306?utm_source=blogxgwz3
    # https://blog.csdn.net/m0_37713821/article/details/80488191
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', 'zh')
        # self.volume = self.engine.getProperty('volume')
        # self.engine.setProperty('volume', self.volume - 0.25)

    def speak(self, num):
        self.engine.say(num)
        self.engine.runAndWait()


if __name__ == '__main__':
    one = speaker()
    one.speak('我愿逆流而上')
