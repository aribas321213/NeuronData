import tkinter
import cv2
import time
import window_main
import os
import shutil
import numpy as np
import json
import config1
from tkinter import *
from PIL import Image, ImageTk

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.attributes('-fullscreen', True)
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.btn_new_man=tkinter.Button(window, text="Информация по человеку", width=50, command=self.info)
        self.btn_upd=tkinter.Button(window, text="Обновить окно", width=50, command=self.upd)
        self.btn_exit=tkinter.Button(window, text="Закрыть окно", width=50, command=self.exit_pr)
        self.btn_new_man.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_upd.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_exit.pack(anchor=tkinter.CENTER, expand=True)
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def info(self):
        self.window.destroy()
        InfoBlock(tkinter.Tk(), "Info", 2)

    def exit_pr(self):
        exit(0)

    def upd(self):
        self.window.destroy()

    def delet(self):
        label = window_main.num
        window_main.write_new(label[0])
        self.window.destroy()

    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            predicted_img = window_main.predict(frame)
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(predicted_img))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class InfoBlock:
    def __init__(self, window, window_title, student):
        self.screen = window
        self.screen.attributes('-fullscreen', True)
        self.screen.title(window_title)
        self.load = Image.open('Z:\\Neuronka\\Training\\s'+str(student)+'\\10cam.jpg')
        self.render = ImageTk.PhotoImage(self.load)
        self.img = Label(window, image=self.render)
        self.img.image = self.render
        self.img.pack(anchor='w', expand=True)
        # self.btn_new_man=tkinter.Button(window, text="Информация по человеку", width=50, command=self.info)
        self.btn_exit=tkinter.Button(window, text="Закрыть окно", width=50, command=self.upd)
        # self.btn_new_man.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_exit.pack(anchor='w', expand=True)
        self.screen.mainloop()
    
    def upd(self):
        self.screen.destroy()


# будем выполнять цикл, пока не нажмем на кнопку для выхода из программы
while True:
    # открытие окна
    App(tkinter.Tk(), "Tkinter and OpenCV")

