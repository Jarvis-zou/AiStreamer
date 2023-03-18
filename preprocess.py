#! /usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
import cv2
import tkinter


def videoJDK(cx, cy, vw, vh, url_1, url_2):
    videoCapture = cv2.VideoCapture(url_1)  # 从文件读取视频
    # 判断视频是否打开
    if (videoCapture.isOpened()):
        print('Open')
    else:
        print('Fail to open!')

    fps = videoCapture.get(cv2.CAP_PROP_FPS)  # 获取原视频的帧率

    size = (int(vw), int(vh))  # 自定义需要截取的画面的大小
    videoWriter = cv2.VideoWriter(url_2, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (360,600))
    success, frame = videoCapture.read()  # 读取第一帧
    print(frame.shape)
    ints = int(fps)
    while success:
        # print(frame)
        print(cy + vh)
        frame = frame[cy:cy + vh,cx:cx + vw]  # 截取画面

        cv2.imshow("Oto Video", frame) #显示
        # cv2.waitKey(int(1000/ints)) #延迟

        videoWriter.write(frame)  # 写视频帧
        success, frame = videoCapture.read()  # 获取下一帧
    videoCapture.release()


cx = 0  # 起点x
cy = 40  # 起点y
vw = 360  # 宽
vh = 600 # 高

url_1 = "videos/fengge/not_talking_10.mp4"  # 源视频
url_2 = "videos/fengge//not_talking_source/not_talking_10.avi"  # 转换后视频

videoJDK(cx, cy, vw, vh, url_1, url_2)