# -*- coding: utf-8 -*-
import tkinter
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import math

global photo

def open_image():
    global photo
    file_path = tkinter.filedialog.askopenfilename()
    image = cv2.imread(file_path)
    cv2.imwrite('tmp.jpg',image)
    image_re = cv2.resize(image,(frame_image_width,frame_image_heigth))
    img = Image.fromarray(cv2.cvtColor(image_re,cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(img)
    imglabel.config(image=photo)
    imglabel.image=photo

    
def take_photo():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    flag = cap.isOpened()
    index = 1
    while (flag):
        ret, frame = cap.read()
        cv2.imshow("Capture_Paizhao", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):  # 按下s键，进入下面的保存图片操作
            cv2.imwrite("E:/PyCharm Workspaces/" + str(index) + ".jpg", frame)
            print("save" + str(index) + ".jpg successfuly!")
            print("-------------------------")
            index += 1
        elif k == ord('q'):  # 按下q键，程序退出
            break
    cap.release() # 释放摄像头
    cv2.destroyAllWindows()# 释放并销毁窗口


def process():
    global photo
    img = cv2.imdecode(np.fromfile('tmp.jpg', dtype=np.uint8), 1)
    img_ori = img.copy()

    #图片先转成灰度的
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #转换二值图
    ret,binary=cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    #反色
    binary=cv2.bitwise_not(binary)
    #轮廓识别
    contours, hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


    #要计算的参数：总数，重量，千粒重，总面积，面积均值，面积标准差，周长均值，周长标准差，直径均值，直径标准差
    num_seed = 0
    weight_seed = 0
    weight_seed_k = 0
    area_all = 0
    area_mean = 0
    area_sd = 0
    length_mean = 0
    length_sd = 0
    r_mean = 0
    r_sd = 0
    
    contours_seed = []
    area_list = []
    length_list = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i],True)
        #设定阈值
        if area < 25 or area > 70:
            cv2.drawContours(binary,[contours[i]],0,0,-1)
            continue
        area_list.append(area)
        length_list.append(length)
        contours_seed.append(contours[i])
    area_np = np.array(area_list)
    length_np = np.array(length_list)
    num_seed = len(area_list)
    weight_seed = 0  # 面积重量转化公式
    weight_seed_k = 0
    area_all = np.sum(area_np)
    area_mean = area_all/num_seed
    area_sd = np.std(area_np, ddof=1)
    length_mean = np.mean(area_np)
    length_sd = np.std(length_np, ddof=1)
    r_np = 2*(area_np/math.pi)**0.5
    r_mean = np.mean(r_np)
    r_sd = np.std(r_np, ddof=1)
    
    str_resutl = "总      数:  {num_seed}\n\
重      量:  {weight_seed}\n\
千  粒  重:  {weight_seed_k}\n\
总  面  积:  {area_all}\n\
面积均值:    {area_mean}\n\
面积标准差:  {area_sd}\n\
周长均值:    {length_mean}\n\
周长标准差:  {length_sd}\n\
直径均值:    {r_mean}\n\
直径标准差:  {r_sd}\n".format(num_seed=str(num_seed), weight_seed=str(weight_seed),weight_seed_k=str(np.round(weight_seed_k,6)),area_all=str(np.round(area_all,6)),area_mean=str(np.round(area_mean,6)),area_sd=str(np.round(area_sd,6)),length_mean=str(np.round(length_mean,6)),length_sd=str(np.round(length_sd,6)),r_mean=str(np.round(r_mean,6)),r_sd=str(np.round(r_sd,6)))
    text_info.delete('1.0','end')
    text_info.insert(0.0, str_resutl)
    cv2.drawContours(img_ori, contours_seed, -1, (0, 255, 0), 1)
    cv2.imwrite('tmp_res.jpg',img_ori)
    time.sleep(0.1)
    image = cv2.imread('tmp_res.jpg')
    image_re = cv2.resize(image,(frame_image_width,frame_image_heigth))
    img = Image.fromarray(cv2.cvtColor(image_re,cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(img)
    imglabel.config(image=photo)
    imglabel.image=photo



frame_image_width = 796
frame_image_heigth = 600

window = tkinter.Tk()
window.title('千粒重识别系统')
window.geometry('1024x600')
main_frame = tkinter.Frame(window, width=1024, height=600, bg="lightblue")


# 图像展示
frame_image = tkinter.Frame(main_frame, bg="lightblue", width = frame_image_width, height = frame_image_heigth)

image = cv2.imread("2.jpg")
image_re = cv2.resize(image,(frame_image_width,frame_image_heigth))
img = Image.fromarray(cv2.cvtColor(image_re,cv2.COLOR_BGR2RGB))

photo = ImageTk.PhotoImage(img)
imglabel = tkinter.Label(frame_image, image=photo)
imglabel.pack(side="left")
frame_image.place(x=0,y=0)


# 功能区
frame_fuction = tkinter.Frame(main_frame, width = 200, height = frame_image_heigth, bg="lightblue")

# 功能区-操作区

label_x = 820
frame_fuction_button = tkinter.Frame(main_frame, width = 200, height = int(frame_image_heigth/4), bg="lightblue")
openButton = tkinter.Button(frame_fuction_button, text='打开图像', command = open_image, font=('Arial', 12,'bold'), width=10, height=1)
openButton.place(x=40,y=0)
takePhotoButton = tkinter.Button(frame_fuction_button, text='拍    照', command = take_photo, font=('Arial', 12,'bold'), width=10, height=1)
takePhotoButton.place(x=40,y=40)
processButton = tkinter.Button(frame_fuction_button, text='处    理', command = process, font=('Arial', 12,'bold'), width=10, height=1)
processButton.place(x=40,y=80)
frame_fuction_button.place(x=820,y=50)


# 功能区-展示区
frame_fuction_display = tkinter.Frame(main_frame, width = 200, height = int(frame_image_heigth*4/5), bg="lightblue")
text_info = tkinter.Text(frame_fuction_display, width=100, height = int(frame_image_heigth*4/5))

text_info.place(x=0,y=0)
frame_fuction_display.place(x=810,y=200)



frame_fuction.place(x=820,y=0)

main_frame.place(x=0,y=0)



tkinter.mainloop()
