# -*- coding: utf-8 -*-
import Tkinter
from PIL import Image, ImageTk
import os
import threading
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import RPi.GPIO as GPIO

sle_p_num = 0
x_l = 0
y_up = 0
x_r = 0
y_bot = 0
global imglabel
global image

frame_image_width = 796
frame_image_heigth = 600
        
def help_my():
    pass

def select_point(event):
    global sle_p_num
    global x_l
    global y_up
    global x_r
    global y_bot
    global imglabel
    global image
    frame_image_width = 796
    frame_image_heigth = 600
    if(sle_p_num%2==0):
        #print("select top-left",event.x,event.y)
        zuoshangjiao.set(str(event.x)+","+str(event.y))
        event.x = int(event.x*2592/796)
        event.y = int(event.y*2592/796)
        x_l = event.x*100000000
        y_up = event.y*100000000
        f_out = open("/home/pi/Desktop/okok/paras.ini","w")
        f_out.write("x_l="+str(int(x_l)))
        f_out.write("\n")
        f_out.write("y_up="+str(int(y_up)))
        f_out.write("\n")
        f_out.close()
    elif(sle_p_num%2==1):
        #print("select right-right",event.x,event.y)
        youxiajiao.set(str(event.x)+","+str(event.y))
        event.x = int(event.x*1952/600)
        event.y = int(event.y*1952/600)
        x_r = event.x*100000000
        y_bot = event.y*100000000
        f_out = open("/home/pi/Desktop/okok/paras.ini","a+")
        f_out.write("x_r="+str(int(x_r)))
        f_out.write("\n")
        f_out.write("y_bot="+str(int(y_bot)))
        f_out.write("\n")
        f_out.close()
    sle_p_num = sle_p_num + 1
    if(x_l*y_up*x_r*y_bot)!=0:
        #imglabel.destroy()
        #image = cv2.imread("/tmp/nomal.jpg")
        cv2.rectangle(image,(int(x_l/100000000),int(y_up/100000000)),(int(x_r/100000000),int(y_bot/100000000)),(0,255,0),3,4)
        image_re = cv2.resize(image,(frame_image_width,frame_image_heigth))       
        #cv2.rectangle(image,(int(x_l*frame_image_width/259200000000),int(y_up*frame_image_heigth/195200000000)),(int(x_r*frame_image_width/259200000000),int(y_bot*frame_image_heigth/195200000000)),(0,255,0),3,4)
        #image_re = cv2.resize(image,(frame_image_width,frame_image_heigth))
        cv2.imshow("请测量绿色矩形左右两边距离，关闭该窗口后填入右侧相应输入框",image_re)#
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def start_normal():
    os.system("python3 /home/pi/Desktop/okok/src/camera_normal.pyc")

def normal():
    str_3 = Entry_aera.get()
    camera_area_x_width = int(float(str_3)*100000000)
    f_out = open("/home/pi/Desktop/okok/paras.ini","a+")
    f_out.write("camera_area_x_width="+str(camera_area_x_width))
    f_out.write("\n")
    f_out.close()
    threading.Thread(target=start_normal).start()


window = Tkinter.Tk()
window.title('校正')
window.geometry('1024x600')
main_frame = Tkinter.Frame(window, width=1024, height=600, bg="lightblue")
pix_width = 2592
pix_heigth = 1952
frame_image_width = 796
frame_image_heigth = 600
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (pix_width, pix_heigth)#2592, 1952
camera.framerate = 32
camera.hflip = False
camera.vflip = False
#rawCapture = PiRGBArray(camera, size=(pix_width, pix_heigth))#2592, 1952
# allow the camera to warmup
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#    image = frame.array

label_x = 50
width_1 = 10
# button frame
frame_button = Tkinter.Frame(main_frame, width = 200, height = frame_image_heigth, bg="lightblue")
helpButton = Tkinter.Button(frame_button, text='使用说明', command = help_my, font=('Arial', 16,'bold'), width=10, height=2)#, height=2
helpButton.place(x=label_x,y=70)
#calibration
calibrationButton = Tkinter.Button(frame_button, text ="校正", command = normal, font=('Arial', 16,'bold'), width=10, height=2)#, bg='Blues',fg='green'
calibrationButton.place(x=label_x,y=170)
#exit
exitButton = Tkinter.Button(frame_button, text='退出', command=window.quit, font=('Arial', 16,'bold'), width=10, height=2)
exitButton.place(x=label_x,y=270)
## display zuobiao
Label_1 = Tkinter.Label(frame_button, text='左上角坐标' , bg="lightblue", font=('Arial', 12))#, height=2
Label_1.place(x=label_x,y=370)
zuoshangjiao = Tkinter.StringVar(value="0,0")
Label_zuoshangjiao = Tkinter.Label(frame_button, width=width_1, textvariable = zuoshangjiao, show=None, font=('Arial', 12)) 
Label_zuoshangjiao.place(x=label_x,y=390)
Label_2 = Tkinter.Label(frame_button, text='右下角坐标' , bg="lightblue", font=('Arial', 12))#, height=2
Label_2.place(x=label_x,y=420)
youxiajiao = Tkinter.StringVar(value="0,0")
Label_youxiajiao = Tkinter.Label(frame_button, width=width_1, textvariable = youxiajiao, show=None, font=('Arial', 12)) 
Label_youxiajiao.place(x=label_x,y=450)
Label_3 = Tkinter.Label(frame_button, text='检测区域宽度(mm)', bg="lightblue", font=('Arial', 12))#, height=2
Label_3.place(x=label_x,y=470)
str_3 = Tkinter.StringVar(value="30")
Entry_aera = Tkinter.Entry(frame_button, width=width_1, textvariable = str_3, show=None, font=('Arial', 12)) 
Entry_aera.place(x=label_x,y=490)



frame_button.place(x=800,y=0)


frame_image = Tkinter.Frame(main_frame, bg="lightblue", width = frame_image_width, height = frame_image_heigth)
camera.capture("/tmp/normal.jpg")
image = cv2.imread("/tmp/normal.jpg")
image_re = cv2.resize(image,(frame_image_width,frame_image_heigth))
img = Image.fromarray(cv2.cvtColor(image_re,cv2.COLOR_BGR2RGB))
#    break
camera.close()
photo = ImageTk.PhotoImage(img)
imglabel = Tkinter.Label(frame_image, image=photo)
imglabel.pack(side="left")
imglabel.bind("<Button-1>", select_point)
frame_image.place(x=0,y=0)
main_frame.place(x=0,y=0)


Tkinter.mainloop()
