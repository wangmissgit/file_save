# -*- coding: utf-8 -*-
import tkinter
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import math
import sys
import random
import scipy.sparse as sparse
import scipy.sparse.linalg as splina


global photo

def open_image():
    global photo
    file_path = tkinter.filedialog.askopenfilename()
    print(file_path)
    image = cv2.imread(file_path)
    print(image.shape)
    image_shape = image.shape
    if(image_shape[0]/image_shape[1]>=image_shape[1]/image_shape[0]):
    	resize_y = frame_image_heigth
    	resize_x = int(image_shape[1]*resize_y/image_shape[0])
    else:
    	resize_x = frame_image_width
    	resize_y = int(image_shape[0]*resize_x/image_shape[1])
    print((resize_x,resize_y))
    image_re = cv2.resize(image,(resize_x,resize_y))
    cv2.imwrite('tmp.jpg',image_re)
    #image_re = cv2.resize(image,(frame_image_width,frame_image_heigth))
    img = Image.fromarray(cv2.cvtColor(image_re,cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(img)
    imglabel.config(image=photo)
    imglabel.image=photo

    
def retinex_SSR():
    global photo
    PATH_IMG_FILE = 'tmp.jpg'
    img = cv2.imread(PATH_IMG_FILE)
    reti = retinex()
    img_ = reti.repair(img, 200, 0)
    cv2.imshow('retinex_SSR result', img_)
    cv2.waitKey(0)


def retinex_MSR():
    PATH_IMG_FILE = 'tmp.jpg'
    img = cv2.imread(PATH_IMG_FILE)
    reti = retinex()
    img_ = reti.repair(img, (65, 180, 200), 1)
    cv2.imshow('retinex_MSR result', img_)
    cv2.waitKey(0)

    
def retinex_MSRCR():
    PATH_IMG_FILE = 'tmp.jpg'
    img = cv2.imread(PATH_IMG_FILE)
    reti = retinex()
    img_ = reti.repair(img, (65, 180, 200), 2)
    cv2.imshow('retinex_MSRCR result', img_)
    cv2.waitKey(0)

def dehaze_dark_channel():
    PATH_IMG_FILE = 'tmp.jpg'
    src_img = cv2.imread(PATH_IMG_FILE)
    recover = dehazer()
    recovered = recover.dehaze(src_img)
    cv2.imshow('dehaze_dark_channel result', recovered)
    cv2.waitKey(0)


def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist


def equalHist():
    PATH_IMG_FILE = 'tmp.jpg'
    img = cv2.imread(PATH_IMG_FILE, 0)
    # 灰度图像矩阵的高、宽
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    cv2.imshow(" equalizeHist", equalHistImage)
    cv2.waitKey(0)

def gamma_trans(img,gamma):
    #具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    #实现映射用的是Opencv的查表函数
    return cv2.LUT(img,gamma_table)
	
def gamma():
    PATH_IMG_FILE = 'tmp.jpg'
    img = cv2.imread(PATH_IMG_FILE)
    (b,g,r) = cv2.split(img)
    img0 = cv2.merge([r, g, b])
    img0_corrted = gamma_trans(img0, 0.5)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    ax[0].imshow(img0)
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[1].imshow(img0_corrted)
    ax[1].axis('off')
    ax[1].set_title('Gamma')
    fig.tight_layout()
    fig.savefig('enhance_Gamma.jpg')
    plt.show()

def laplacian():
    PATH_IMG_FILE = 'tmp.jpg'
    img = cv2.imread(PATH_IMG_FILE)
    (b,g,r) = cv2.split(img)
    img = cv2.merge([r, g, b])
    kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])  # 定义卷积核
    imageEnhance = cv2.filter2D(img,-1, kernel)  # 进行卷积运算
	
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[1].imshow(imageEnhance)
    ax[1].axis('off')
    ax[1].set_title('Laplacian')
    fig.tight_layout()
    fig.savefig('enhance_laplacian.jpg')
    plt.show()
	


class retinex(object):
    def repair(self, img, sigma, type):
        if type == 0:
            return self.repair_SSR(img, sigma)
        if type == 1:
            return self.repair_MSR(img, sigma)
        if type == 2:
            return self.repair_MSRCR(img, sigma, 5, 25, 125, 46, 0.01, 0.8)

    def repair_SSR(self, img, sigma):
        # 单尺度
        # 其实感觉跟形态学顶帽差不多的意思
        temp = cv2.GaussianBlur(img, (0,0), sigma)
        gaussian = np.where(temp == 0, 0.01, temp)
        retinex = np.log10(img+0.01) - np.log10(gaussian)
        return retinex

    def repair_MSR(self, img, sigma_list):
        # 多尺度
        retinex = np.zeros_like(img*1.0)
        for sigma in sigma_list:
            retinex += self.repair_SSR(img, sigma)
        retinex = retinex / len(sigma_list)
        return retinex

    def repair_MSRCR(self, img, sigma_list, gain, offset, alpha, beta, low_clip, high_clip):
        # 带颜色恢复的多尺度
        img = np.float64(img) + 1.0
        img_msr = self.repair_MSR(img, sigma_list)
        img_color = self.color_restor(img, alpha, beta)
        img_msrcr = gain * (img_msr * img_color + offset)

        for ch in range(img_msrcr.shape[2]):
            img_msrcr[:,:,ch] = (img_msrcr[:,:,ch] - np.min(img_msrcr[:,:,ch])) / \
                                (np.max(img_msrcr[:,:,ch]) - np.min(img_msrcr[:,:,ch]))*255
        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        img_msrcr = self.color_balance(img_msrcr, low_clip, high_clip)
        return img_msrcr

 
    def color_restor(self, img, alpha, beta):
        img_sum = np.sum(img, axis=2, keepdims=True)
        color_res = beta * (np.log10(alpha*img) - np.log10(img_sum))
        return color_res


    def color_balance(self, img, low, high):
        area = img.shape[0]*img.shape[1]
        for ch in range(img.shape[2]):
            unique, counts = np.unique(img[:,:,ch], return_counts=True)
            current = 0
            low_val = 0
            high_val = 0
            for u, c in zip(unique, counts):
                if float(current) / area < low:
                    low_val = u
                if float(current) / area < high:
                    high_val = u
                current += c
            img[:,:,ch] = np.maximum(np.minimum(img[:,:,ch], high_val), low_val)
        return img


class dehazer(object):
    def __init__(self, win=9, ap=0.001, omiga=0.95, max_t=0.05):
        self._ap = ap
        self._omiga = omiga
        self._max_t = max_t
        self._win_size = win
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win,win))

    def dehaze(self, img):
        img_float = img.astype(np.float32)/255.0

        dark_img = self.dark_channel(img_float)
        atmos = self.atmospheric_light(img_float, dark_img)
        trans = self.transmission(img_float, atmos)
        trans = self.guide_filter(img_float, trans)
        trans = np.maximum(trans, self._max_t)
        dehazed = np.zeros(img.shape, img_float.dtype)
        for ch in range(3):
            dehazed[:,:,ch] = (img_float[:,:,ch] - atmos[0,ch]) / trans + atmos[0,ch]
        dehazed = np.maximum(dehazed, 0)
        dehazed = np.minimum(dehazed, 1)
        dehazed *= 255

        cv2.imshow('dark', dark_img)
        cv2.imshow('tran', trans)
        return dehazed.astype(np.uint8)

    def dark_channel(self, img, use_win=True):
        dark_img = img.min(axis=2)
        if use_win:
            dark_img = cv2.erode(dark_img, self._kernel)
        return dark_img

    def atmospheric_light(self, srcimg, darkimg):
        vec_sz = darkimg.shape[0]*darkimg.shape[1]
        dark_vec = darkimg.reshape(vec_sz,1)
        src_vec = srcimg.reshape(vec_sz, 3)

        num_ap = int(vec_sz*self._ap)
        indx = dark_vec[:,0].argsort()[vec_sz-num_ap:]

        atmos = np.zeros((1,3), dtype=np.float32)
        for i in range(num_ap):
            atmos += src_vec[indx[i]]
        atmos /= num_ap
        return atmos

    def transmission(self, srcimg, atmos):
        img_ = np.zeros(srcimg.shape, srcimg.dtype)
        for i in range(3):
            img_[:,:,i] = srcimg[:,:,i] / atmos[0,i]

        trans = 1 - self._omiga*self.dark_channel(img_, False)
        # trans = self.soft_matting(srcimg, trans)
        return trans

    def soft_matting(self, srcimg, trans):
        epsilon = 10**-8
        lambda_ = 10**-4
        
        window_size = 3
        num_window_pixels = window_size * window_size
        inv_num_window_pixels = 1.0 / num_window_pixels

        im_height = srcimg.shape[0]
        im_width  = srcimg.shape[1]
        num_image_pixels = im_height*im_width

        # matting_laplacian = np.zeros((num_image_pixels, num_image_pixels))
        matting_laplacian = sparse.lil_matrix((num_image_pixels, num_image_pixels))
        for row in range(window_size/2, im_height-window_size/2):
            for col in range(window_size/2, im_width-window_size/2):
                window_indice = (row-window_size/2, col-window_size/2, 
                                 row+window_size/2+1, col+window_size/2+1)

                window = srcimg[window_indice[0]:window_indice[2], 
                                window_indice[1]:window_indice[3],:]
                
                window_flat = window.reshape(num_window_pixels, 3)
                window_mean = np.mean(window_flat, 0).reshape(1,3) #1*3
                window_conv = np.cov(window_flat.T, bias=True) #3*3
                window_inv_cov = window_conv + (epsilon / num_window_pixels)*np.eye(3)
                window_inv_cov = np.linalg.inv(window_inv_cov)

                for sub_row_1 in range(window_indice[0], window_indice[2]):
                    for sub_col_1 in range(window_indice[1], window_indice[3]):
                        matting_laplace_row = sub_row_1*im_width + sub_col_1
                        
                        for sub_row_2 in range(window_indice[0], window_indice[2]):
                            for sub_col_2 in range(window_indice[1], window_indice[3]):
                                matting_laplace_col = sub_row_2*im_width + sub_col_2
                                
                                ker_delta = 0
                                if matting_laplace_row == matting_laplace_col:
                                    ker_delta = 1
                                
                                row_pixel_var = srcimg[sub_row_1, sub_col_1, :] - window_mean
                                col_pixel_var = srcimg[sub_row_2, sub_col_2, :] - window_mean
                                
                                matting_laplacian[matting_laplace_row, matting_laplace_col] += ker_delta - inv_num_window_pixels*(1+np.dot(np.dot(row_pixel_var,window_inv_cov),col_pixel_var.T))[0,0]

        trans_flat = trans.reshape(num_image_pixels, 1)
        matting_laplacian_inv = matting_laplacian + (lambda_*sparse.eye(num_image_pixels))
        matting_laplacian_inv = splina.inv(matting_laplacian_inv.tocsc())
        refined_trans_flat = np.dot(lambda_*trans_flat, matting_laplacian_inv)
        refined_trans = refined_trans_flat.reshape(im_height, im_width)

        return refined_trans

    def guide_filter(self, srcimg, trans, radius=11, eps=0.01):
        srcimg_gray = srcimg
        if len(srcimg.shape) == 3:
            srcimg_gray = cv2.cvtColor(srcimg, cv2.COLOR_BGR2GRAY) 

        mean_gui = cv2.boxFilter(srcimg_gray, -1, (radius,radius), normalize=True)
        mean_fil = cv2.boxFilter(trans, -1, (radius,radius), normalize=True)
        mean_gf  = cv2.boxFilter(trans*srcimg_gray, -1, (radius,radius), normalize=True)
            
        cov_gf   = mean_gf - mean_gui*mean_fil
        mean_gui_gui = cv2.boxFilter(srcimg_gray*srcimg_gray, -1, (radius,radius), normalize=True)
        var_gui = mean_gui_gui - mean_gui * mean_gui

        a = cov_gf / (var_gui + eps)
        b = mean_fil - a * mean_gui

        mean_a = cv2.boxFilter(a, -1, (radius,radius), normalize=True)
        mean_b = cv2.boxFilter(b, -1, (radius,radius), normalize=True)
        trans_ = mean_a * srcimg_gray + mean_b
        return trans_



frame_image_width = 796
frame_image_heigth = 600

window = tkinter.Tk()
window.title('图像增强系统')
window.geometry('1024x600')
main_frame = tkinter.Frame(window, width=1024, height=600, bg="lightblue")


# 图像展示
frame_image = tkinter.Frame(main_frame, bg="lightblue", width = frame_image_width, height = frame_image_heigth)

image = cv2.imread("01.jpg")
print(image.shape)
image_shape = image.shape
if(image_shape[0]/image_shape[1]>=image_shape[1]/image_shape[0]):
    resize_y = frame_image_heigth
    resize_x = int(image_shape[1]*resize_y/image_shape[0])
else:
    resize_x = frame_image_width
    resize_y = int(image_shape[0]*resize_x/image_shape[1])
print((resize_x,resize_y))
image_re = cv2.resize(image,(resize_x,resize_y))
img = Image.fromarray(cv2.cvtColor(image_re,cv2.COLOR_BGR2RGB))

photo = ImageTk.PhotoImage(img)
imglabel = tkinter.Label(frame_image, image=photo)
imglabel.pack(side="left")
frame_image.place(x=0,y=0)


# 功能区
# 功能区-操作区
label_x = 820
frame_fuction_button = tkinter.Frame(main_frame, width = 200, height = frame_image_heigth, bg="lightblue")
openButton = tkinter.Button(frame_fuction_button, text='打开图像', command = open_image, font=('Arial', 12,'bold'), width=18, height=1)
openButton.place(x=20,y=0)
takePhotoButton = tkinter.Button(frame_fuction_button, text='retinex-SSR', command = retinex_SSR, font=('Arial', 12,'bold'), width=18, height=1)
takePhotoButton.place(x=20,y=40)
processButton = tkinter.Button(frame_fuction_button, text='retinex-MSR', command = retinex_MSR, font=('Arial', 12,'bold'), width=18, height=1)
processButton.place(x=20,y=80)
processButton = tkinter.Button(frame_fuction_button, text='retinex-MSRCR', command = retinex_MSRCR, font=('Arial', 12,'bold'), width=18, height=1)
processButton.place(x=20,y=120)
processButton = tkinter.Button(frame_fuction_button, text='dehaze-dark-channel', command = dehaze_dark_channel, font=('Arial', 12,'bold'), width=18, height=1)
processButton.place(x=20,y=160)
processButton = tkinter.Button(frame_fuction_button, text=' equalizeHist', command = equalHist, font=('Arial', 12,'bold'), width=18, height=1)
processButton.place(x=20,y=200)
processButton = tkinter.Button(frame_fuction_button, text='gamma', command = gamma, font=('Arial', 12,'bold'), width=18, height=1)
processButton.place(x=20,y=240)
processButton = tkinter.Button(frame_fuction_button, text='laplacian', command = laplacian, font=('Arial', 12,'bold'), width=18, height=1)
processButton.place(x=20,y=280)



frame_fuction_button.place(x=800,y=50)



main_frame.place(x=0,y=0)



tkinter.mainloop()
