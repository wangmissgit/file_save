# -*- coding: utf-8 -*-
import cv2
import numpy as np

def detect(img_name, framename):
    img = cv2.imread(img_name)
    # cv2.imshow("img_ori",img)
    frame = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波
    rgb_list = np.array([165, 165, 165])
    lower_rgb = rgb_list - 50  # np.array([0, 100, 200])  # 设置图像掩模参数
    upper_rgb = rgb_list + 50  # np.array([70, 255, 255])  # 设置图像掩模参数
    extract_yellow = cv2.inRange(frame, lowerb=lower_rgb, upperb=upper_rgb)  # inRange API进行黄色颜色区域提取
    extract_yellow = cv2.medianBlur(extract_yellow, 15)  # 中值滤波
    # cv2.imshow("extract_yellow demo " + framename, extract_yellow)  # 显示黄色提取效果
    contours, hierarchy = cv2.findContours(extract_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv2.imshow("img "+ framename, img)


if __name__ == '__main__':
    # 165.8095238095238, 165.8095238095238, 165.8095238095238
    for i in range(19, 30):
        framename = str(i)
        img_name = "./pics/frame%s.jpg" % framename
        detect(img_name,str(i))
    cv2.waitKey(0)
