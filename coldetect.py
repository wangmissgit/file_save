import cv2
import numpy as np
import time


def mouse_click(event, x, y, flags, para):
    global lsPointsChoose
    if event == cv2.EVENT_LBUTTONDOWN:
        lsPointsChoose.append([x, y])
        len_list = len(lsPointsChoose)
        if len_list%2==0 and len_list>1:
            print(lsPointsChoose)
            image_re = img[lsPointsChoose[-2][1]:lsPointsChoose[-1][1], lsPointsChoose[-2][0]:lsPointsChoose[-1][0], :]
            #cv2.imshow('image_re',image_re)
            r = np.mean(image_re[:,:,0])
            g = np.mean(image_re[:,:,1])
            b = np.mean(image_re[:,:,2])
            f_out = open("rgb.ini",'w')
            f_out.write(str(int(r))+'-'+str(int(g))+'-'+str(int(b)))
            f_out.close()
            print("rgb:", r, g, b)


if __name__ == '__main__':
    lsPointsChoose = []
    img = cv2.imread('./pics/frame19.jpg')
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_click)
    while True:
        cv2.imshow('img', img)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()
