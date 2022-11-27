# encoding:utf-8

import sys
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import json
from matplotlib import pyplot as plt 

def show(img):
    cv2.imshow('img', img)
    cv2.waitKey()
    print('donw')

if __name__ == '__main__':
    img = cv2.imread('text_img.png')
    # 参数1 image：必选参数。表示原图像数组，即8位输入图像
    # 参数2 threshold1：必选参数。用于设置最小阈值
    # 参数3 threshold2：必选参数。用于设置最大阈值--用于进一步删选边缘信息
    # 最后决定哪些是边缘，哪些是真正的边，哪些不是边。Canny边缘检测会设置两个阈值，我们称为高阈值（MaxVal）和低阈值（MinVal）。当像素点的幅值超过高阈值时，该像素点被保留为边缘像素；当像素点的幅值小于低闽值时，该像素点被排除；当像素点位于低阈值和高阈值之间时，只有当像素点连接一个超过高阈值的像素时才被保留。 
    canny_img = cv2.Canny(img, 100, 250)
    cnts,hierarchy = cv2.findContours(canny_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('new_image', cv2.drawContours(canny_img,cnts,-1,(255,73,95),1))
    # cv2.waitKey()
    cv2.imshow('new_image1', cv2.drawContours(img,[cnts[2]],-1,(255,73,95),1))
    cv2.waitKey()
    area=0
    for item in [cnts[2]]:
        # print(1)
        # test
        # draw(img, [item])
        item_area = cv2.contourArea(item)
        # print(item_area)
        area += item_area
    w = img.shape[0]
    h = img.shape[1]
    print(area, 'done')
    print(w * h, 'done')
    print(area / (w * h), 'done')

    # show(canny_img)

    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # show(gray_img)
    # plt.plot(hist)
    # plt.show()

    # cv2.waitKey()
    # show(hist)

    # 使用opencv的函数cv2.calcHist(images, channels, mask, histSize, ranges):
    # 参数1：要计算的原图，以方括号的传入，如：[img]。
    # 参数2：类似前面提到的dims，灰度图写[0]就行，彩色图B/G/R分别传入[0]/[1]/[2]。
    # 参数3：要计算的区域ROI，计算整幅图的话，写None。
    # 参数4：也叫bins,子区段数目，如果我们统计0-255每个像素值，bins=256；如果划分区间，比如0-15, 16-31…240-255这样16个区间，bins=16。
    # 参数5：range,要计算的像素值范围，一般为[0,256)。

    # 计算roi直方图
    # roihist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # # 归一化，参数为原图像和输出图像，归一化后值全部在0到255范围
    # # cv2.NORM_MINMAX 对数组的所有值进行转化,使它们线性映射到最小值和最大值之  间
    # cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    # dst = cv2.calcBackProject([gray_img], [0,1],roihist, [0, 256], 1)

    # # 此处卷积可以把分散的点连在一起
    # disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # dst = cv2.filter2D(dst, -1, disc)
    # print(dst)

    # n, bins, patches = plt.hist(gray_img.ravel(), 256, [0, 256])
    # plt.close(1)
    # max = np.array(n).argmax(axis=0)
    # print(max)