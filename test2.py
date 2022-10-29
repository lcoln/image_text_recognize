# encoding:utf-8

import sys
import easyocr
import cv2
import matplotlib.pyplot as plot
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def detect(IMG_P):
    reader = easyocr.Reader(['ch_sim', "en"])
    RST = reader.readtext(IMG_P)
    RST

    # font = cv2.FONT_HERSHEY_SIMPLEX

    IMG = cv2.imread(IMG_P)
    spacer = 100
    for detection in RST:
        T_LEFT = tuple(detection[0][0])
        B_RIGHT = tuple(detection[0][2])
        TEXT = detection[1]
        # fontscale = abs(B_RIGHT[1] - T_LEFT[1])

        img_pil = Image.fromarray(cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB))

        fonttype = 'msyh.ttc' #微软雅黑字体，和具体操作系统相关
        # fontscale = 30        #字体大小
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()
        fontscale = int(abs(T_LEFT[0] - B_RIGHT[0]) / len(TEXT))
        print(fontscale)
        font = ImageFont.truetype("./simsun.ttc", fontscale, encoding="unic")
        draw.text(T_LEFT, TEXT, font=font)  #PIL中BGR=(255,0,0)表示红色
        img_ocv = np.array(img_pil)                     #PIL图片转换为numpy
        IMG = cv2.cvtColor(img_ocv,cv2.COLOR_RGB2BGR)

        IMG = cv2.rectangle(IMG,T_LEFT,B_RIGHT,(0,255,0),3)
        # IMG = cv2.putText(IMG,TEXT,(20,spacer), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        spacer+=15

    # plot.imshow(IMG)
    # plot.show()


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img / 255.0)
    dst_Lblur = cv2.log(L_blur / 255.0)
    dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


def repair(path):
    img = cv2.imread(path)

    b = cv2.imread('cavity3.png',0)
    dst = cv2.inpaint(img, b, 5, cv2.INPAINT_TELEA)
    cv2.imshow('img', img)
    cv2.imshow('b', b)
    cv2.imshow('dst', dst)
    cv2.imwrite(f'repair_{path}', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 读取文件
    imagePath = sys.argv[1]
    detect(imagePath)

    size = 3
    src_img = cv2.imread(imagePath)
    b_gray, g_gray, r_gray = cv2.split(src_img)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])
    
    # cv2.imshow('img', src_img)
    # cv2.imshow('aaa', result)
    cv2.imwrite('cavity1.png', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img = cv2.imread('cavity1.png', cv2.IMREAD_GRAYSCALE)
    canny_img = cv2.Canny(img, 200, 150)
    cv2.imwrite('cavity2.png', canny_img)

    img = cv2.imread('cavity2.png', 1)
    k = np.ones((10, 10), np.uint8)
    img2 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)  # 闭运算
    cv2.imwrite('cavity3.png', img2)

    repair(imagePath)