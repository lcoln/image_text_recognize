# encoding:utf-8

import sys
import easyocr
import cv2
import matplotlib.pyplot as plot
import numpy as np
from PIL import Image, ImageDraw, ImageFont
list = []          ## 空列表

def detect(IMG_P):
    # reader = easyocr.Reader(['ch_tra', 'ja', "en"])
    reader = easyocr.Reader(['ch_tra', "en"])
    RST = reader.readtext(IMG_P)
    RST
    im = Image.open(IMG_P)
    im_copy = im.copy()
    # font = cv2.FONT_HERSHEY_SIMPLEX

    IMG = cv2.imread(IMG_P)
    spacer = 100
    for detection in RST:
        T_LEFT = tuple(detection[0][0])
        B_RIGHT = tuple(detection[0][2])
        TEXT = detection[1]
        # fontscale = abs(B_RIGHT[1] - T_LEFT[1])

        img_pil = Image.fromarray(cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB))

        fonttype = './msyh.ttf' #微软雅黑字体，和具体操作系统相关
        # fontscale = 30        #字体大小
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()
        fontscale = int(abs(T_LEFT[0] - B_RIGHT[0]) / len(TEXT))
        print(fontscale)
        font = ImageFont.truetype(fonttype, fontscale, encoding="unic")

        list.append([T_LEFT + B_RIGHT, TEXT, font, T_LEFT, B_RIGHT]) 




        # draw.text(T_LEFT, TEXT, font=font)  #PIL中BGR=(255,0,0)表示红色
        # img_ocv = np.array(img_pil)                     #PIL图片转换为numpy
        # IMG = cv2.cvtColor(img_ocv,cv2.COLOR_RGB2BGR)

        # IMG = cv2.rectangle(IMG,T_LEFT,B_RIGHT,(0,255,0),3)
        # # IMG = cv2.putText(IMG,TEXT,(20,spacer), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        # spacer+=15

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


def repair(imagePath, info):
    tmp_img = cv2.imread(imagePath)

    b = cv2.imread('cavity3.png', 0)
    src_img = tmp_img[info[1]:info[3], info[0]:info[2]]  # 裁剪坐标为[y0:y1, x0:x1]

    # cimg = merge_img(b, tmp_img, info[1], info[3], info[0], info[2])
    # dst = cv2.inpaint(cimg, b, 5, cv2.INPAINT_TELEA)
    dst = cv2.inpaint(src_img, b, 5, cv2.INPAINT_TELEA)
    # cv2.imshow('src_img', src_img)
    # cv2.imshow('b', b)

    tmp_img[info[1]:info[3], info[0]:info[2]] = dst
    cv2.imshow('tmp_img', tmp_img)
    return tmp_img
    # cv2.imwrite(f'repair_{path}', dst)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

def transform(imagePath, info):
    size = 3
    tmp_img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    # src_img = im.crop(info[0])

    src_img = tmp_img[info[0][1]:info[0][3], info[0][0]:info[0][2]]  # 裁剪坐标为[y0:y1, x0:x1]
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
    k = np.ones((15, 15), np.uint8)
    img2 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)  # 闭运算
    cv2.imwrite('cavity3.png', img2)

    tmp_img = repair(imagePath, info[0])
    write(tmp_img, info)


def add_alpha_channel(img):
    b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
 
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    return img_new

def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
        
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
 
    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]
 
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2,xx1:xx2,2] / 255.0
    alpha_jpg = 1 - alpha_png
    
    # 开始叠加
    for c in range(0,2):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg*jpg_img[y1:y2,x1:x2,c]) + (alpha_png*png_img[yy1:yy2,xx1:xx2,c]))
 
    return jpg_img

def write(IMG, info):

    img_pil = Image.fromarray(cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB))

    # fontscale = 30        #字体大小
    draw = ImageDraw.Draw(img_pil)

    # list.append([T_LEFT + B_RIGHT, TEXT, font, draw, T_LEFT]) 
    TEXT = info[1]
    font = info[2]
    T_LEFT = info[3]
    B_RIGHT = info[4]
    draw.text(T_LEFT, TEXT, font=font)  #PIL中BGR=(255,0,0)表示红色

    img_ocv = np.array(img_pil)                     #PIL图片转换为numpy
    IMG = cv2.cvtColor(img_ocv,cv2.COLOR_RGB2BGR)

    IMG = cv2.rectangle(IMG,T_LEFT,B_RIGHT,(0,255,0),3)

    cv2.imshow('IMG', IMG)


if __name__ == '__main__':
    # 读取文件
    imagePath = sys.argv[1]
    detect(imagePath)
    print(list)
    # im = Image.open(imagePath)
    tmp_img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)

    for it in list:
        transform(imagePath, it)
    cv2.waitKey()
    cv2.destroyAllWindows()