# encoding:utf-8

import sys
import easyocr
import cv2
import matplotlib.pyplot as plot
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyclipper

def detect(imagePath, lang):
    # 解析繁体/英文/数字
    reader = easyocr.Reader(lang)
    RST = reader.readtext(imagePath)
    
    origin_cv_image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)

    for detection in RST:
        # easyocr获取到文字区域的左上角坐标
        left_top_coordinate = tuple(detection[0][0])
        # easyocr获取到文字区域的右上角坐标
        right_bottom_coordinate = tuple(detection[0][2])
        # easyocr获取到的文字
        text = detection[1]
        #微软雅黑字体
        fontType = './msyh.ttf'         
        font = ImageFont.load_default()
        #字体大小
        fontScale = int(abs(left_top_coordinate[0] - right_bottom_coordinate[0]) / len(text))
        # 加载字体并定义其编码方式和大小
        font = ImageFont.truetype(fontType, fontScale, encoding="unic")

        origin_cv_image = put_text_into_image(
            origin_cv_image, 
            left_top_coordinate, 
            right_bottom_coordinate, 
            text, 
            font
        )

    return origin_cv_image


def replace_zeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

# 图像增强
def enhance(src_img, size):
    b_gray, g_gray, r_gray = cv2.split(src_img)
    result = []
    for it in [b_gray, g_gray, r_gray]:
        L_blur = cv2.GaussianBlur(it, (size, size), 0)
        img = replace_zeroes(it)
        L_blur = replace_zeroes(L_blur)

        dst_Img = cv2.log(img / 255.0)
        dst_Lblur = cv2.log(L_blur / 255.0)
        dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
        log_R = cv2.subtract(dst_Img, dst_IxL)

        dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
        log_uint8 = cv2.convertScaleAbs(dst_R)
        
        result.append(log_uint8)
    
    return cv2.merge(result)

# 修复图像
def repair(origin_cv_image, lt, rb):
    # 读取梯度运算后的图片
    gradient_png = cv2.imread('cavity_gradient.png', 0)

    # 裁剪原图坐标为[y0:y1, x0:x1]
    src_img = origin_cv_image[lt[1]:rb[1], lt[0]:rb[0]]  
    # 将通过梯度运算出来的图片与裁剪原图进行融合修复，去除掉梯度运算轨迹包括到的内容
    dst = cv2.inpaint(src_img, gradient_png, 5, cv2.INPAINT_TELEA)
    # 将修复完的图片块再放回去
    origin_cv_image[lt[1]:rb[1], lt[0]:rb[0]] = dst

    return origin_cv_image

def perimeter(poly):
    p = 0
    nums = poly.shape[0]
    for i in range(nums):
        p += abs(np.linalg.norm(poly[i % nums] - poly[(i + 1) % nums]))
    return p

def proportional_zoom_contour(contour, ratio):
    """
    多边形轮廓点按照比例进行缩放
    :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
    :param ratio: 缩放的比例，如果大于1是放大小于1是缩小
    :return:
    """
    poly = contour[:, 0, :]
    area_poly = abs(pyclipper.Area(poly))
    perimeter_poly = perimeter(poly)
    poly_s = []
    pco = pyclipper.PyclipperOffset()
    pco.MiterLimit = 10
    if perimeter_poly:
        d = area_poly * (1 - ratio * ratio) / perimeter_poly
        pco.AddPath(poly, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        poly_s = pco.Execute(-d)
    poly_s = np.array(poly_s).reshape(-1, 1, 2).astype(int)

    return poly_s

def get_text_color(originImg, cannyImg):

    # 指定范围为3*3的矩阵，kernel（卷积核核）指定为全为1的33矩阵，卷积计算后，该像素点的值等于以该像素点为中心的3*3范围内的最大值。
    # kernel = np.ones((3, 3),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))

    # 灰度
    gray_img = cv2.cvtColor(originImg, cv2.COLOR_BGR2GRAY)
    # 二值化
    _,RedThresh = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
    # 腐蚀，由于我们是二值图像，所以只要包含周围黑的部分，就变为黑的。
    # erode = cv2.erode(RedThresh,kernel)
    # 膨胀，由于我们是二值图像，所以只要包含周围白的部分，就变为白的。
    dilated = cv2.dilate(RedThresh,kernel)

    cnts,hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # ncnts = proportional_zoom_contour(np.array(cnts[0]), 1.1)
    if len(cnts)>=2 and (cnts[1] & cnts[1][0] & cnts[1][0][0]).any():
        point = list(cnts[1][0][0])
        # img=cv2.drawContours(originImg,[cnts[1]],-1,(0,255,0),1)
        # cv2.imshow('new_image', img)
        # cv2.waitKey()
    else:
        point = None
    if point:
        color = tuple(originImg[point[1], point[0]])
    else:
        color = (0,0,0)
    return color

def put_text_into_image(origin_cv_image, lt, rb, text, font):
    size = 3
    # 裁剪文本坐标，[y0:y1, x0:x1]
    text_img = origin_cv_image[lt[1]:rb[1], lt[0]:rb[0]] 
    # 裁剪文本坐标，[y0:y1, x0:x1]

    # todo, 5个思路
    # 1. 获取轮廓，缩小轮廓让其刚好经过线上的点，取点的颜色
    # 2. 获取轮廓，缩小轮廓让其刚好经过线上的点，取所有点的颜色的平均值
    # 3. 将原图进行腐蚀，获取轮廓，缩小轮廓让其刚好经过线上的点，取点的颜色
    # 4. 直方图，取2个波峰的值，一个是背景一个是字体颜色
    # 5. 取区域内平均色值
    color = get_text_color(text_img, cv2.Canny(text_img, 200, 150))

    # print(color)
    # 获取图片上文字颜色

    # 增强图片
    result = enhance(text_img, size)
    # print(color_name)
    cv2.imwrite('cavity_enhance.png', result)
    # 增强图片

    # 边缘检测
    img = cv2.imread('cavity_enhance.png')
    # 边缘检测

    canny_img = cv2.Canny(img, 200, 150)
    cv2.imwrite('cavity_canny.png', canny_img)
    # 边缘检测

    # 梯度运算
    img = cv2.imread('cavity_canny.png', 1)
    k = np.ones((15, 15), np.uint8)
    img2 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)  
    cv2.imwrite('cavity_gradient.png', img2)
    # 梯度运算

    # 图像修复，将通过梯度运算出来的图片与原始图像进行融合
    repair_cv_image = repair(origin_cv_image, lt, rb)
    # 图像修复

    # 在修复完的图片块上写入文本
    return write(repair_cv_image, lt, rb, text, font, color)
    # 在修复完的图片块上写入文本

def write(IMG, lt, rb, text, font, color):
    print(color)
    # 将图片的array转换成image
    img_pil = Image.fromarray(cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB))
    # 进行图像绘制
    draw = ImageDraw.Draw(img_pil)
    # 进行文本写入
    draw.text(lt, text, font=font, fill=color)  #PIL中BGR=(255,0,0)表示红色
    #PIL图片转换为numpy
    img_ocv = np.array(img_pil)                     
    IMG = cv2.cvtColor(img_ocv,cv2.COLOR_RGB2BGR)
    return cv2.rectangle(IMG, lt, rb, (0,255,0), 3)

if __name__ == '__main__':
    # 读取文件
    imagePath = sys.argv[1]
    new_image = detect(sys.argv[1], ['ch_sim', "en"])
    
    cv2.imshow('new_image', new_image)
    cv2.waitKey()
    cv2.destroyAllWindows()