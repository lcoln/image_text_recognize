# encoding:utf-8

import sys
import easyocr
import cv2
import matplotlib.pyplot as plot
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# import pandas as pd
import colorsys
import imutils

# index = ["color", "color_name", "hex", "R", "G", "B"]
# csv_df = pd.read_csv('colors.csv', names=index, header=None)

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

# def get_color_name(r, g, b):
#     # print(r)
#     min_diff = 10000
#     color_name = ''
#     for i in range(len(csv_df)):
#         d = abs(r- int(csv_df.loc[i,"R"])) + abs(g- int(csv_df.loc[i,"G"]))+ abs(b- int(csv_df.loc[i,"B"]))
#         print(d)
#         if d <= min_diff:
#             min_diff = d
#             color_name = csv_df.loc[i,"color_name"]
#     return color_name


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
    # gradient_png1 = cv2.imread('cavity_gradient.png', cv2.IMREAD_UNCHANGED)

    # 裁剪原图坐标为[y0:y1, x0:x1]
    src_img = origin_cv_image[lt[1]:rb[1], lt[0]:rb[0]]  
    # 将通过梯度运算出来的图片与裁剪原图进行融合修复，去除掉梯度运算轨迹包括到的内容
    dst = cv2.inpaint(src_img, gradient_png, 5, cv2.INPAINT_TELEA)
    # dst1 = cv2.inpaint(gradient_png1, src_img, 5, cv2.INPAINT_TELEA)
    # 将修复完的图片块再放回去
    origin_cv_image[lt[1]:rb[1], lt[0]:rb[0]] = dst

    # cv2.imshow('232', dst1)
    # cv2.waitKey()
    # cv2.destroyWindow()
    return origin_cv_image

def get_color(img):
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    imgShape = img.shape
    imgHeightHalf = int(imgShape[0] / 2)
    imgWidthHalf = int(imgShape[1] / 2)
    BGR = np.array([
        img[imgHeightHalf][imgWidthHalf][0], 
        img[imgHeightHalf][imgWidthHalf][1], 
        img[imgHeightHalf][imgWidthHalf][2]
    ])
    return tuple(BGR)

def get_dominant_colors(infile, lt, rb):
    max_score = 0
    dominant_color = None

    image = Image.open(infile)
    # (left, upper, right, lower)
    image = image.crop((lt[0], lt[1], rb[0], rb[1])) 
    
    # cv2.imshow('image', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # 缩小图片，否则计算机压力太大
    # small_image = image.resize((80, 80))
    result = image.convert('RGBA')  
    result.thumbnail((200, 200))
	
    for count, (r, g, b, a) in result.getcolors(result.size[0] * result.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue

        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]

        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

        y = (y - 16.0) / (235 - 16)

        # 忽略高亮色
        if y > 0.9:
            continue

        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

    return dominant_color
	# 10个主要颜色的图像


def compute(img):
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    per_image_Rmean.append(np.mean(img[:,:,0]))
    per_image_Gmean.append(np.mean(img[:,:,1]))
    per_image_Bmean.append(np.mean(img[:,:,2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return int(R_mean), int(G_mean), int(B_mean)

def put_text_into_image(origin_cv_image, lt, rb, text, font):
    size = 3
    # print(lt, rb)
    # 裁剪文本坐标，[y0:y1, x0:x1]
    text_img = origin_cv_image[lt[1]:rb[1], lt[0]:rb[0]] 
    # 裁剪文本坐标，[y0:y1, x0:x1]

    # todo
    # 获取图片上文字颜色
    color = get_color(text_img)
    # color = get_dominant_colors('cavity_enhance.png', lt, rb)
    # print(color)
    # 获取图片上文字颜色

    # 增强图片
    result = enhance(text_img, size)
    # print(color_name)
    cv2.imwrite('cavity_enhance.png', result)
    # 增强图片



    # 边缘检测
    img = cv2.imread('cavity_enhance.png')
    # img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # retv,thresh = cv2.threshold(img_gray,125,255,1)
    # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # length = len(contours[0])
    # # [x0, y0]
    # co1 = contours[0][0][0]
    # # [x1, y1]
    # co2 = contours[0][length - 1][0]
    # # 裁剪原图坐标为[y0:y1, x0:x1]
    # i = img[co1[1]:co2[1], co1[0]:co2[0]]  
    # print(i, 11111)


    resized = imutils.resize(img, width=900)
    ratio = img.shape[0] / float(resized.shape[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # if you want cv2.contourArea >1, you can just comment line bellow
    cnts = np.array(cnts)[[cv2.contourArea(c)>10 for c in cnts]]
    grains = [np.int0(cv2.boxPoints(cv2.minAreaRect(c))) for c in cnts]
    centroids =[(grain[2][1]-(grain[2][1]-grain[0][1])//2, grain[2][0]-(grain[2][0]-grain[0][0])//2) for grain in grains]

    colors = [resized[centroid] for centroid in centroids]
    r = []
    g = []
    b = []
    for it in colors:
        r.append(it[0])
        g.append(it[1])
        b.append(it[2])

    # color = compute(img)
    color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))
    print(color)


    # hist_full = cv2.calcHist([img],[1],None,[256],[0,256])
    # # cv2.imshow('img_hist', hist_full)
    # plot.imshow(hist_full)
    # plot.show()
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    canny_img = cv2.Canny(img, 200, 150)
    cv2.imwrite('cavity_canny.png', canny_img)
    # 边缘检测

    # 梯度运算
    img = cv2.imread('cavity_canny.png', 1)
    k = np.ones((15, 15), np.uint8)
    img2 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)  
    cv2.imwrite('cavity_gradient.png', img2)
    # 梯度运算

    # img3 = cv2.imread('cavity_enhance.png')
    # img3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    # retv,thresh = cv2.threshold(img3,125,255,1)
    # cnts,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.drawContours(img3,cnts,-1,(100,100,100),3,lineType=cv2.LINE_AA)
    # cnts = np.array(cnts)[[cv2.contourArea(c)>10 for c in cnts]]
    # grains = [np.int0(cv2.boxPoints(cv2.minAreaRect(c))) for c in cnts]
    # centroids =[(grain[2][1]-(grain[2][1]-grain[0][1])//2, grain[2][0]-(grain[2][0]-grain[0][0])//2) for grain in grains]
    # resized = imutils.resize(img3, width=900)

    # colors = [resized[centroid] for centroid in centroids]
    # r = []
    # g = []
    # b = []
    # for it in colors:
    #     r.append(it[0])
    #     g.append(it[1])
    #     b.append(it[2])
    
    # print(np.mean(r), 1111)
    # print(np.mean(g), 2222)
    # print(np.mean(b), 3333)


    # cv2.imshow('new_image', img3)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # 图像修复，将通过梯度运算出来的图片与原始图像进行融合
    repair_cv_image = repair(origin_cv_image, lt, rb)
    # 图像修复

    # 在修复完的图片块上写入文本
    return write(repair_cv_image, lt, rb, text, font, color)
    # 在修复完的图片块上写入文本

def add_alpha_channel(img):
    b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
 
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    return img_new

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